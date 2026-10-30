#!/usr/bin/env python3
# =============================================================================
#  attnbreaker_core.py    AttentionBreaker / GenBFA on the selective code's own INT8 tensors
# -----------------------------------------------------------------------------
#  Developed By ANON, ANON, ECE Department
#
#  WHAT THIS IS
#  ------------
#  A faithful re-implementation of AttentionBreaker (Das et al., "GenBFA: An
#  Evolutionary Optimization Approach to Bit-Flip Attacks on LLMs", 2024,
#  github.com/TIES-Lab/attnbreaker) that runs on the same `layers` dict that
#  ecc_llm_mmlu.py builds, i.e. on the exact INT8 integers that the selective code's BCH
#  parity and STDM lattice see. The public code runs on bitsandbytes
#  LLM.int8() tensors; here the integers are the selective code's fake-quant ints, so a
#  flip lands on the same bit the ECC protects, with no write-back mismatch.
#
#  THE ALGORITHM (as in the public repo)
#  ------------------------------------
#    loss      : causal-LM cross-entropy over 5-shot MMLU prompts (labels =
#                input_ids), a small batch, one forward pass. Attack success
#                is "loss above a threshold" (7 in the paper), not accuracy.
#    gradients : one backward pass of the same loss, |dL/dw| per weight.
#    score     : importance(w,g,alpha) = alpha*minmax(|g|) + (1-alpha)*minmax(|w|)
#                per layer (alpha = 0.5 by default).
#    flip      : bit 7 of the INT8 weight, i.e. the sign bit. Only that bit.
#    layer rank: for each attn/mlp weight layer, flip the top p% by score,
#                measure the loss, restore. Keep layers whose loss exceeds a
#                layer threshold (4) with the fewest flips; the top one seeds
#                the search.
#    pop space : top 2% (by score) of the seed layer, sorted by score.
#    GenBFA    : population of index subsets. fitness = sign(loss >= thr) *
#                loss / |subset|. Tournament selection, uniform crossover,
#                mutation = drop each gene with prob U{1..9}%. The best subset
#                is the smallest one that keeps the loss above the threshold.
#
#  WHAT IT RETURNS
#  ---------------
#  A solution = list of (layer_name, flat_index) whose sign bits are flipped.
#  The driver (ecc_llm_attnbreaker.py) turns that into attacked ints, runs
#  the selective code correction, and measures MMLU accuracy protected vs unprotected.
# =============================================================================
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple, Callable, Optional

import numpy as np
import torch


# (layer_name, flat_index) flips bit 7; an optional third element names another plane
Solution = List[Tuple]


# ---------------------------------------------------------------------------
# INT8 helpers (same conventions as ecc_llm_mmlu.py: q is int32 holding int8)
# ---------------------------------------------------------------------------
def flip_sign_bits(q: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Return a copy of q with bit 7 (sign) flipped at the flat indices idx."""
    out = q.reshape(-1).copy()
    u = (out[idx].astype(np.int32) & 0xFF) ^ 0x80
    v = u.astype(np.int32)
    v[v >= 128] -= 256
    out[idx] = v
    return out.reshape(q.shape)


def flip_bits(q: np.ndarray, idx: np.ndarray, plane: int) -> np.ndarray:
    """Return a copy of q with bit `plane` flipped at the flat indices idx."""
    if plane == 7:
        return flip_sign_bits(q, idx)
    out = q.reshape(-1).copy()
    u = (out[idx].astype(np.int32) & 0xFF) ^ (1 << plane)
    v = u.astype(np.int32)
    v[v >= 128] -= 256
    out[idx] = v
    return out.reshape(q.shape)


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-8)


def importance_score(w_abs: np.ndarray, g_abs: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """AttentionBreaker's alpha-blend of normalized |w| and |g| (their SSCORE)."""
    return alpha * minmax(g_abs) + (1.0 - alpha) * minmax(w_abs)


# ---------------------------------------------------------------------------
# MMLU loss batches (their mmlu_loss: 5-shot prompts, LM loss on the prompt)
# ---------------------------------------------------------------------------
def build_loss_batches(tok, rows, n_prompts=4, batch_size=4, device="cuda",
                       max_len=2048):
    """rows: list of (prefix, question, choices, answer) from ecc_llm_mmlu.load_mmlu.
    Returns a list of dicts {input_ids, attention_mask, labels} on `device`.
    labels = input_ids with padding masked, exactly the public code's objective."""
    from ecc_llm_mmlu import fmt
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [fmt(p, q, c) for (p, q, c, _a) in rows[:n_prompts]]
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        out.append({"input_ids": enc["input_ids"].to(device),
                    "attention_mask": enc["attention_mask"].to(device),
                    "labels": labels.to(device)})
    return out


@torch.no_grad()
def lm_loss(model, batches) -> float:
    model.eval()
    tot = 0.0
    for b in batches:
        l = model(**b).loss.item()
        if not np.isfinite(l):
            l = 50.0
        tot += l
    return tot / max(len(batches), 1)


# ---------------------------------------------------------------------------
class AttentionBreaker:
    """AttentionBreaker on the selective code's `layers` dict.

    layers[name] = [q_int32, scale, shape, module]   (ecc_llm_mmlu format)
    write(name, q_int)                                writes ints into the model
    loss_fn()                                         returns the attack loss on
                                                      the CURRENT model weights
    The driver may hand in a loss_fn that first applies the selective code correction
    (adaptive attacker): then only flips that survive BCH count.
    """

    def __init__(self, model, layers: Dict[str, list], write: Callable,
                 clean: Dict[str, np.ndarray], batches, device="cuda",
                 alpha: float = 0.5, seed: int = 0,
                 loss_fn: Optional[Callable[[], float]] = None,
                 layer_filter: Sequence[str] = ("attn", "mlp")):
        self.model = model
        self.layers = layers
        self.write = write
        self.clean = clean
        self.batches = batches
        self.device = device
        self.alpha = float(alpha)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.loss_fn = loss_fn or (lambda: lm_loss(self.model, self.batches))
        self.cands = [n for n in layers if any(f in n for f in layer_filter)]
        self.grads: Dict[str, np.ndarray] = {}
        self.scores: Dict[str, np.ndarray] = {}
        self.n_loss_evals = 0

    # -- gradients (one backward pass on the first batch) -------------------
    def compute_gradients(self, layer_names: Optional[Sequence[str]] = None):
        names = list(layer_names) if layer_names else self.cands
        mods = {n: self.layers[n][3] for n in names}
        for m in mods.values():
            m.weight.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        self.model.eval()
        b = self.batches[0]
        loss = self.model(**b).loss
        loss.backward()
        for n, m in mods.items():
            g = m.weight.grad
            self.grads[n] = (np.zeros(m.weight.numel(), np.float32) if g is None
                             else g.detach().float().cpu().numpy().reshape(-1))
            m.weight.grad = None
            m.weight.requires_grad_(False)
        self.model.zero_grad(set_to_none=True)
        # importance scores on the dequantized float weights (what they score on)
        for n in names:
            q, s, shp, _ = self.layers[n]
            w_abs = np.abs(self._dequant(q, s, shp)).reshape(-1)
            self.scores[n] = importance_score(w_abs, np.abs(self.grads[n]), self.alpha)
        return self.grads

    @staticmethod
    def _dequant(q, s, shp):
        if isinstance(s, (float, int)):
            return q.astype(np.float32).reshape(shp) * float(s)
        sc = np.asarray(s, np.float32)
        while sc.ndim < len(shp):
            sc = sc[..., None]
        return q.astype(np.float32).reshape(shp) * sc

    # -- evaluate a solution: flip, measure loss, restore ------------------
    def _group(self, sol: Solution) -> Dict[str, np.ndarray]:
        """{layer: flat indices}, one entry per bit error (a weight hit on two
        planes appears twice, so per-codeword counts are bit-error counts)."""
        g: Dict[str, list] = {}
        for e in sol:
            g.setdefault(e[0], []).append(int(e[1]))
        return {n: np.asarray(v, np.int64) for n, v in g.items()}

    def _group_planes(self, sol: Solution) -> Dict[str, Dict[int, np.ndarray]]:
        g: Dict[str, Dict[int, list]] = {}
        for e in sol:
            plane = int(e[2]) if len(e) > 2 else 7
            g.setdefault(e[0], {}).setdefault(plane, []).append(int(e[1]))
        return {n: {p: np.unique(np.asarray(v, np.int64)) for p, v in d.items()}
                for n, d in g.items()}

    def _apply(self, name: str, planes: Dict[int, np.ndarray]) -> np.ndarray:
        q = self.clean[name]
        for p, idx in planes.items():
            q = flip_bits(q, idx, p)
        return q

    def attacked_ints(self, sol: Solution) -> Dict[str, np.ndarray]:
        att = {n: self.clean[n] for n in self.layers}
        for n, planes in self._group_planes(sol).items():
            att[n] = self._apply(n, planes)
        return att

    def loss_of(self, sol: Solution) -> float:
        grp = self._group_planes(sol)
        for n, planes in grp.items():
            self.write(n, self._apply(n, planes))
        l = self.loss_fn()
        self.n_loss_evals += 1
        for n in grp:
            self.write(n, self.clean[n])
        return l

    # -- stage 1: layer sensitivity (flip top p% by score per layer) -------
    def layer_sensitivity(self, pcts=(0.001, 0.01, 0.1, 1.0, 5.0),
                          layer_thresh: float = 4.0, verbose=True):
        """Returns rows (layer, pct, k, loss) and the ranked seed layers:
        layers whose loss >= layer_thresh at their smallest such k, sorted by
        fewest flips then highest loss (the public repo's selection rule)."""
        rows = []
        base = self.loss_fn()
        if verbose:
            print(f"   [ab] clean attack-loss = {base:.4f}", flush=True)
        for n in self.cands:
            sc = self.scores[n]; N = sc.size
            order = np.argsort(-sc)
            for p in pcts:
                k = int(N * p / 100.0)
                if k < 1:
                    continue
                l = self.loss_of([(n, int(i)) for i in order[:k]])
                rows.append((n, p, k, l))
                if verbose:
                    print(f"   [ab] {n:55s} top {p:>6g}% ({k:>8,} flips)  loss={l:.3f}",
                          flush=True)
        # seed layers: first k per layer that crosses the threshold
        best = {}
        for (n, p, k, l) in rows:
            if l >= layer_thresh and (n not in best or k < best[n][0]):
                best[n] = (k, l, p)
        ranked = sorted(best.items(), key=lambda kv: (kv[1][0], -kv[1][1]))
        seeds = [n for n, _ in ranked]
        return rows, seeds, ranked

    # -- stage 2: population space (top pct of seed layer(s) by score) ----
    def population_space(self, seed_layers: Sequence[str], pct: float = 2.0
                         ) -> List[Tuple[str, int, float]]:
        pop = []
        for n in seed_layers:
            sc = self.scores[n]
            k = max(1, int(sc.size * pct / 100.0))
            top = np.argpartition(-sc, k - 1)[:k]
            pop += [(n, int(i), float(sc[i])) for i in top]
        pop.sort(key=lambda t: -t[2])
        return pop

    # -- stage 3: GenBFA evolutionary optimization -------------------------
    def _mutate(self, sol: Solution) -> Solution:
        r = self.rng.randrange(1, 10) / 100.0          # their U{1..9}% drop rate
        out = [g for g in sol if self.rng.random() > r]
        return out if out else sol[:1]

    def _crossover(self, p1: Solution, p2: Solution, prob=0.9) -> Solution:
        if self.rng.random() < prob:
            child = [g1 if self.rng.random() < 0.5 else g2 for g1, g2 in zip(p1, p2)]
            return child if child else p1[:1]
        return self._mutate(p1)

    def _tournament(self, sols):
        pool = []
        while len(pool) < 2:
            i, j = self.rng.sample(range(len(sols)), 2)
            pool.append(sols[i] if sols[i][0] > sols[j][0] else sols[j])
        return pool

    def genbfa(self, pop_space, loss_thresh: float = 7.0, gens: int = 150,
               init_sols: int = 50, per_gen: int = 40, patience: int = 0,
               verbose=True):
        """Returns (best_solution, history) where history rows are
        (generation, n_flips, loss, fitness)."""
        space: Solution = [(n, i) for (n, i, _s) in pop_space]
        sols = [[0.0, list(space), 0.0]]
        for _ in range(init_sols):
            sols.append([0.0, self._mutate(space), 0.0])
        history = []
        best = sols[0]
        stall = 0; last_len = None
        for gen in range(gens):
            for s in sols:
                l = self.loss_of(s[1])
                f = (1.0 if l >= loss_thresh else -1.0) * (l / max(len(s[1]), 1))
                s[0], s[2] = f, l
            sols.sort(key=lambda s: -s[0])
            best = sols[0]
            history.append((gen, len(best[1]), best[2], best[0]))
            if verbose:
                print(f"   [genbfa] gen {gen:3d}  best: {len(best[1]):6d} flips  "
                      f"loss={best[2]:.3f}  fit={best[0]:.4f}  "
                      f"(evals so far {self.n_loss_evals})", flush=True)
            if patience:
                stall = stall + 1 if last_len == len(best[1]) else 0
                last_len = len(best[1])
                if stall >= patience:
                    if verbose:
                        print(f"   [genbfa] no change for {patience} gens, stopping")
                    break
            nxt = [best, [0.0, self._mutate(best[1]), 0.0]]
            while len(nxt) < per_gen:
                p1, p2 = self._tournament(sols)
                parent = p1[1] if self.rng.random() > 0.5 else p2[1]
                child = self._mutate(self._crossover(best[1], parent))
                nxt.append([0.0, child, 0.0])
            sols = nxt
        return best[1], history
