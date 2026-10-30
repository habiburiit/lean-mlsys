#!/usr/bin/env python3
# =============================================================================
#  pbs_core.py    Progressive Bit Search (BFA) with sensitivity-band targeting
# -----------------------------------------------------------------------------
#  Developed By ANON, ANON, ECE Department
#
#  WHY THIS EXISTS
#  ---------------
#  The adversarial injector in ecc_engine.corrupt_adversarial flips t+1 bits in
#  RANDOMLY chosen codewords. That is enough to defeat the BCH code, but the
#  bits themselves are harmless: 13 random bits out of 36.6M leave accuracy
#  untouched, so a reviewer can fairly say the attack was never dangerous.
#
#  Progressive Bit Search (Rakin, He, Fan, ICCV 2019) is the real thing. It uses
#  gradients to FIND the most damaging bits, and published results destroy a
#  network with on the order of ten flips. The comparison that matters is:
#
#      13 random bits    -> accuracy unchanged, code defeated, hash detects
#      13 PBS bits       -> accuracy destroyed, code defeated, hash detects
#
#  Same detection either way, because the digest fires on a single flipped bit
#  regardless of how that bit was chosen. That is the point: correction depends
#  on how many bits and where, detection does not.
#
#  BAND TARGETING
#  --------------
#  --pbs-band restricts the search to one slice of the sensitivity ranking:
#      top   the most fragile third of the protected layers
#      mid   the middle third
#      low   the least fragile third
#      all   no restriction (standard PBS)
#      carrier  the unprotected carrier layers
#  This answers a question the flat 40% coverage cannot: does the attacker have
#  to hit the layers we protect most heavily, or can equal damage be done in the
#  layers we protect lightly? If mid-band PBS is nearly as effective as top-band,
#  a single uniform coverage target is the wrong design and the budget should be
#  tiered.
#
#  ALGORITHM (per flip)
#  --------------------
#    1. forward/backward on a batch, get dL/dw for every weight
#    2. for each candidate weight and each protected bit plane j, the flip
#       changes the stored integer by delta_q = (q XOR 2^j) - q, so the float
#       weight moves by delta_w = delta_q * scale
#    3. first-order loss change is grad . delta_w; rank all candidates by it
#    4. take the top-k candidates, actually apply each, measure the true loss,
#       keep the best one (the gradient is only a linear approximation, so this
#       verification step matters)
#    5. repeat. The gradient is recomputed after every flip, which is what makes
#       the search "progressive" rather than a one-shot ranking.
# =============================================================================
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
def _bands(ranked: Sequence[str], band: str) -> List[str]:
    """Slice the sensitivity-ranked layer list into thirds."""
    ranked = list(ranked)
    n = len(ranked)
    if band == "all" or n == 0:
        return ranked
    a, b = n // 3, (2 * n) // 3
    if band == "top":
        return ranked[:max(a, 1)]
    if band == "mid":
        return ranked[a:max(b, a + 1)]
    if band == "low":
        return ranked[b:] or ranked[-1:]
    return ranked


def _toggle(v, j: int) -> int:
    """Flip bit j of a signed 8-bit value, staying in signed range."""
    return int(np.int8(np.uint8(int(v) & 0xFF) ^ np.uint8(1 << j)))


def _flip_delta(q: np.ndarray, j: int, b: int = 8) -> np.ndarray:
    """Change in the signed integer value when bit plane j is flipped.

    Two's complement, so flipping the sign bit is not simply +/-2^j. Compute it
    by actually toggling the bit in the unsigned byte view and re-interpreting.
    """
    u = (q.astype(np.int32) & 0xFF).astype(np.uint8)
    v = (u ^ np.uint8(1 << j)).astype(np.int8).astype(np.int32)
    return v - u.astype(np.int8).astype(np.int32)


# ---------------------------------------------------------------------------
class PBSAttack:
    """Progressive Bit Search over the top-K bit planes of chosen layers.

    Parameters
    ----------
    model, qinfo : the network and the {name: (q, scale, shape)} dict
    handles : {name: nn.Module} so the attacked weights can be written back
    K : protected bit planes (only these are attacked, since only these are
        what the ECC defends and what the digest covers)
    topk : candidates verified by true loss at each step
    """

    def __init__(self, model, qinfo, handles, K: int = 4, bits: int = 8,
                 topk: int = 20, device="cuda", all_planes: bool = False):
        self.model = model
        self.qinfo = qinfo
        self.handles = handles
        self.K = int(K)
        self.b = int(bits)
        self.topk = int(topk)
        self.device = device
        # all_planes lets PBS search every bit plane, not just the protected
        # top-K. Published BFA searches all 8. Flips below plane b-K are outside
        # what the ECC repairs and what the digest covers, so they are neither
        # correctable nor detectable here -- worth measuring precisely for that
        # reason: it bounds what the scheme can and cannot see.
        self.all_planes = bool(all_planes)
        self.log: List[dict] = []

    # -- helpers ------------------------------------------------------------
    def _scale_arr(self, name, shape):
        _, sinfo, _ = self.qinfo[name]
        kind, sc = sinfo
        if kind == "per_tensor":
            return np.full(shape, float(sc), np.float32)
        sc = np.asarray(sc, np.float32)
        while sc.ndim < len(shape):
            sc = sc[..., None]
        return np.broadcast_to(sc, shape).astype(np.float32)

    def _write_back(self, q_cur: Dict[str, np.ndarray]):
        with torch.no_grad():
            for name, q in q_cur.items():
                if name not in self.handles:
                    continue
                s = self._scale_arr(name, self.qinfo[name][2])
                w = torch.from_numpy((q.astype(np.float32) * s)).to(self.device)
                self.handles[name].weight.copy_(w.reshape(
                    self.handles[name].weight.shape))

    def _loss_and_grad(self, x, y, need_grad=True):
        self.model.zero_grad(set_to_none=True)
        crit = nn.CrossEntropyLoss()
        out = self.model(x)
        loss = crit(out, y)
        if need_grad:
            loss.backward()
        return float(loss.item())

    # -- main ---------------------------------------------------------------
    def run(self, batches, target_layers: Sequence[str], n_flips: int = 13,
            verbose: bool = True,
            weight_mask: Optional[Dict[str, np.ndarray]] = None
            ) -> Dict[str, np.ndarray]:
        """Flip n_flips bits, one at a time, recomputing gradients each step.

        `batches` is a list of (x, y) already on the device. Returns the
        attacked integer weights, same layout as qinfo.
        """
        target_layers = [n for n in target_layers if n in self.qinfo]
        if not target_layers:
            raise ValueError("no target layers in qinfo")
        q_cur = {n: self.qinfo[n][0].copy() for n in self.qinfo}
        self._write_back(q_cur)

        planes = (list(range(self.b)) if self.all_planes
                  else list(range(self.b - self.K, self.b)))

        for step in range(n_flips):
            # ---- 1. gradients ------------------------------------------
            for x, y in batches[:1]:
                self._loss_and_grad(x, y, need_grad=True)
            grads = {}
            for name in target_layers:
                mod = self.handles.get(name)
                if mod is None or mod.weight.grad is None:
                    continue
                grads[name] = mod.weight.grad.detach().cpu().numpy(
                    ).astype(np.float32).reshape(-1)
            if not grads:
                break

            # ---- 2+3. first-order score for every (weight, plane) -------
            cands = []
            for name, g in grads.items():
                q = q_cur[name].reshape(-1)
                s = self._scale_arr(name, self.qinfo[name][2]).reshape(-1)
                m = None
                if weight_mask is not None:
                    m = weight_mask.get(name)
                    if m is None or not m.any():
                        continue
                    m = m.reshape(-1)
                for j in planes:
                    dq = _flip_delta(q, j, self.b)
                    score = g * (dq.astype(np.float32) * s)   # dL ~ grad.dw
                    if m is not None:
                        score = np.where(m, score, -np.inf)   # bin restriction
                    k = min(self.topk, score.size)
                    idx = np.argpartition(-score, k - 1)[:k]
                    for i in idx:
                        cands.append((float(score[i]), name, int(i), j))
            if not cands:
                break
            cands.sort(key=lambda c: -c[0])
            cands = cands[:self.topk]

            # ---- 4. verify each candidate against the true loss ---------
            best = None
            for sc, name, i, j in cands:
                q_try = q_cur[name].reshape(-1).copy()
                orig = q_try[i]
                q_try[i] = _toggle(orig, j)
                saved = q_cur[name]
                q_cur[name] = q_try.reshape(self.qinfo[name][2])
                self._write_back({name: q_cur[name]})
                with torch.no_grad():
                    tot = 0.0
                    for x, y in batches:
                        tot += self._loss_and_grad(x, y, need_grad=False)
                    tot /= max(len(batches), 1)
                q_cur[name] = saved
                self._write_back({name: q_cur[name]})
                if best is None or tot > best[0]:
                    best = (tot, name, i, j, int(orig))
            if best is None:
                break

            # ---- 5. commit the winner ----------------------------------
            loss, name, i, j, orig = best
            flat = q_cur[name].reshape(-1)
            flat[i] = _toggle(orig, j)
            q_cur[name] = flat.reshape(self.qinfo[name][2])
            self._write_back({name: q_cur[name]})
            self.log.append({"step": step + 1, "layer": name, "index": i,
                             "plane": j, "loss": loss})
            if verbose:
                print(f"   [pbs] flip {step+1:3d}/{n_flips}  {name}"
                      f"  idx={i}  plane={j}  loss={loss:.4f}", flush=True)

        return q_cur


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  T-BFA : Targeted Bit-Flip Adversarial Weight Attack
# ---------------------------------------------------------------------------
#  Rakin, He, Fan, "T-BFA: Targeted Bit-Flip Adversarial Weight Attack", 2021.
#
#  PBS destroys the model for everyone, so top-1 accuracy collapses and anyone
#  benchmarking would notice. T-BFA is surgical: it forces one SOURCE class to
#  be predicted as a chosen TARGET class while leaving accuracy on every other
#  class essentially untouched. Overall top-1 barely moves.
#
#  That makes it the sharpest possible case for the digest. Accuracy is blind
#  here by construction, not by luck: the model benchmarks normally and is
#  fully compromised. The hash does not look at behaviour, it looks at bits, so
#  it fires exactly as it would for any other flip.
#
#  The objective differs from PBS in two ways:
#    * the loss is computed ONLY on source-class samples, relabelled as the
#      target class, so the gradient points at "make source look like target"
#      instead of "make everything wrong";
#    * a penalty term keeps the loss on other classes from rising, which is
#      what preserves overall accuracy and keeps the attack quiet.
# ---------------------------------------------------------------------------
class TBFAAttack(PBSAttack):
    """Targeted variant: force source class -> target class, quietly."""

    def __init__(self, *a, source: int = 0, target: int = 1,
                 stealth: float = 1.0, **kw):
        super().__init__(*a, **kw)
        self.source = int(source)
        self.target = int(target)
        self.stealth = float(stealth)   # weight on the "keep the rest intact" term

    def _tbfa_loss(self, src_batches, other_batches) -> float:
        """L = CE(source samples -> target label) + stealth * CE(others -> true)."""
        crit = nn.CrossEntropyLoss()
        tot, n = 0.0, 0
        for x, _ in src_batches:
            if x.numel() == 0:
                continue
            yt = torch.full((x.shape[0],), self.target, dtype=torch.long,
                            device=x.device)
            tot += float(crit(self.model(x), yt).item()); n += 1
        src = tot / max(n, 1)
        tot, n = 0.0, 0
        for x, y in other_batches:
            if x.numel() == 0:
                continue
            tot += float(crit(self.model(x), y).item()); n += 1
        oth = tot / max(n, 1)
        # minimise source-to-target CE, and do not let the rest degrade
        return -(src) - self.stealth * (oth)

    def run_targeted(self, src_batches, other_batches,
                     target_layers: Sequence[str], n_flips: int = 13,
                     verbose: bool = True,
                     weight_mask: Optional[Dict[str, np.ndarray]] = None,
                     target_asr: float = 0.0, asr_loader=None,
                     asr_every: int = 5, device=None
                     ) -> Dict[str, np.ndarray]:
        """Flip bits until n_flips, or until the targeted success rate reaches
        target_asr, whichever comes first.

        Early stopping is what makes a full source-by-target sweep affordable.
        Without it, finding the flip count at which each pair crosses a success
        threshold needs a sweep per pair; with it, one run per pair reports the
        crossing directly. self.flips_used holds the answer, and stays at
        n_flips when the threshold was never reached.
        """
        target_layers = [n for n in target_layers if n in self.qinfo]
        q_cur = {n: self.qinfo[n][0].copy() for n in self.qinfo}
        self._write_back(q_cur)
        planes = (list(range(self.b)) if self.all_planes
                  else list(range(self.b - self.K, self.b)))
        crit = nn.CrossEntropyLoss()
        self.flips_used = n_flips
        self.asr_reached = 0.0
        self.asr_trace = []

        for step in range(n_flips):
            # gradient of the TARGETED objective
            self.model.zero_grad(set_to_none=True)
            for x, _ in src_batches[:1]:
                if x.numel() == 0:
                    continue
                yt = torch.full((x.shape[0],), self.target, dtype=torch.long,
                                device=x.device)
                crit(self.model(x), yt).backward()
            grads = {}
            for name in target_layers:
                mod = self.handles.get(name)
                if mod is None or mod.weight.grad is None:
                    continue
                grads[name] = mod.weight.grad.detach().cpu().numpy(
                    ).astype(np.float32).reshape(-1)
            if not grads:
                break

            cands = []
            for name, g in grads.items():
                q = q_cur[name].reshape(-1)
                sarr = self._scale_arr(name, self.qinfo[name][2]).reshape(-1)
                m = None
                if weight_mask is not None:
                    m = weight_mask.get(name)
                    if m is None or not m.any():
                        continue
                    m = m.reshape(-1)
                for j in planes:
                    dq = _flip_delta(q, j, self.b)
                    # we DESCEND the source-to-target loss, so take -grad.dw
                    score = -g * (dq.astype(np.float32) * sarr)
                    if m is not None:
                        score = np.where(m, score, -np.inf)
                    k = min(self.topk, score.size)
                    idx = np.argpartition(-score, k - 1)[:k]
                    for i in idx:
                        cands.append((float(score[i]), name, int(i), j))
            if not cands:
                break
            cands.sort(key=lambda c: -c[0])
            cands = cands[:self.topk]

            best = None
            for sc, name, i, j in cands:
                saved = q_cur[name]
                flat = saved.reshape(-1).copy()
                orig = flat[i]
                flat[i] = _toggle(orig, j)
                q_cur[name] = flat.reshape(self.qinfo[name][2])
                self._write_back({name: q_cur[name]})
                with torch.no_grad():
                    val = self._tbfa_loss(src_batches, other_batches)
                q_cur[name] = saved
                self._write_back({name: q_cur[name]})
                if best is None or val > best[0]:
                    best = (val, name, i, j, int(orig))
            if best is None:
                break

            val, name, i, j, orig = best
            flat = q_cur[name].reshape(-1)
            flat[i] = _toggle(orig, j)
            q_cur[name] = flat.reshape(self.qinfo[name][2])
            self._write_back({name: q_cur[name]})
            self.log.append({"step": step + 1, "layer": name, "index": i,
                             "plane": j, "loss": val})
            if verbose:
                print(f"   [tbfa] flip {step+1:3d}/{n_flips}  {name}"
                      f"  idx={i}  plane={j}  obj={val:.4f}", flush=True)

            # ---- early stop once the success threshold is met --------------
            if (target_asr > 0 and asr_loader is not None
                    and ((step + 1) % asr_every == 0 or step + 1 == n_flips)):
                r = targeted_rates(self.model, asr_loader,
                                   device or self.device,
                                   self.source, self.target)
                self.asr_trace.append((step + 1, r["asr"], r["other_acc"]))
                if verbose:
                    print(f"   [tbfa]   ASR after {step+1} flips: "
                          f"{r['asr']:.2f}%  (others {r['other_acc']:.2f}%)",
                          flush=True)
                if r["asr"] >= target_asr:
                    self.flips_used = step + 1
                    self.asr_reached = r["asr"]
                    if verbose:
                        print(f"   [tbfa] reached {target_asr:.0f}% ASR at "
                              f"{step+1} flips, stopping", flush=True)
                    return q_cur
                self.asr_reached = r["asr"]
        return q_cur


def split_class_batches(loader, device, source: int, n_batches=4,
                        max_size=128, min_source=32, max_scan=400):
    """Return (source-class batches, other-class batches).

    With 1000 classes a random batch holds roughly one image of any given
    class, so an early break leaves T-BFA with a gradient computed on a
    handful of samples. The search then flails and does general damage rather
    than targeted redirection, which is exactly the failure seen on ImageNet:
    accuracy collapsing to near zero while the targeted success rate stayed
    low. Keep scanning until at least `min_source` source-class images have
    been collected, up to `max_scan` batches.
    """
    src_x, oth = [], []
    n_src = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_scan:
            break
        m = (y == source)
        if m.any():
            src_x.append(x[m]); n_src += int(m.sum())
        if (~m).any() and len(oth) < n_batches:
            oth.append((x[~m][:max_size].to(device), y[~m][:max_size].to(device)))
        if len(oth) >= n_batches and n_src >= min_source:
            break
    src = []
    if src_x:
        cat = torch.cat(src_x)[:max_size]
        src = [(cat.to(device), torch.full((cat.shape[0],), source,
                                           dtype=torch.long, device=device))]
    if n_src < min_source:
        print(f"   [tbfa] WARNING only {n_src} images of class {source} found "
              f"in {max_scan} batches; the gradient will be noisy and the "
              f"attack may do general damage instead of targeted redirection")
    else:
        print(f"   [tbfa] collected {n_src} source-class images for the gradient")
    return src, oth[:n_batches]


@torch.no_grad()
def targeted_rates(model, loader, device, source: int, target: int,
                   max_batches: int = 0):
    """Measure the attack: source->target success, and accuracy on the rest.

    max_batches=0 scans the ENTIRE loader. Anything less risks never seeing the
    source class at all on a 1000-class dataset, which reports n=0 and an
    attack success rate of zero regardless of what actually happened.
    """
    model.eval()
    n_src = hit = n_oth = ok = 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        m = (y == source)
        if m.any():
            n_src += int(m.sum()); hit += int((p[m] == target).sum())
        if (~m).any():
            n_oth += int((~m).sum()); ok += int((p[~m] == y[~m]).sum())
    return {"source_n": n_src,
            "asr": 100.0 * hit / max(n_src, 1),
            "other_n": n_oth,
            "other_acc": 100.0 * ok / max(n_oth, 1)}


def collect_batches(loader, device, n_batches=2, max_size=128):
    """Grab a few batches and pin them to the device for repeated use."""
    out = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        out.append((x[:max_size].to(device), y[:max_size].to(device)))
    return out




# ---------------------------------------------------------------------------
#  PER-WEIGHT sensitivity bins
# ---------------------------------------------------------------------------
#  Layer-level sensitivity treats every weight inside a layer identically, but
#  even in the most fragile layer the great majority of weights barely matter.
#  Scoring each weight individually gives a much finer picture, and it is cheap:
#  one backward pass yields all 9.2M scores at once.
#
#  Score = the first-order loss increase from flipping that weight's MSB, which
#  is the single most damaging flip available on it:
#
#       damage(w) = | dL/dw  *  delta_w_from_MSB_flip |
#
#  This is the same quantity PBS ranks by, so the bins and the attack agree on
#  what "sensitive" means, rather than using two different definitions.
# ---------------------------------------------------------------------------
def weight_sensitivity(model, qinfo, handles, batches, layers=None,
                       bits: int = 8, device="cuda") -> Dict[str, np.ndarray]:
    """Per-weight MSB-flip damage score. Returns {layer: score array}."""
    layers = list(layers) if layers is not None else list(qinfo)
    crit = nn.CrossEntropyLoss()
    model.zero_grad(set_to_none=True)
    for x, y in batches:
        crit(model(x), y).backward()

    out = {}
    msb = bits - 1
    for name in layers:
        mod = handles.get(name)
        if mod is None or mod.weight.grad is None or name not in qinfo:
            continue
        g = mod.weight.grad.detach().cpu().numpy().astype(np.float32).reshape(-1)
        q, sinfo, shape = qinfo[name]
        kind, sc = sinfo
        if kind == "per_tensor":
            s = np.full(g.shape, float(sc), np.float32)
        else:
            s = np.asarray(sc, np.float32)
            while s.ndim < len(shape):
                s = s[..., None]
            s = np.broadcast_to(s, shape).astype(np.float32).reshape(-1)
        dq = _flip_delta(q.reshape(-1), msb, bits).astype(np.float32)
        out[name] = np.abs(g * dq * s)
    model.zero_grad(set_to_none=True)
    return out


def make_weight_bins(scores: Dict[str, np.ndarray],
                     fracs: Sequence[float] = (0.01, 0.10)
                     ) -> Dict[str, Dict[str, np.ndarray]]:
    """Split ALL weights globally into bins by score.

    fracs=(0.01, 0.10) gives:
        top  the most damaging 1%
        mid  the next 9%
        low  the remaining 90%
    Thresholds are global, not per layer, so a layer that happens to hold many
    high-scoring weights contributes more to the top bin. That is the intent:
    the bins describe weights, not layers.
    """
    allv = np.concatenate([v for v in scores.values()]) if scores else np.zeros(1)
    f = sorted(float(x) for x in fracs)
    cuts = [float(np.quantile(allv, 1.0 - x)) for x in f]      # high -> low
    masks = {"top": {}, "mid": {}, "low": {}}
    for name, v in scores.items():
        masks["top"][name] = v >= cuts[0]
        masks["mid"][name] = (v >= cuts[-1]) & (v < cuts[0])
        masks["low"][name] = v < cuts[-1]
    n_tot = allv.size
    for b in masks:
        n = int(sum(m.sum() for m in masks[b].values()))
        print(f"   [wbin] {b:>4}: {n:>12,} weights ({100*n/max(n_tot,1):5.2f}%)  "
              f"score >= {cuts[0] if b=='top' else (cuts[-1] if b=='mid' else 0):.3e}")
    return masks


def pbs_targets(sens_score: Dict[str, float], protected: Sequence[str],
                band: str = "top") -> List[str]:
    """Sensitivity-ranked layer list, sliced to the requested band."""
    if band == "carrier":
        pool = [n for n in sens_score if n not in set(protected)]
    else:
        pool = list(protected)
    ranked = sorted(pool, key=lambda n: -sens_score.get(n, 0.0))
    return _bands(ranked, band if band != "carrier" else "all")
