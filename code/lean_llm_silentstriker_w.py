#!/usr/bin/env python
"""
lean_llm_silentstriker_w.py  -  LEAN vs SilentStriker (Xu et al., NeurIPS 2025; stealthy sign-bit BFA) with the same
weight-level code as lean_llm_attnbreaker_w.py: the top --cov of all weights by INT8 sign-flip damage |g| x |w| are coded
and unflippable.  SilentStriker (selcode_silentstriker.py, unchanged) is subclassed only to mask those weights out of its
per-layer candidate set.  Reports MMLU and PPL after every iteration (k_top flips each) for the model given, saves the
flip list (json, same format as lean_llm_showcase.py) so lean_llm_harness.py can evaluate the six tasks.
SEPARATE file; existing scripts unchanged.

usage (control and LEAN exported dirs, as for AttentionBreaker):
  python -u lean_llm_silentstriker_w.py --model <exported dir> --cov 0.005 --objective mmlu --n-iter 5 --k-top 10 \
      --tag qwen2_0.5b_lam0 --out results_llm/ss
"""
import os, sys, json, argparse, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecc_llm_attnbreaker as E, ecc_llm_mmlu as M, attnbreaker_core as AB, selcode_silentstriker as SS


class MaskedSilentStriker(SS.SilentStriker):
    """SilentStriker whose candidate set excludes the coded weights (attacker knows the code and does not waste flips on it)"""
    def __init__(self, *a, protected=None, **k):
        super().__init__(*a, **k); self.protected = protected or {}

    def _one_iter(self, committed):
        best = None
        for name in self.layer_pool:
            g = self._grad_layer(name)
            pm = self.protected.get(name)
            if pm is not None and pm.size: g[pm] = 0.0                               # coded weights: never candidates
            if self.k_top * 10 >= g.size: continue
            idx = np.argpartition(-np.abs(g), self.k_top * 10)[:self.k_top * 10]
            idx = idx[np.argsort(-np.abs(g[idx]))][:self.k_top]
            trial = committed + [(name, int(i)) for i in idx]
            grp = {}
            for n, i in trial: grp.setdefault(n, []).append(i)
            for n, ii in grp.items(): self.ab.write(n, AB.flip_sign_bits(self.ab.clean[n], np.array(ii)))
            l = self._loss()
            for n in grp: self.ab.write(n, self.ab.clean[n])
            if best is None or l < best[0]: best = (l, name, [int(i) for i in idx])
        return best


def main():
    ap = argparse.ArgumentParser("LEAN vs SilentStriker with a weight-level code")
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", default="results_llm/ss")
    ap.add_argument("--exclude", default="lm_head,embed_tokens,wte,wpe"); ap.add_argument("--bits", type=int, default=8); ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--cov", type=float, default=0.005); ap.add_argument("--layer-filter", default="attn,mlp"); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--objective", choices=["ppl", "keytoken", "mmlu"], default="mmlu"); ap.add_argument("--mmlu-attack-n", type=int, default=16)
    ap.add_argument("--n-iter", type=int, default=5); ap.add_argument("--k-top", type=int, default=10)
    ap.add_argument("--n-prompts", type=int, default=4); ap.add_argument("--n-questions", type=int, default=200); ap.add_argument("--ppl-tokens", type=int, default=20000)
    ap.add_argument("--rank", choices=["loss", "mmlu", "both"], default="loss",
                    help="how the coded weights are chosen: loss = |dCE/dw| x |w| on the attack prompts (as the AttentionBreaker tables); "
                         "mmlu = |d(-log p(correct letter))/dw| x |w| on the DEFENDER's probe questions (disjoint from the attacker's); both = half each")
    ap.add_argument("--score", choices=["flip", "gw"], default="flip",
                    help="per-weight damage used for the ranking: flip = |g| x |sign-flip step| (the real INT8 damage, same for every weight of a tensor); "
                         "gw = |g| x |w| (AttentionBreaker's sensitivity, used in the earlier LLM runs)")
    ap.add_argument("--rank-n", type=int, default=32, help="defender probe questions for --rank mmlu (taken after the attacker's --mmlu-attack-n)")
    ap.add_argument("--attack-prompts", default="", help="prompts separated by || that the attacker targets (SilentStriker suppresses the key tokens of THEIR clean answers); default: 4 MMLU astronomy prompts")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); np.random.seed(a.seed); torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)

    rows = M.load_mmlu(a.n_questions); atk = M.load_mmlu(a.n_prompts, subjects=["astronomy"]); prompts = [M.fmt(p, q, c) for (p, q, c, _a) in atk]
    if a.attack_prompts: prompts = [p.strip() for p in a.attack_prompts.split("||") if p.strip()]; print(f"[attack] targeting {len(prompts)} custom prompts", flush=True)
    import ecc_llm_adapter as LA
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ppl_te = LA.TextEval(tok("\n\n".join(ds["text"]), return_tensors="pt"), a.ppl_tokens); ppl_now = lambda: LA._perplexity(model, ppl_te, dev)

    excl = [e for e in a.exclude.split(",") if e]; layers = {}
    for name, mod in model.named_modules():
        if (isinstance(mod, nn.Linear) or mod.__class__.__name__ == "Conv1D") and hasattr(mod, "weight") and not any(e in name for e in excl):
            q, s, shp = E.quantize_any(mod.weight, a.bits, a.per_channel); layers[name] = [q, s, shp, mod]
    tot = sum(v[0].size for v in layers.values()); clean = {n: v[0].copy() for n, v in layers.items()}
    def write(name, q_int):
        v = layers[name]; w = torch.from_numpy(E.dequant_any(q_int, v[1], v[2])).to(v[3].weight.device, v[3].weight.dtype)
        with torch.no_grad(): v[3].weight.copy_(w.reshape(v[3].weight.shape))
    def write_all(ints):
        for n in layers: write(n, ints[n])
    write_all(clean); base_acc = M.accuracy(model, tok, rows, dev); base_ppl = ppl_now()
    print(f"[{a.tag}] INT{a.bits} clean MMLU = {base_acc:.2f}%  PPL = {base_ppl:.3f}  ({len(layers)} layers, {tot:,} weights)", flush=True)

    ab = AB.AttentionBreaker(model, layers, write, clean, AB.build_loss_batches(tok, atk, a.n_prompts, a.n_prompts, dev),
                             device=dev, alpha=a.alpha, seed=a.seed, layer_filter=[f for f in a.layer_filter.split(",") if f])
    t0 = time.time(); ab.compute_gradients(); print(f"[ab] gradients for {len(ab.cands)} layers ({time.time()-t0:.0f}s)", flush=True)

    def flip_step(n):
        """|dequant(sign-flipped q) - dequant(q)| for every weight of layer n (128 x scale for per-tensor INT8)"""
        q = clean[n]; qf = AB.flip_sign_bits(q, np.arange(q.size))
        return np.abs(E.dequant_any(qf, layers[n][1], layers[n][2]).reshape(-1) - E.dequant_any(q, layers[n][1], layers[n][2]).reshape(-1))
    def wmag(n):
        return flip_step(n) if a.score == "flip" else np.abs(E.dequant_any(clean[n], layers[n][1], layers[n][2]).reshape(-1))

    def mmlu_damage():
        """defender-side ranking: |d(-log p(correct letter))/dw| x |w| over the defender's probe questions (not the attacker's)"""
        probe = rows[a.mmlu_attack_n:a.mmlu_attack_n + a.rank_n]; lid = SS._answer_ids(tok); letters = ["A", "B", "C", "D"]
        mods = [layers[n][3] for n in layers]
        for m in mods: m.weight.requires_grad_(True); m.weight.grad = None
        for (prefix, q, ch, ans) in probe:
            enc = tok(M.fmt(prefix, q, ch), return_tensors="pt", truncation=True, max_length=3072).to(dev)
            lg = model(**enc).logits[0, -1, :].float(); (-torch.log_softmax(lg, -1)[lid[letters[ans]]]).backward()
        out = {}
        for n in layers:
            m = layers[n][3]; g = m.weight.grad
            if g is not None:
                out[n] = np.abs(g.detach().float().cpu().numpy().reshape(-1)) * wmag(n)
            m.weight.grad = None; m.weight.requires_grad_(False)
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); return out

    def top_set(dmg, k, exclude):
        allv = np.concatenate([d[np.setdiff1d(np.arange(d.size), exclude.get(n, np.zeros(0, int)), assume_unique=True)] if n in exclude else d for n, d in dmg.items()])
        thr = np.partition(allv, -k)[-k]; sel = {}
        for n, d in dmg.items():
            idx = np.nonzero(d >= thr)[0]
            if n in exclude: idx = np.setdiff1d(idx, exclude[n], assume_unique=True)
            if idx.size: sel[n] = idx
        return sel

    # weight-level code: top --cov of all weights, ranked by loss damage (as the AttentionBreaker tables), by the defender's
    # answer-letter damage (threat-matched placement for stealthy attacks), or half each
    protected = {}; n_prot = 0
    if a.cov > 0:
        k_all = max(1, int(a.cov * tot)); k_loss = k_all if a.rank == "loss" else (0 if a.rank == "mmlu" else k_all // 2)
        if k_loss > 0:
            dmg = {}
            for n in layers:
                if n in ab.grads: dmg[n] = np.abs(np.asarray(ab.grads[n]).reshape(-1)) * wmag(n)
            protected = top_set(dmg, k_loss, {})
        if k_all - k_loss > 0:
            t0 = time.time(); dm = mmlu_damage(); add = top_set(dm, k_all - k_loss, protected)
            for n, idx in add.items(): protected[n] = np.union1d(protected[n], idx) if n in protected else idx
            print(f"[code] answer-letter ranking on {a.rank_n} defender probe questions ({time.time()-t0:.0f}s)", flush=True)
        n_prot = sum(v.size for v in protected.values())
        per = {}
        for n, v in protected.items(): per[n.split(".")[-1]] = per.get(n.split(".")[-1], 0) + int(v.size)
        print(f"[code] weight-level code (rank {a.rank}, score {a.score}) on {n_prot:,} weights = {100*n_prot/tot:.2f}% of the model; unflippable; by module: " + ", ".join(f"{k} {v}" for k, v in sorted(per.items(), key=lambda x: -x[1])), flush=True)

    import gc; gc.collect(); torch.cuda.empty_cache()
    ss = MaskedSilentStriker(ab, tok, prompts, device=dev, objective=a.objective, k_top=a.k_top, layers=ab.cands, seed=a.seed,
                             mmlu_rows=rows, mmlu_n=a.mmlu_attack_n, protected=protected)
    committed, hist = [], []
    for it in range(a.n_iter):
        t0 = time.time(); l, name, idx = ss._one_iter(committed); committed += [(name, i) for i in idx]
        att = ab.attacked_ints(committed); write_all(att); acc = M.accuracy(model, tok, rows, dev); ppl = ppl_now(); write_all(clean)
        hist.append((it + 1, name, len(committed), l, acc, ppl))
        print(f"[{a.tag}] iter {it+1}: layer {name}  flips {len(committed):3d}  attack loss {l:.4f}  MMLU {acc:.2f}  PPL {ppl:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"model": a.model, "tag": a.tag, "cov": a.cov, "objective": a.objective, "n_flips": len(committed), "flips": [(n, int(i)) for n, i in committed],
               "clean_mmlu": base_acc, "clean_ppl": base_ppl, "history": hist}, open(f"{a.out}_{a.tag}.json", "w"), indent=1)
    print(f"RESULT {a.tag} cov={a.cov} rank={a.rank} score={a.score} objective={a.objective} clean_mmlu={base_acc:.2f} clean_ppl={base_ppl:.2f} "
          + " ".join(f"flips{nf}:{acc:.2f}/{ppl:.1f}" for (_, _, nf, _, acc, ppl) in hist), flush=True)


if __name__ == "__main__":
    main()
