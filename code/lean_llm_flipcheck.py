#!/usr/bin/env python
"""
lean_llm_flipcheck.py  -  diagnose a saved AttentionBreaker flip list (results_llm/showcase_<tag>.json) on any model:
  for every flip: layer, flat index, INT8 code before/after, dequantized weight before/after, the layer's scale, and the
  weight's rank under the two code-placement scores (|g|x|w| and the attacker's own score) with "covered" = inside the
  top --cov fraction (i.e. the code of that placement would have made it unflippable); then the attack loss on the
  attacker's batches before and after applying the flips.  No attack search, no training; one model load.
  SEPARATE file: imports the same helpers as lean_llm_showcase.py, nothing is modified.
usage (bf16 shim as for the showcase):
  python -u lean_llm_bf16.py lean_llm_flipcheck.py --model ckpt_llm/llama31_8b/lam10_sum --flips results_llm/showcase_llama31_8b_lam10_sum_attacker_s1.json --cov 0.005 --seed 1
  python -u lean_llm_bf16.py lean_llm_flipcheck.py --model ckpt_llm/llama31_8b/lam0_1ep  --flips results_llm/showcase_llama31_8b_lam10_sum_attacker_s1.json --cov 0.005 --seed 1
"""
import os, sys, json, argparse, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecc_llm_attnbreaker as E, ecc_llm_mmlu as M, attnbreaker_core as AB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--flips", required=True)
    ap.add_argument("--exclude", default="lm_head,embed_tokens,wte,wpe"); ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--per-channel", action="store_true"); ap.add_argument("--cov", type=float, default=0.005)
    ap.add_argument("--layer-filter", default="attn,mlp"); ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-prompts", type=int, default=4); ap.add_argument("--loss-batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
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

    sol = [tuple(x) for x in json.load(open(a.flips))["flips"]]
    print(f"[flipcheck] model {a.model}  flips {a.flips}  n={len(sol)}", flush=True)
    atk_rows = M.load_mmlu(a.n_prompts, subjects=["astronomy"]); batches = AB.build_loss_batches(tok, atk_rows, a.n_prompts, a.loss_batch, dev)
    ab = AB.AttentionBreaker(model, layers, write, clean, batches, device=dev, alpha=a.alpha, seed=a.seed, layer_filter=[f for f in a.layer_filter.split(",") if f])
    write_all(clean); loss0 = AB.lm_loss(model, batches); print(f"[flipcheck] clean attack loss {loss0:.3f}", flush=True)
    t0 = time.time(); ab.compute_gradients(); print(f"[flipcheck] gradients ({time.time()-t0:.0f}s)", flush=True)

    # the two placement scores over all weights, and their top-cov thresholds
    k = max(1, int(a.cov * tot)); gw = {}; at = {}
    for n in layers:
        if n in ab.grads:
            w = E.dequant_any(clean[n], layers[n][1], layers[n][2]).reshape(-1)
            gw[n] = np.abs(np.asarray(ab.grads[n]).reshape(-1)) * np.abs(w)
            if n in ab.scores:
                sc = np.asarray(ab.scores[n]).reshape(-1).astype(np.float64); sc[~np.isfinite(sc)] = 0.0; at[n] = np.abs(sc)
    thr_gw = np.partition(np.concatenate(list(gw.values())), -k)[-k]
    thr_at = np.partition(np.concatenate(list(at.values())), -k)[-k] if at else float("inf")
    print(f"[flipcheck] code = top {a.cov*100:.2f}% = {k:,} weights; |g|x|w| threshold {thr_gw:.3e}; attacker-score threshold {thr_at:.3e}", flush=True)

    att = ab.attacked_ints(sol)
    n_cov_gw = n_cov_at = 0
    print("flip | layer | index | int8 before -> after | weight before -> after | scale | |g|x|w| (covered) | attacker score (covered)", flush=True)
    for j, (n, i) in enumerate(sol):
        i = int(i); q0 = int(clean[n].reshape(-1)[i]); q1 = int(att[n].reshape(-1)[i])
        w0 = float(E.dequant_any(clean[n], layers[n][1], layers[n][2]).reshape(-1)[i]); w1 = float(E.dequant_any(att[n], layers[n][1], layers[n][2]).reshape(-1)[i])
        sv = np.asarray(layers[n][1]).reshape(-1); shp = tuple(layers[n][2])
        s = float(sv[0]) if sv.size == 1 else float(sv[min(i // int(np.prod(shp[1:])) if len(shp) > 1 else 0, sv.size - 1)])
        g = gw.get(n); v = at.get(n)
        cg = bool(g[i] >= thr_gw) if g is not None else False; ca = bool(v[i] >= thr_at) if v is not None else False
        n_cov_gw += cg; n_cov_at += ca
        print(f"{j} | {n} | {i} | {q0} -> {q1} | {w0:.5f} -> {w1:.5f} | {s:.3e} | {g[i] if g is not None else 0:.3e} ({cg}) | {v[i] if v is not None else 0:.3e} ({ca})", flush=True)
    print(f"COVERED flips={len(sol)} covered_by_gradw_code={n_cov_gw} covered_by_attacker_code={n_cov_at}", flush=True)
    if len(sol) > 8:
        write_all(att); l = AB.lm_loss(model, batches); write_all(clean)
        print(f"RESULT flipcheck model={a.model} flips={len(sol)} clean_loss={loss0:.3f} attacked_loss={l:.3f}", flush=True); return
    # loss after each prefix of the list, and after all
    for m in range(1, len(sol) + 1):
        part = ab.attacked_ints(sol[:m]); write_all(part); l = AB.lm_loss(model, batches); write_all(clean)
        print(f"[flipcheck] after first {m} flip(s): attack loss {l:.3f}", flush=True)
    print(f"RESULT flipcheck model={a.model} flips={len(sol)} clean_loss={loss0:.3f} attacked_loss={l:.3f}", flush=True)


if __name__ == "__main__":
    main()
