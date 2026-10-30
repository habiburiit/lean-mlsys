#!/usr/bin/env python
"""
lean_llm_showcase.py  -  qualitative table for the AttentionBreaker experiment: what the model SAYS before the attack and
after the GenBFA solution, for a control model and a LEAN model, with the same weight-level code as lean_llm_attnbreaker_w.py
(same seed -> same attack).  Greedy decoding on a few fixed prompts.  Also saves the flip list (json) so the attack never has
to be rerun for this model.  SEPARATE file; existing scripts unchanged (imports ecc_llm_attnbreaker, ecc_llm_mmlu,
attnbreaker_core, ecc_llm_adapter like lean_llm_attnbreaker_w.py).

usage:
  python -u lean_llm_showcase.py --model <exported dir> --cov 0.005 --seed 0 --tag qwen2_0.5b_lam0 --out results_llm/showcase
output: <out>_<tag>.md (markdown table), <out>_<tag>.json (prompts, generations, flips)
"""
import os, sys, json, argparse, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecc_llm_attnbreaker as E, ecc_llm_mmlu as M, attnbreaker_core as AB

PROMPTS = [                                  # default: completion-style prompts a small base model handles; override with --prompts "a||b||c"
    "The largest planet in the solar system is",
    "Water boils at sea level at a temperature of",
    "Photosynthesis is the process by which plants",
]


def generate(model, tok, dev, prompts, max_new=60):
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(dev)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(g[0, ids["input_ids"].size(1):], skip_special_tokens=True).strip().replace("\n", " "))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", default="results_llm/showcase")
    ap.add_argument("--exclude", default="lm_head,embed_tokens,wte,wpe"); ap.add_argument("--bits", type=int, default=8); ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--cov", type=float, default=0.005); ap.add_argument("--layer-filter", default="attn,mlp"); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-prompts", type=int, default=4); ap.add_argument("--loss-batch", type=int, default=4)
    ap.add_argument("--sens-pcts", default="0.001,0.01,0.1,1,5"); ap.add_argument("--layer-thresh", type=float, default=4.0)
    ap.add_argument("--n-seed-layers", type=int, default=1); ap.add_argument("--pop-pct", type=float, default=2.0)
    ap.add_argument("--loss-thresh", type=float, default=7.0); ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--init-sols", type=int, default=50); ap.add_argument("--per-gen", type=int, default=40); ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--prompts", default="", help='prompts separated by || (overrides the defaults)')
    ap.add_argument("--first", type=int, default=0, help="with --flips: apply only the first N flips of the saved list (SilentStriker: 10 = iteration 1)")
    ap.add_argument("--mmlu-prompts", type=int, default=0, help="K>0: use K MMLU evaluation questions (after the attacker's --n-prompts, so unseen by it) in MMLU format as the prompts")
    ap.add_argument("--flips", default="", help="json with a saved flip list: skip the attack and just apply it")
    ap.add_argument("--fixed", type=int, default=0, help="N>0: skip the GenBFA search, flip the top-N population weights (the fixed_N rows of the earlier RESULT lines)")
    a = ap.parse_args(); np.random.seed(a.seed); torch.manual_seed(a.seed)
    global PROMPTS
    if a.prompts: PROMPTS = [p.strip() for p in a.prompts.split("||") if p.strip()]
    if a.mmlu_prompts > 0:
        qs = M.load_mmlu(a.n_prompts + a.mmlu_prompts)[a.n_prompts:]
        PROMPTS = [M.fmt(p, q, c) for (p, q, c, _ans) in qs]
        print("[prompts] MMLU questions with correct letters: " + ", ".join("ABCD"[ans] for (_p, _q, _c, ans) in qs), flush=True)
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

    write_all(clean); before = generate(model, tok, dev, PROMPTS, a.max_new)
    print(f"[{a.tag}] before attack:", flush=True)
    for p, o in zip(PROMPTS, before): print("   Q:", p.strip(), "\n   A:", o[:200], flush=True)

    if a.flips:
        sol = [tuple(x) for x in json.load(open(a.flips))["flips"]]
        if a.first > 0: sol = sol[:a.first]
        atk_rows = M.load_mmlu(a.n_prompts, subjects=["astronomy"]); batches = AB.build_loss_batches(tok, atk_rows, a.n_prompts, a.loss_batch, dev)
        ab = AB.AttentionBreaker(model, layers, write, clean, batches, device=dev, alpha=a.alpha, seed=a.seed, layer_filter=[f for f in a.layer_filter.split(",") if f])
    else:
        atk_rows = M.load_mmlu(a.n_prompts, subjects=["astronomy"]); batches = AB.build_loss_batches(tok, atk_rows, a.n_prompts, a.loss_batch, dev)
        ab = AB.AttentionBreaker(model, layers, write, clean, batches, device=dev, alpha=a.alpha, seed=a.seed, layer_filter=[f for f in a.layer_filter.split(",") if f])
        t0 = time.time(); ab.compute_gradients(); print(f"[ab] gradients ({time.time()-t0:.0f}s)", flush=True)
        n_prot = 0
        if a.cov > 0:                                                  # identical to lean_llm_attnbreaker_w.py
            dmg = {}
            for n in layers:
                if n in ab.grads:
                    w = E.dequant_any(clean[n], layers[n][1], layers[n][2]).reshape(-1); dmg[n] = np.abs(np.asarray(ab.grads[n]).reshape(-1)) * np.abs(w)
            allv = np.concatenate(list(dmg.values())); k = max(1, int(a.cov * tot)); thr = np.partition(allv, -k)[-k]
            for n, d in dmg.items():
                idx = np.nonzero(d >= thr)[0]
                if idx.size and n in ab.scores:
                    sc = np.asarray(ab.scores[n]); sc = sc.astype(np.float64) if sc.dtype.kind != "f" else sc; sc[idx] = -np.inf; ab.scores[n] = sc; n_prot += idx.size
            print(f"[code] {n_prot:,} weights ({100*n_prot/tot:.2f}%) unflippable", flush=True)
        pcts = [float(p) for p in a.sens_pcts.split(",") if p]
        _, seeds, ranked = ab.layer_sensitivity(pcts, a.layer_thresh); print("[ab] seed layers:", [(n, k, round(l, 3)) for n, (k, l, p) in ranked[:3]], flush=True)
        if not seeds: print(f"[{a.tag}] no layer breaks; nothing to show", flush=True); return
        pop = ab.population_space(seeds[:a.n_seed_layers], a.pop_pct); pop = [e for e in pop if np.isfinite(e[2])]
        if a.fixed > 0: sol = [(n, i) for (n, i, _s) in pop[:a.fixed]]
        else: sol, _ = ab.genbfa(pop, a.loss_thresh, a.gens, a.init_sols, a.per_gen, a.patience)
    att = ab.attacked_ints(sol); write_all(att); loss = AB.lm_loss(model, batches)
    after = generate(model, tok, dev, PROMPTS, a.max_new); write_all(clean)
    print(f"[{a.tag}] after attack ({len(sol)} flips, attack loss {loss:.3f}):", flush=True)
    for p, o in zip(PROMPTS, after): print("   Q:", p.strip(), "\n   A:", o[:200], flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"model": a.model, "tag": a.tag, "cov": a.cov, "n_flips": len(sol), "attack_loss": float(loss), "flips": [(n, int(i)) for n, i in sol],
               "prompts": PROMPTS, "before": before, "after": after}, open(f"{a.out}_{a.tag}.json", "w"), indent=1)
    with open(f"{a.out}_{a.tag}.md", "w") as f:
        f.write(f"| Prompt | Before attack | After AttentionBreaker ({len(sol)} flips) |\n|---|---|---|\n")
        for p, b, o in zip(PROMPTS, before, after): f.write(f"| {p.strip()} | {b[:160]} | {o[:160]} |\n")
    print(f"RESULT {a.tag} flips={len(sol)} loss={loss:.3f} saved {a.out}_{a.tag}.md/.json", flush=True)


if __name__ == "__main__":
    main()
