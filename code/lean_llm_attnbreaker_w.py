#!/usr/bin/env python3
"""
lean_llm_attnbreaker_w.py  -  AttentionBreaker (GenBFA) against a WEIGHT-level selective code on an LLM.
SEPARATE file; imports ecc_llm_attnbreaker.py, ecc_llm_mmlu.py and attnbreaker_core.py unchanged.

ecc_llm_attnbreaker.py protects whole layers (--sens-frac of the weights, chosen by layer).  This script protects the top --cov
fraction of WEIGHTS of the whole model by deployed damage |g_i| x |w_i| (the INT8 sign-flip damage, as in the paper's vision
experiments), treats them as unflippable for the attacker (the code corrects anything that lands there), and lets GenBFA
search everything else.  Reports the flips GenBFA needs and MMLU / perplexity after the attack, plus the same at fixed
flip counts.  This is the setting in which LEAN acts (it reshapes weights inside layers, not which layers matter).

usage:
  python -u lean_llm_attnbreaker_w.py --model ckpt_llm/qwen2_0.5b/lam0_1ep --cov 0.01 --out results_llm/lean_abw_lam0_cov0.01
"""
import os, sys, csv, argparse, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecc_llm_attnbreaker as E
import ecc_llm_mmlu as M
import attnbreaker_core as AB


def main():
    ap = argparse.ArgumentParser("LEAN vs AttentionBreaker with a weight-level code")
    ap.add_argument("--model", required=True); ap.add_argument("--exclude", default="lm_head,embed_tokens,wte,wpe")
    ap.add_argument("--bits", type=int, default=8); ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--cov", type=float, default=0.01, help="fraction of all weights under the code (0 = none)")
    ap.add_argument("--n-questions", type=int, default=200); ap.add_argument("--ppl-tokens", type=int, default=20000)
    ap.add_argument("--layer-filter", default="attn,mlp"); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-prompts", type=int, default=4); ap.add_argument("--loss-batch", type=int, default=4)
    ap.add_argument("--sens-pcts", default="0.001,0.01,0.1,1,5"); ap.add_argument("--layer-thresh", type=float, default=4.0)
    ap.add_argument("--n-seed-layers", type=int, default=1); ap.add_argument("--pop-pct", type=float, default=2.0)
    ap.add_argument("--loss-thresh", type=float, default=7.0); ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--init-sols", type=int, default=50); ap.add_argument("--per-gen", type=int, default=40)
    ap.add_argument("--patience", type=int, default=0); ap.add_argument("--fixed-flips", default="10,20,50,100")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results_llm/lean_abw")
    a = ap.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)

    rows = M.load_mmlu(a.n_questions); atk_rows = M.load_mmlu(a.n_prompts, subjects=["astronomy"])
    batches = AB.build_loss_batches(tok, atk_rows, a.n_prompts, a.loss_batch, dev)
    import ecc_llm_adapter as LA
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ppl_te = LA.TextEval(tok("\n\n".join(ds["text"]), return_tensors="pt"), a.ppl_tokens)
    ppl_now = lambda: LA._perplexity(model, ppl_te, dev)

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
    print(f"[base] INT{a.bits} MMLU = {base_acc:.2f}%  PPL = {base_ppl:.3f}  ({len(layers)} layers, {tot:,} weights)", flush=True)

    ab = AB.AttentionBreaker(model, layers, write, clean, batches, device=dev, alpha=a.alpha, seed=a.seed,
                             layer_filter=[f for f in a.layer_filter.split(",") if f])
    t0 = time.time(); ab.compute_gradients(); print(f"[ab] gradients for {len(ab.cands)} layers ({time.time()-t0:.0f}s)", flush=True)

    # ---- weight-level code: top --cov of all weights by |g| x |w| (INT8 sign-flip damage), made unflippable ----
    n_prot = 0
    if a.cov > 0:
        dmg = {}
        for n in layers:
            if n in ab.grads:
                w = E.dequant_any(clean[n], layers[n][1], layers[n][2]).reshape(-1)
                dmg[n] = np.abs(np.asarray(ab.grads[n]).reshape(-1)) * np.abs(w)
        allv = np.concatenate(list(dmg.values())); k = max(1, int(a.cov * tot))
        thr = np.partition(allv, -k)[-k]
        for n, d in dmg.items():
            idx = np.nonzero(d >= thr)[0]
            if idx.size and n in ab.scores:
                sc = np.asarray(ab.scores[n]); sc = sc.astype(np.float64) if sc.dtype.kind != "f" else sc
                sc[idx] = -np.inf; ab.scores[n] = sc; n_prot += idx.size
        print(f"[code] weight-level code on {n_prot:,} weights = {100*n_prot/tot:.2f}% of the model (threshold {thr:.3e}); "
              f"they are unflippable for the attacker", flush=True)

    pcts = [float(p) for p in a.sens_pcts.split(",") if p]
    sens_rows, seeds, ranked = ab.layer_sensitivity(pcts, a.layer_thresh)
    print("[ab] seed layers:", [(n, k, round(l, 3)) for n, (k, l, p) in ranked[:5]], flush=True)
    if not seeds:
        print(f"RESULT model={a.model} cov={a.cov} clean_mmlu={base_acc:.2f} clean_ppl={base_ppl:.2f} genbfa=NO_LAYER_BREAKS", flush=True)
        return
    pop = ab.population_space(seeds[:a.n_seed_layers], a.pop_pct)
    pop = [e for e in pop if np.isfinite(e[2])]
    print(f"[ab] population {len(pop):,} weights", flush=True)
    sols = []
    best, hist = ab.genbfa(pop, a.loss_thresh, a.gens, a.init_sols, a.per_gen, a.patience); sols.append((f"genbfa_{len(best)}", best))
    for N in [int(x) for x in a.fixed_flips.split(",") if x]:
        sols.append((f"fixed_{N}", [(n, i) for (n, i, _s) in pop[:N]]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    f = open(a.out + ".csv", "w", newline=""); w = csv.writer(f)
    w.writerow(["tag", "n_flips", "attack_loss", "base_mmlu", "mmlu_after", "base_ppl", "ppl_after", "cov", "n_protected"])
    res = []
    for tag, sol in sols:
        att = ab.attacked_ints(sol); write_all(att); loss = AB.lm_loss(model, batches)
        acc = M.accuracy(model, tok, rows, dev); ppl = ppl_now(); write_all(clean)
        print(f"[{tag}] flips={len(sol)} loss={loss:.3f} | MMLU base={base_acc:.2f} after={acc:.2f} | PPL base={base_ppl:.2f} after={ppl:.2f}", flush=True)
        w.writerow([tag, len(sol), f"{loss:.4f}", f"{base_acc:.2f}", f"{acc:.2f}", f"{base_ppl:.3f}", f"{ppl:.3f}", a.cov, n_prot]); f.flush()
        res.append(f"{tag}:{acc:.2f}/{ppl:.1f}")
    f.close()
    print(f"RESULT model={a.model} cov={a.cov} clean_mmlu={base_acc:.2f} clean_ppl={base_ppl:.2f} " + " ".join(res), flush=True)


if __name__ == "__main__":
    main()
