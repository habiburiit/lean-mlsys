#!/usr/bin/env python
"""
lean_llm_harness.py  -  LM-Evaluation-Harness accuracies (the tasks AttentionBreaker reports: ARC-Easy, ARC-Challenge, BoolQ,
HellaSwag, PIQA, OpenBookQA, SocialIQA) for an exported model BEFORE and AFTER a saved AttentionBreaker flip list
(the .json written by lean_llm_showcase.py), with the same per-tensor INT8 quantization and weight-level code as the attack.
No attack is rerun.  Scoring by lean_llm_tasks.py (own 0-shot log-likelihood evaluator, harness protocol; no pip needed).

usage:
  python -u lean_llm_harness.py --model <exported dir> --flips results_llm/showcase_qwen2_0.5b_lam0_1ep.json --tag q05_control \
         --tasks arc_easy,arc_challenge,boolq,hellaswag,piqa,openbookqa,social_iqa --out results_llm/harness
output: <out>_<tag>.json and a RESULT line:  RESULT <tag> flips=N before: task=acc ... after: task=acc ...
"""
import os, sys, json, argparse, time, shutil, tempfile
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecc_llm_attnbreaker as E, ecc_llm_mmlu as M, attnbreaker_core as AB


def run_harness(model, tok, dev, tasks, limit):
    """own evaluator (lean_llm_tasks.py); the server cannot pip-install lm-eval"""
    from lean_llm_tasks import evaluate
    return evaluate(model, tok, dev, tasks, limit=limit, log=lambda m: print(m, flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--flips", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", default="arc_easy,arc_challenge,boolq,hellaswag,piqa,openbookqa,social_iqa")
    ap.add_argument("--exclude", default="lm_head,embed_tokens,wte,wpe"); ap.add_argument("--bits", type=int, default=8); ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--limit", type=int, default=None, help="examples per task (None = all)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results_llm/harness"); ap.add_argument("--tmp", default="")
    a = ap.parse_args(); np.random.seed(a.seed); torch.manual_seed(a.seed); tasks = [t for t in a.tasks.split(",") if t]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    excl = [e for e in a.exclude.split(",") if e]; layers = {}
    for name, mod in model.named_modules():
        if (isinstance(mod, nn.Linear) or mod.__class__.__name__ == "Conv1D") and hasattr(mod, "weight") and not any(e in name for e in excl):
            q, s, shp = E.quantize_any(mod.weight, a.bits, a.per_channel); layers[name] = [q, s, shp, mod]
    clean = {n: v[0].copy() for n, v in layers.items()}
    def write(name, q_int):
        v = layers[name]; w = torch.from_numpy(E.dequant_any(q_int, v[1], v[2])).to(v[3].weight.device, v[3].weight.dtype)
        with torch.no_grad(): v[3].weight.copy_(w.reshape(v[3].weight.shape))
    def write_all(ints):
        for n in layers: write(n, ints[n])
    sol = [tuple(x) for x in json.load(open(a.flips))["flips"]]
    atk_rows = M.load_mmlu(4, subjects=["astronomy"]); batches = AB.build_loss_batches(tok, atk_rows, 4, 4, dev)
    ab = AB.AttentionBreaker(model, layers, write, clean, batches, device=dev, alpha=0.5, seed=a.seed, layer_filter=["attn", "mlp"])
    res = {"tag": a.tag, "model": a.model, "n_flips": len(sol), "tasks": tasks}
    for phase, ints in [("before", clean), ("after", ab.attacked_ints(sol))]:
        write_all(ints); t0 = time.time()
        res[phase] = run_harness(model, tok, dev, tasks, a.limit)
        print(f"[{a.tag}] {phase} ({len(sol) if phase == 'after' else 0} flips, {time.time()-t0:.0f}s): " + "  ".join(f"{t}={v:.1f}" for t, v in res[phase].items() if v is not None), flush=True)
    write_all(clean); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(f"{a.out}_{a.tag}.json", "w"), indent=1)
    mean_b = np.mean([v for v in res["before"].values() if v is not None]); mean_a = np.mean([v for v in res["after"].values() if v is not None])
    fmt = lambda d: " ".join(f"{t}={v:.1f}" if v is not None else f"{t}=NA" for t, v in d.items())
    print(f"RESULT {a.tag} flips={len(sol)} before: {fmt(res['before'])} mean={mean_b:.1f} | after: {fmt(res['after'])} mean={mean_a:.1f}", flush=True)


if __name__ == "__main__":
    main()
