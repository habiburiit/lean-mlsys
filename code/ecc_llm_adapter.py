# Developed By ANON
# ANON, ECE Department
#
# ecc_llm_adapter.py
#
# Runs the REAL ecc_protect.py hiding engine on LLM weights.
# It does NOT reimplement the method. It builds the same `qinfo` dict the
# engine expects from an LLM's Linear/Conv1D weights, swaps the evaluator to
# return PERPLEXITY instead of top-1 accuracy, and then calls the engine's own
# measure_sensitivity() and run_at_ber() -- so the parity is really BCH-encoded
# from sensitive layers and really hidden via STDM in the low bits of
# insensitive carrier layers at zero added memory, identical to the paper.
#
# Because the engine's sensitivity/return values are named "acc" and it prints
# "accuracy", we feed it NEGATIVE perplexity as the "accuracy" signal: higher
# neg-PPL = better = like higher accuracy, so "acc drop" = PPL rise. The final
# reported numbers are converted back to true perplexity.
#
# Requires: transformers, datasets, and ecc_protect.py + ecc_engine in the same
# dir (the engine this imports).  pip install transformers datasets
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 python ecc_llm_adapter.py --model distilgpt2 \
#       --wikitext wikitext2 --bits 8 --bers 0,0.01,0.03,0.06 \
#       --mode sensitive --sens-frac 0.40 --plane-bits 4 --host-bits 5 \
#       --strong-t 12 --bch-m 7 --embed stdm --stdm-L 1 --store xlayer \
#       --parity-rep 1 --out distilgpt2_wt2
#
from __future__ import annotations
import os, sys, math, argparse, types
import numpy as np
import torch
import torch.nn as nn

# import the REAL engine
import ecc_protect as EP          # your file, unchanged
# the engine imports its BCH/STDM core as ENG (ecc_engine). We reuse EP.* below.


# ----------------------------------------------------------------------------
# 1. Build qinfo (name -> [int_weights, scale, shape]) from an LLM, in exactly
#    the format ecc_protect expects: q as int32 numpy, scale float, shape tuple.
# ----------------------------------------------------------------------------
def quantize_symmetric(w: torch.Tensor, bits=8, per_channel=False):
    """Quantize a weight matrix to signed INT `bits`.
    per_channel=False -> one scale for the whole tensor (fine for small models).
    per_channel=True  -> one scale per output channel (row); needed for large
    models (e.g. 70B) whose per-channel outliers make a single tensor-wide
    scale far too coarse, which otherwise corrupts the quantized model."""
    w = w.detach().float()
    qmax = (1 << (bits - 1)) - 1
    if per_channel and w.ndim >= 2:
        # scale per output channel (dim 0), matching the engine's per_channel
        # format: scale is an np.ndarray of length = w.shape[0].
        amax = w.abs().amax(dim=tuple(range(1, w.ndim))) + 1e-12   # [OC]
        scale = (amax / qmax)                                      # [OC]
        s = scale.view(-1, *([1] * (w.ndim - 1)))                  # broadcastable
        q = torch.clamp(torch.round(w / s), -qmax - 1, qmax)
        return (q.cpu().numpy().astype(np.int32),
                ("per_channel", scale.cpu().numpy().astype(np.float32)),
                tuple(w.shape))
    # per-tensor (default)
    scale = (w.abs().max().item() + 1e-12) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return q.cpu().numpy().astype(np.int32), ("per_tensor", float(scale)), tuple(w.shape)


def build_llm_qinfo(model, bits=8, max_layers=None, min_size=4096, exclude=None, per_channel=False):
    """Collect Linear (and GPT-2 Conv1D) weights. Returns (qinfo, handles)
    where handles[name] = the nn.Module whose .weight this is, so we can write
    recovered weights back. `exclude` = list of substrings; any layer whose
    name contains one is skipped (e.g. 'lm_head', 'wte')."""
    exclude = exclude or []
    qinfo, handles = {}, {}
    for name, mod in model.named_modules():
        is_lin = isinstance(mod, nn.Linear)
        is_c1d = mod.__class__.__name__ == "Conv1D" and hasattr(mod, "weight")
        if not (is_lin or is_c1d):
            continue
        full = name + ".weight"
        if any(x in full for x in exclude):
            continue
        w = mod.weight
        if w.numel() < min_size:
            continue
        q, s, shp = quantize_symmetric(w, bits, per_channel=per_channel)
        qinfo[full] = [q, s, shp]
        handles[full] = mod
    if max_layers:
        items = sorted(qinfo.items(), key=lambda kv: -kv[1][0].size)[:max_layers]
        keep = dict(items)
        handles = {k: handles[k] for k in keep}
        qinfo = keep
    return qinfo, handles


# ----------------------------------------------------------------------------
# 2. Write integer weights back into the LLM (engine calls apply_int_to_model,
#    which uses qinfo scale/shape; we monkeypatch that to use our handles).
# ----------------------------------------------------------------------------
def make_apply_int_to_model(handles):
    def _apply(model, recovered_int, qinfo, device):
        with torch.no_grad():
            for k, qb in recovered_int.items():
                if k not in handles:
                    continue
                _, scale_obj, shape = qinfo[k]
                kind, sc = scale_obj
                tgt_dev = handles[k].weight.device
                q = qb.astype(np.float32).reshape(shape)
                if kind == "per_tensor":
                    wf = q * float(sc)
                else:
                    # per_channel: broadcast one scale per output channel (dim0)
                    s = np.asarray(sc, dtype=np.float32)
                    while s.ndim < q.ndim:
                        s = s[..., None]
                    wf = q * s
                w = torch.from_numpy(wf).to(tgt_dev)
                handles[k].weight.copy_(w.reshape(handles[k].weight.shape))
    return _apply


# ----------------------------------------------------------------------------
# 3. Perplexity evaluator, wrapped so the engine's _eval(model, te, device)
#    returns NEGATIVE perplexity (so "higher = better", like accuracy).
#    te carries the tokenized text; we ignore its loader semantics.
# ----------------------------------------------------------------------------
class TextEval:
    def __init__(self, enc, max_tokens=20000, stride=512):
        self.enc = enc
        self.max_tokens = max_tokens
        self.stride = stride


@torch.no_grad()
def _perplexity(model, te: TextEval, device):
    model.eval()
    ids = te.enc["input_ids"]
    n = min(ids.size(1), te.max_tokens)
    nll_sum, ntok, prev = 0.0, 0, 0
    for begin in range(0, n, te.stride):
        end = min(begin + te.stride, n)
        inp = ids[:, begin:end].to(device)
        tgt = inp.clone()
        trg = end - prev
        tgt[:, :-trg] = -100
        out = model(inp, labels=tgt)
        loss = out.loss.item()
        if not math.isfinite(loss):
            loss = 50.0  # cap: a broken layer -> huge but finite NLL
        nll_sum += loss * trg
        ntok += trg
        prev = end
        if end >= n:
            break
    mean_nll = nll_sum / max(ntok, 1)
    # cap PPL so a single catastrophic layer cannot overflow to 1e83 and
    # dominate the sensitivity ranking; 1e6 is already "fully broken"
    return min(math.exp(min(mean_nll, 50.0)), 1e6)


def make_eval(as_neg_ppl=True):
    def _eval(model, te, device):
        ppl = _perplexity(model, te, device)
        # engine treats this as "accuracy": use -PPL so higher is better
        return (-ppl) if as_neg_ppl else ppl
    return _eval


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Run the real STDM hiding engine on LLM weights")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--wikitext", default="wikitext2",
                    choices=["wikitext2", "wikitext103"])
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--bers", default="0,0.01,0.03,0.06")
    ap.add_argument("--mode", choices=["sensitive", "uniform"], default="sensitive")
    ap.add_argument("--sens-frac", type=float, default=0.40)
    ap.add_argument("--probe-ber", type=float, default=0.02)
    ap.add_argument("--ber-hosts", type=int, default=0)
    ap.add_argument("--plane-bits", type=int, default=4)
    ap.add_argument("--host-bits", type=int, default=5)
    ap.add_argument("--strong-t", type=int, default=12)
    ap.add_argument("--light-t", type=int, default=2)
    ap.add_argument("--bch-m", type=int, default=7)
    ap.add_argument("--parity-rep", type=int, default=1)
    ap.add_argument("--embed", choices=["qim", "stdm"], default="stdm")
    ap.add_argument("--stdm-L", type=int, default=1)
    ap.add_argument("--store", choices=["self", "xlayer"], default="xlayer")
    ap.add_argument("--host-key", type=int, default=12345)
    ap.add_argument("--attack-key", type=int, default=-1)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--no-correct", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--exclude", default="lm_head,wte,wpe",
                    help="comma list of name substrings to skip (default skips "
                         "the output/embedding heads, whose corruption overflows PPL)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--multi-gpu", action="store_true",
                    help="shard the model across all visible GPUs via "
                         "device_map=auto (needed for 7B+ models). Requires "
                         "the accelerate package.")
    ap.add_argument("--fp16", action="store_true",
                    help="load the model in float16 (halves memory; needed to "
                         "fit ~70B models across the GPUs). Only affects the "
                         "loaded model dtype; quantization still uses the "
                         "dequantized values.")
    ap.add_argument("--per-channel", action="store_true",
                    help="use per-output-channel INT8 scales instead of one "
                         "scale per tensor. Needed for large models (~70B) "
                         "whose channel outliers make per-tensor INT8 too "
                         "coarse (broken perplexity). Recommended for 30B+.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="comma list of seeds to average over, e.g. 0,1,2. "
                         "When given, the BER sweep runs once per seed and "
                         "reports mean and std of perplexity across seeds. "
                         "Overrides --seed for the sweep.")
    ap.add_argument("--out", default="llm_ber")
    args = ap.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing libs. Run: pip install transformers datasets")

    # --- global reproducibility: seed python, numpy, and torch ---
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[seed] all RNGs seeded with {args.seed}", flush=True)

    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # --- model + data ---
    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.multi_gpu:
        # shard the model across all visible GPUs. Each layer lives on some
        # GPU; we must place inputs on the FIRST device and write each weight
        # back to the device where its layer already resides.
        load_dtype = torch.float16 if args.fp16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=load_dtype)
        # the input embedding's device is where inputs must go
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cuda:0")
        print(f"[multi-gpu] model sharded across "
              f"{torch.cuda.device_count()} GPUs; inputs on {dev}", flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model).to(dev)
    # NOTE: WikiText-2 and WikiText-103 share the SAME test split (identical
    # held-out articles), so split="test" gives identical tokens for both. To
    # make WikiText-103 a genuinely different, larger evaluation we use its
    # validation split (which differs from and is larger than WT-2's test).
    if args.wikitext == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        # WikiText-103 validation is ~60x larger than WT-2 test and distinct
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt")
    te = TextEval(enc, args.max_tokens)
    print(f"[data] {args.wikitext}: {enc['input_ids'].size(1):,} tokens", flush=True)

    # --- build qinfo from LLM weights ---
    excl = [x.strip() for x in args.exclude.split(",") if x.strip()]
    qinfo, handles = build_llm_qinfo(model, args.bits, args.max_layers, exclude=excl,
                                     per_channel=args.per_channel)
    tot = sum(v[0].size for v in qinfo.values())
    print(f"[qinfo] {len(qinfo)} layers, {tot:,} weights quantized to INT{args.bits}",
          flush=True)

    # --- monkeypatch the engine to use LLM eval + LLM write-back ---
    EP._eval = make_eval(as_neg_ppl=True)
    EP.apply_int_to_model = make_apply_int_to_model(handles)
    # apply quantized ints so the "base" reflects INT8, like the vision path
    EP.apply_int_to_model(model, {k: qinfo[k][0] for k in qinfo}, qinfo, dev)
    base_negppl = EP._eval(model, te, dev)
    base_ppl = -base_negppl
    print(f"[base] INT{args.bits} perplexity = {base_ppl:.3f}", flush=True)

    # --- build the engine's code object (identical to vision path) ---
    import ecc_engine as ENG  # same module the engine uses
    code = ENG.XLayerECC(strong_t=args.strong_t, light_t=args.light_t,
                         bch_m=args.bch_m, delta=args.delta, host_key=args.host_key)
    code.host_bits = args.host_bits
    code.plane_bits = args.plane_bits
    code.parity_rep = args.parity_rep
    code.ber_hosts = bool(args.ber_hosts)
    code.bit_width = int(args.bits)
    code.no_correct = bool(args.no_correct)
    code.embed_mode = args.embed
    code.stdm_L = args.stdm_L
    code.store_mode = args.store
    code.recover_key = args.attack_key if args.attack_key >= 0 else None

    # --- sensitivity (engine's own routine; base_acc = base neg-PPL) ---
    if args.mode == "uniform":
        sens = set(qinfo.keys()); sens_score = {n: 1.0 for n in qinfo}
        print(f"[select] uniform: all {len(sens)} layers")
    else:
        print("[sensitivity] measuring per-layer PPL rise ...", flush=True)
        sens_score = EP.measure_sensitivity(model, qinfo, code, dev, te,
                                            base_negppl,
                                            probe_ber=args.probe_ber,
                                            seed=args.seed, verbose=True)
        ranked = sorted(sens_score, key=lambda n: -sens_score[n])
        acc = 0; sel = []
        for n in ranked:
            if acc + qinfo[n][0].size > args.sens_frac * tot and sel:
                break
            sel.append(n); acc += qinfo[n][0].size
        sens = set(sel)
        print(f"[select] sensitive: {len(sens)} layers = {acc:,} wts "
              f"= {100*acc/tot:.1f}% of model", flush=True)
    # reapply clean quantized ints
    EP.apply_int_to_model(model, {k: qinfo[k][0] for k in qinfo}, qinfo, dev)

    # --- BER sweep via the REAL run_at_ber (does the STDM hiding + BCH correct) ---
    bers = [float(b) for b in args.bers.split(",")]
    # seeds to run: either the --seeds list, or the single --seed
    if args.seeds:
        seed_list = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    else:
        seed_list = [args.seed]
    print(f"[sweep] running seeds {seed_list} over BERs "
          f"{[b*100 for b in bers]}%", flush=True)

    # collect per-seed results: prot[seed_idx][ber_idx], unprot[...]
    prot_all, unprot_all = [], []
    for si, sd in enumerate(seed_list):
        prot_row, unprot_row = [], []
        for i, ber in enumerate(bers):
            # per-(seed,BER) reproducible seed; protected and unprotected use
            # the SAME value so both see identical bit-errors.
            ber_seed = sd * 100003 + i
            rng = np.random.default_rng(ber_seed)
            code.no_correct = False
            neg = EP.run_at_ber(model, qinfo, code, ber, dev, te, rng, sens,
                                sens_score, report=(si == 0 and i == 0))
            prot_row.append(-neg)
            EP.apply_int_to_model(model, {k: qinfo[k][0] for k in qinfo}, qinfo, dev)

            rng = np.random.default_rng(ber_seed)
            code.no_correct = True
            negu = EP.run_at_ber(model, qinfo, code, ber, dev, te, rng, sens,
                                 sens_score, report=False)
            unprot_row.append(-negu)
            code.no_correct = False
            EP.apply_int_to_model(model, {k: qinfo[k][0] for k in qinfo}, qinfo, dev)
        prot_all.append(prot_row)
        unprot_all.append(unprot_row)
        print(f"  [seed {sd}] protected: "
              + "  ".join(f"{p:.2f}" for p in prot_row), flush=True)

    prot_arr = np.array(prot_all)      # [n_seeds, n_bers]
    unprot_arr = np.array(unprot_all)
    prot_mean, prot_std = prot_arr.mean(0), prot_arr.std(0)
    unprot_mean, unprot_std = unprot_arr.mean(0), unprot_arr.std(0)

    print("\n=== PERPLEXITY vs BER "
          f"(mean +/- std over {len(seed_list)} seed(s)) ===")
    print("BER%      : " + "  ".join(f"{b*100:8.2f}" for b in bers))
    print("Prot mean : " + "  ".join(f"{m:8.2f}" for m in prot_mean))
    print("Prot std  : " + "  ".join(f"{s:8.2f}" for s in prot_std))
    print("Unpr mean : " + "  ".join(f"{m:8.1f}" for m in unprot_mean))
    print("Unpr std  : " + "  ".join(f"{s:8.1f}" for s in unprot_std))

    import csv
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ber", "protected_ppl_mean", "protected_ppl_std",
                    "unprotected_ppl_mean", "unprotected_ppl_std",
                    "n_seeds", "seeds"])
        for j, b in enumerate(bers):
            w.writerow([b, prot_mean[j], prot_std[j],
                        unprot_mean[j], unprot_std[j],
                        len(seed_list), ";".join(map(str, seed_list))])
    # also save every individual seed's numbers for full transparency
    with open(args.out + "_perseed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "ber", "protected_ppl", "unprotected_ppl"])
        for si, sd in enumerate(seed_list):
            for j, b in enumerate(bers):
                w.writerow([sd, b, prot_all[si][j], unprot_all[si][j]])
    print(f"saved -> {args.out}.csv  and  {args.out}_perseed.csv", flush=True)


if __name__ == "__main__":
    main()
