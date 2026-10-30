#!/usr/bin/env python3
# ============================================================================
# Developed By ANON, ANON, ECE Department
# the selective code - MMLU ACCURACY vs BER using the REAL STDM + BCH primitives.
#
# Uses ecc_engine.XLayerECC directly: real BCH parity (planes_encode), real
# STDM hiding in carrier layers (embed_store/embed_load), real BCH correction
# (planes_decode). Same defense as the perplexity path, measured as MMLU
# accuracy. Protected recovers the top-K bits of sensitive layers; unprotected
# keeps the corrupted weights.
#
# USAGE:
#   python ecc_llm_mmlu.py --model meta-llama/Meta-Llama-3-8B-Instruct \
#       --exclude lm_head,embed_tokens --bers 0,0.01,0.03,0.06 --seeds 0 \
#       --bits 8 --sens-frac 0.40 --plane-bits 4 --host-bits 5 --strong-t 12 \
#       --bch-m 7 --embed stdm --stdm-L 1 --n-questions 200 \
#       --multi-gpu --fp16 --out results_llm/llama3_8b_mmlu.csv
# ============================================================================
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecc_engine as ENG


def quantize(w, bits=8):
    w = w.detach().float().cpu().numpy()
    qmax = (1 << (bits - 1)) - 1
    scale = np.abs(w).max() / qmax if np.abs(w).max() > 0 else 1.0
    q = np.clip(np.round(w / scale), -qmax - 1, qmax).astype(np.int32)
    return q, scale, w.shape

def dequant(q, scale, shape):
    return (q.astype(np.float32) * scale).reshape(shape)

def to_ub(q):
    return (q.astype(np.int32) & 0xFF).astype(np.uint8)

def from_ub(b):
    x = b.astype(np.int32); x[x >= 128] -= 256; return x


MMLU_SUBJECTS = ["astronomy","business_ethics","college_biology","college_computer_science",
    "computer_security","conceptual_physics","electrical_engineering","high_school_biology",
    "high_school_computer_science","high_school_mathematics","high_school_physics",
    "machine_learning","management","marketing","medical_genetics","philosophy",
    "professional_accounting","sociology","us_foreign_policy","world_religions"]

def _ex(row, include_answer):
    # row: question, choices(list4), answer(int)
    L = ["A","B","C","D"]
    q, ch, ans = row
    t = q.strip() + "".join(f"\n{x}. {c}" for x, c in zip(L, ch)) + "\nAnswer:"
    if include_answer:
        t += f" {L[ans]}\n\n"
    return t

def load_arc(n, kshot=5):
    """ARC-Easy: question + labeled choices, 5-shot from train split."""
    from datasets import load_dataset
    train = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    test = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    LMAP = {"A":0,"B":1,"C":2,"D":3,"1":0,"2":1,"3":2,"4":3}
    def one(r, inc):
        L=["A","B","C","D"]; ch=r["choices"]["text"][:4]
        t=r["question"].strip()+"".join(f"\n{L[i]}. {c}" for i,c in enumerate(ch))+"\nAnswer:"
        if inc: t+=f" {L[LMAP.get(r['answerKey'],0)]}\n\n"
        return t
    head="The following are multiple choice science questions (with answers).\n\n"
    shots="".join(one(train[i],True) for i in range(kshot))
    prefix=head+shots
    out=[]
    for i in range(min(n,len(test))):
        r=test[i]; ch=r["choices"]["text"][:4]
        out.append((prefix, r["question"], ch, LMAP.get(r["answerKey"],0)))
    return out

def load_hellaswag(n, kshot=5):
    """HellaSwag: context + 4 endings, pick best ending."""
    from datasets import load_dataset
    train = load_dataset("Rowan/hellaswag", split="train")
    val = load_dataset("Rowan/hellaswag", split="validation")
    def one(r, inc):
        L=["A","B","C","D"]
        t=r["ctx"].strip()+"".join(f"\n{L[i]}. {e}" for i,e in enumerate(r["endings"]))+"\nAnswer:"
        if inc: t+=f" {L[int(r['label'])]}\n\n"
        return t
    head="Choose the best ending (A, B, C, or D) for each context.\n\n"
    shots="".join(one(train[i],True) for i in range(kshot))
    prefix=head+shots
    out=[]
    for i in range(min(n,len(val))):
        r=val[i]
        out.append((prefix, r["ctx"], r["endings"], int(r["label"])))
    return out

def load_mmlu(n, subjects=None, kshot=5):
    """Return a list of (fewshot_prefix, question, choices, answer) using
    subject-specific k-shot exemplars from the dev split, like the standard
    MMLU harness. n = total questions across subjects."""
    from datasets import load_dataset
    subs = subjects or MMLU_SUBJECTS
    per = max(1, n // len(subs))
    out = []
    for sub in subs:
        try:
            dev = load_dataset("cais/mmlu", sub, split="dev")
            test = load_dataset("cais/mmlu", sub, split="test")
        except Exception as e:
            print(f"[warn] skip {sub}: {e}", flush=True); continue
        head = ("The following are multiple choice questions (with answers) about "
                + sub.replace("_", " ") + ".\n\n")
        shots = ""
        for i in range(min(kshot, len(dev))):
            r = dev[i]
            shots += _ex((r["question"], r["choices"], int(r["answer"])), True)
        prefix = head + shots
        for i in range(min(per, len(test))):
            r = test[i]
            out.append((prefix, r["question"], r["choices"], int(r["answer"])))
        if n and len(out) >= n:
            break
    return out

def fmt(prefix, q, ch):
    L = ["A", "B", "C", "D"]
    body = q.strip() + "".join(f"\n{x}. {c}" for x, c in zip(L, ch)) + "\nAnswer:"
    return prefix + body

@torch.no_grad()
def accuracy(model, tok, rows, dev):
    model.eval()
    lid = [tok(" " + L, add_special_tokens=False)["input_ids"][0] for L in "ABCD"]
    ok = 0
    for prefix, q, ch, ans in rows:
        enc = tok(fmt(prefix, q, ch), return_tensors="pt",
                  truncation=True, max_length=3072).to(dev)
        lg = model(**enc).logits[0, -1, :]
        pred = int(torch.tensor([lg[i].item() for i in lid]).argmax())
        ok += (pred == ans)
    return 100.0 * ok / max(1, len(rows))


def main():
    ap = argparse.ArgumentParser("MMLU accuracy vs BER, real STDM+BCH")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--exclude", default="lm_head,embed_tokens,wte,wpe")
    ap.add_argument("--bers", default="0,0.01,0.03,0.06")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--sens-frac", type=float, default=0.40)
    ap.add_argument("--plane-bits", type=int, default=4)
    ap.add_argument("--host-bits", type=int, default=5)
    ap.add_argument("--strong-t", type=int, default=12)
    ap.add_argument("--light-t", type=int, default=2)
    ap.add_argument("--bch-m", type=int, default=7)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--host-key", type=int, default=1234)
    ap.add_argument("--embed", choices=["qim", "stdm"], default="stdm")
    ap.add_argument("--stdm-L", type=int, default=1)
    ap.add_argument("--max-layers", type=int, default=40)
    ap.add_argument("--n-questions", type=int, default=200)
    ap.add_argument("--dataset", choices=["mmlu","arc","hellaswag"], default="mmlu")
    ap.add_argument("--multi-gpu", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default="results_llm/mmlu.csv")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32
    kw = {"torch_dtype": dtype}
    if args.multi_gpu:
        kw["device_map"] = "auto"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if not args.multi_gpu:
        model = model.to(dev)
    model.eval()

    if args.dataset == "arc":
        rows = load_arc(args.n_questions)
    elif args.dataset == "hellaswag":
        rows = load_hellaswag(args.n_questions)
    else:
        rows = load_mmlu(args.n_questions)
    print(f"[data] {args.dataset}: {len(rows)} questions", flush=True)

    # real engine code object
    code = ENG.XLayerECC(strong_t=args.strong_t, light_t=args.light_t,
                         bch_m=args.bch_m, delta=args.delta, host_key=args.host_key)
    code.host_bits = args.host_bits; code.plane_bits = args.plane_bits
    code.bit_width = int(args.bits); code.embed_mode = args.embed
    code.stdm_L = args.stdm_L
    K = args.plane_bits; B = args.host_bits

    # collect layers
    excl = [e for e in args.exclude.split(",") if e]
    layers = {}
    for name, mod in model.named_modules():
        if (isinstance(mod, nn.Linear) or mod.__class__.__name__ == "Conv1D") and hasattr(mod, "weight"):
            if any(e in name for e in excl):
                continue
            q, s, shp = quantize(mod.weight, args.bits)
            layers[name] = [q, s, shp, mod]
    layers = dict(sorted(layers.items(), key=lambda kv: -kv[1][0].size)[:args.max_layers])
    tot = sum(v[0].size for v in layers.values())

    # sensitivity proxy = weight L1 magnitude (stable)
    score = {n: float(np.sum(np.abs(dequant(v[0], v[1], v[2])))) for n, v in layers.items()}
    ranked = sorted(score, key=lambda n: -score[n])
    acc = 0; sens = set()
    for n in ranked:
        if acc + layers[n][0].size > args.sens_frac * tot and sens:
            break
        sens.add(n); acc += layers[n][0].size
    print(f"[protect] {len(sens)} sensitive layers = {acc:,} wts = {100*acc/tot:.1f}%", flush=True)

    clean = {n: v[0].copy() for n, v in layers.items()}

    def write(name, q_int):
        v = layers[name]
        w = torch.from_numpy(dequant(q_int, v[1], v[2])).to(v[3].weight.device, v[3].weight.dtype)
        with torch.no_grad():
            v[3].weight.copy_(w.reshape(v[3].weight.shape))

    for n in layers: write(n, clean[n])
    base = accuracy(model, tok, rows, dev)
    print(f"[base] INT{args.bits} MMLU accuracy = {base:.2f}%", flush=True)

    # ---- encode parity for each sensitive layer (real BCH) ----
    parity = {}
    for name in sens:
        ecc, n0, G = code.planes_encode(to_ub(clean[name]), K, W=8)
        parity[name] = (ecc, n0, G)

    # ---- STDM-hide all parity in carrier layers (insensitive first) ----
    payload = np.concatenate([parity[n][0] for n in sens]).astype(np.uint8) if sens else np.zeros(0, np.uint8)
    spans = {}; pos = 0
    for n in sens:
        spans[n] = (pos, len(parity[n][0])); pos += len(parity[n][0])
    carriers = sorted(layers.keys(), key=lambda n: score.get(n, 0))  # insensitive first
    host = {n: dequant(clean[n], layers[n][1], layers[n][2]).astype(np.float32).reshape(-1) for n in carriers}
    seg = []; ptr = 0
    Lf = args.stdm_L if args.embed == "stdm" else 1
    for cn in carriers:
        if ptr >= len(payload): break
        cap = (host[cn].size // Lf) * B
        take = min(cap, len(payload) - ptr)
        if take <= 0: continue
        stored, _ = code.embed_store(host[cn].copy(), payload[ptr:ptr+take], B)
        host[cn] = stored
        seg.append((cn, ptr, take)); ptr += take
    if ptr < len(payload):
        print(f"[warn] storage overflow: need {len(payload):,}, fit {ptr:,}. "
              f"raise --host-bits or lower --stdm-L or --sens-frac.", flush=True)

    bers = [float(b) for b in args.bers.split(",")]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f = open(args.out, "w")
    f.write("ber,protected_acc_mean,protected_acc_std,"
            "unprotected_acc_mean,unprotected_acc_std,n_seeds,seeds\n")

    for ber in bers:
        pa, ua = [], []
        for sd in seeds:
            rng = np.random.default_rng(sd * 100003)
            # corrupt sensitive layers (bytes)
            corrupted = {n: code._corrupt_bytes(to_ub(clean[n]), ber, rng) for n in sens}

            # ---- PROTECTED: reload parity from STDM carriers, BCH-correct ----
            reloaded = np.zeros(len(payload), np.uint8)
            for (cn, start, take) in seg:
                bits = code.embed_load(host[cn], take, B)
                reloaded[start:start+take] = bits[:take]
            for n in layers:
                if n in sens:
                    ecc, n0, G = parity[n]
                    st, plen = spans[n]
                    voted = reloaded[st:st+plen]
                    fixed = code.planes_decode(corrupted[n], voted, K, n0, G, W=8)
                    write(n, from_ub(fixed).reshape(layers[n][2]))
                else:
                    write(n, clean[n])
            pa.append(accuracy(model, tok, rows, dev))

            # ---- UNPROTECTED: keep corrupted, no correction ----
            for n in layers:
                if n in sens:
                    write(n, from_ub(corrupted[n]).reshape(layers[n][2]))
                else:
                    write(n, clean[n])
            ua.append(accuracy(model, tok, rows, dev))
            for n in layers: write(n, clean[n])
        pm, ps = float(np.mean(pa)), float(np.std(pa))
        um, us = float(np.mean(ua)), float(np.std(ua))
        print(f"[BER {ber:.2%}] protected={pm:.2f}%  unprotected={um:.2f}%", flush=True)
        f.write(f"{ber},{pm},{ps},{um},{us},{len(seeds)},{';'.join(map(str,seeds))}\n"); f.flush()
    f.close()
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
