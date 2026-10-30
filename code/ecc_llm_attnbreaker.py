#!/usr/bin/env python3
# =============================================================================
#  ecc_llm_attnbreaker.py   the selective code vs AttentionBreaker (GenBFA) on an LLM
# -----------------------------------------------------------------------------
#  Developed By ANON, ANON, ECE Department
#
#  Same model/quantization/MMLU path as ecc_llm_mmlu.py, same real engine
#  (ecc_engine.XLayerECC: BCH parity over the top-K bit planes, STDM-hidden in
#  carrier layers at zero added memory). The random BER injector is replaced
#  by AttentionBreaker's search (attnbreaker_core.py), which finds the
#  smallest set of INT8 sign-bit flips that breaks the model.
#
#  Reported per attack:
#    flips, attack loss, MMLU accuracy unprotected / protected, how many of the
#    flipped weights fall inside the selective code's protected layers, and the per-codeword
#    flip count versus the BCH t (this decides correctability before running).
#
#  Modes
#    --attack cwaware    codeword-aware adaptive attacker: knows the selective code's codeword
#                        layout and puts t+1 flips into each of the N highest-
#                        scoring codewords (the minimum that defeats BCH), for N
#                        in --cw-counts. Reports the flip cost of breaking the selective code.
#    --attack genbfa     full AttentionBreaker: layer ranking -> population ->
#                        evolutionary minimization (default)
#    --attack fixed      skip the search: flip the top-N population weights for
#                        each N in --fixed-flips (their ablation; gives a curve)
#    --adaptive          the attacker's loss is measured AFTER the selective code correction,
#                        so the search only "sees" flips that survive BCH.
#
#  USAGE
#   python ecc_llm_attnbreaker.py --model Qwen/Qwen2-0.5B \
#       --exclude lm_head,embed_tokens --bits 8 --sens-frac 0.40 \
#       --plane-bits 4 --host-bits 5 --strong-t 12 --bch-m 7 \
#       --embed stdm --stdm-L 1 --n-questions 200 \
#       --attack genbfa --gens 60 --per-gen 30 --loss-thresh 7 \
#       --out results_llm/qwen05b_attnbreaker
#
#   python ecc_llm_attnbreaker.py --model meta-llama/Meta-Llama-3-8B-Instruct \
#       --multi-gpu --fp16 --per-channel ... --attack fixed --fixed-flips 3,10,28,100
# =============================================================================
from __future__ import annotations
import os, sys, csv, json, argparse, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecc_engine as ENG
import ecc_llm_mmlu as M
import attnbreaker_core as AB


# ---------------------------------------------------------------------------
# quantization (per-tensor as in ecc_llm_mmlu, or per-output-channel)
# ---------------------------------------------------------------------------
def quantize_any(w, bits=8, per_channel=False):
    if not per_channel:
        return M.quantize(w, bits)
    w = w.detach().float().cpu().numpy()
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).reshape(w.shape[0], -1).max(axis=1) + 1e-12
    scale = (amax / qmax).astype(np.float32)
    s = scale.reshape(-1, *([1] * (w.ndim - 1)))
    q = np.clip(np.round(w / s), -qmax - 1, qmax).astype(np.int32)
    return q, scale, w.shape


def dequant_any(q, scale, shape):
    return AB.AttentionBreaker._dequant(q, scale, shape)


# ---------------------------------------------------------------------------
# the selective code protection for the LLM, with SUBSET decoding so that correcting a
# handful of flips does not decode every codeword of a 14M-weight layer.
# ---------------------------------------------------------------------------
class SelCodeLLM:
    def __init__(self, code, layers, clean, sens, score, K, B, Lf,
                 interleave=False, seed=0):
        self.code, self.layers, self.clean = code, layers, clean
        self.sens, self.score, self.K, self.B, self.Lf = sens, score, K, B, Lf
        self.nk = code.nk                     # parity bits per codeword
        self.t = code.strong_t
        self.interleave = bool(interleave)
        self.perm, self._inv = {}, {}         # keyed permutation per protected layer
        self.parity = {}                      # name -> (ecc bits, n0, G)
        for li, name in enumerate(sorted(sens)):
            wb = M.to_ub(clean[name]).reshape(-1)
            if self.interleave:
                # same definition as run_interleave.py: rng(seed).permutation(N),
                # encode in permuted order, decode, un-permute. One key per layer.
                rng = np.random.default_rng(seed * 1000003 + li)
                self.perm[name] = rng.permutation(wb.size).astype(np.int32)
                wb = wb[self.perm[name]]
            self.parity[name] = code.planes_encode(wb, K, W=8)

        # hide all parity in carriers (insensitive first), then reload it once
        self.payload = (np.concatenate([self.parity[n][0] for n in sens]).astype(np.uint8)
                        if sens else np.zeros(0, np.uint8))
        self.spans = {}; pos = 0
        for n in sens:
            self.spans[n] = (pos, len(self.parity[n][0])); pos += len(self.parity[n][0])
        carriers = sorted(layers.keys(), key=lambda n: score.get(n, 0))
        self.host = {n: dequant_any(clean[n], layers[n][1], layers[n][2])
                     .astype(np.float32).reshape(-1) for n in carriers}
        self.seg = []; ptr = 0
        for cn in carriers:
            if ptr >= len(self.payload):
                break
            cap = (self.host[cn].size // Lf) * B
            take = min(cap, len(self.payload) - ptr)
            if take <= 0:
                continue
            stored, _ = code.embed_store(self.host[cn].copy(), self.payload[ptr:ptr + take], B)
            self.host[cn] = stored
            self.seg.append((cn, ptr, take)); ptr += take
        self.overflow = ptr < len(self.payload)
        self.carrier_of = {cn: (st, tk) for (cn, st, tk) in self.seg}
        self.reloaded = self._reload({})
        self.parity_ber_clean = float(np.mean(self.reloaded != self.payload)) if len(self.payload) else 0.0

    # -- codeword geometry (in ENCODED order) ---------------------------------
    def inv(self, name):
        if name not in self._inv:
            self._inv[name] = np.argsort(self.perm[name]).astype(np.int32)
        return self._inv[name]

    def pos_of(self, name, idx):
        """Original flat index -> position in the encoded (maybe permuted) array."""
        return self.inv(name)[idx] if self.interleave else idx

    def members(self, name, c):
        """Original flat indices of the weights in codeword c of layer name."""
        G = self.parity[name][2]; n = self.clean[name].size
        pos = np.arange(c * G, min((c + 1) * G, n))
        return self.perm[name][pos] if self.interleave else pos

    def _reload(self, carrier_flips):
        """Read parity back from carriers. carrier_flips: {name: idx} of sign
        flips the attacker put INTO carrier layers; those perturb the host
        floats the parity lives in, so read from the perturbed copy."""
        out = np.zeros(len(self.payload), np.uint8)
        for (cn, st, tk) in self.seg:
            h = self.host[cn]
            if cn in carrier_flips:
                h = h.copy()
                q, s, shp, _ = self.layers[cn]
                idx = carrier_flips[cn]
                qf = AB.flip_sign_bits(self.clean[cn], idx).reshape(-1)
                dq = (qf[idx] - self.clean[cn].reshape(-1)[idx]).astype(np.float32)
                sc = s if isinstance(s, (float, int)) else np.broadcast_to(
                    np.asarray(s, np.float32).reshape(-1, *([1] * (len(shp) - 1))), shp).reshape(-1)[idx]
                h[idx] += dq * sc
            out[st:st + tk] = self.code.embed_load(h, tk, self.B)[:tk]
        return out

    def codeword_stats(self, flips):
        """flips: {name: idx}. Returns rows (layer, protected?, n_flips,
        n_codewords_touched, max_flips_in_one_codeword, correctable?)."""
        rows = []
        for n, idx in flips.items():
            if n in self.parity:
                G = self.parity[n][2]
                cw, cnt = np.unique(self.pos_of(n, idx) // G, return_counts=True)
                mx = int(cnt.max())
                rows.append((n, True, int(idx.size), int(cw.size), mx, mx <= self.t))
            else:
                rows.append((n, False, int(idx.size), 0, 0, False))
        return rows

    def correct(self, attacked, flips, use_reload=True):
        """attacked: {name: q}; flips: {name: idx}. Returns corrected ints for
        every layer. Only codewords touched by a flip are decoded."""
        carrier_flips = {n: i for n, i in flips.items() if n in self.carrier_of and n not in self.parity}
        reloaded = self._reload(carrier_flips) if (use_reload and carrier_flips) else self.reloaded
        rec = {n: attacked[n] for n in self.layers}
        for n, idx in flips.items():
            if n not in self.parity:
                continue                                     # unprotected layer
            ecc, n0, G = self.parity[n]
            st, plen = self.spans[n]
            par = reloaded[st:st + plen]
            cws = np.unique(self.pos_of(n, idx) // G)
            wb = M.to_ub(attacked[n]).reshape(-1)
            if self.interleave:
                wb = wb[self.perm[n]]                      # encoded order
            pad = (-n0) % G
            wf = np.concatenate([wb, np.zeros(pad, np.uint8)]) if pad else wb
            sub_w = np.concatenate([wf[c * G:(c + 1) * G] for c in cws])
            sub_p = np.concatenate([par[c * self.nk:(c + 1) * self.nk] for c in cws])
            fixed = self.code.planes_decode(sub_w, sub_p, self.K, len(sub_w), G, W=8)
            out = wf.copy()
            for j, c in enumerate(cws):
                out[c * G:(c + 1) * G] = fixed[j * G:(j + 1) * G]
            out = out[:n0]
            if self.interleave:
                out = out[self.inv(n)]                     # back to model order
            rec[n] = M.from_ub(out).reshape(self.layers[n][2])
        return rec


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("the selective code vs AttentionBreaker on an LLM (MMLU)")
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--exclude", default="lm_head,embed_tokens,wte,wpe")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--max-layers", type=int, default=0, help="0 = all")
    # the selective code
    ap.add_argument("--sens-frac", type=float, default=0.40)
    ap.add_argument("--sens-proxy", choices=["l1", "grad", "probe"], default="l1",
                    help="layer ranking for protection: l1 = weight L1 (ecc_llm_mmlu "
                         "default), grad = sum|g*w|, probe = AttentionBreaker's own "
                         "layer-sensitivity loss (protect what the attacker finds)")
    ap.add_argument("--plane-bits", type=int, default=4)
    ap.add_argument("--host-bits", type=int, default=5)
    ap.add_argument("--strong-t", type=int, default=12)
    ap.add_argument("--light-t", type=int, default=2)
    ap.add_argument("--bch-m", type=int, default=7)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--host-key", type=int, default=1234)
    ap.add_argument("--embed", choices=["qim", "stdm"], default="stdm")
    ap.add_argument("--stdm-L", type=int, default=1)
    ap.add_argument("--interleave", type=int, default=0,
                    help="1 = keyed permutation before BCH (run_interleave.py), 0 = adjacent")
    ap.add_argument("--attacker-knows-key", type=int, default=0,
                    help="cwaware: 1 = attacker uses the selective code's true codeword layout; "
                         "0 = attacker assumes adjacent grouping")
    # evaluation
    ap.add_argument("--n-questions", type=int, default=200)
    ap.add_argument("--benchmarks", default="",
                    help="extra MC datasets to also report: arc,hellaswag,truthfulqa")
    ap.add_argument("--demo-prompts", default="",
                    help="'|'-separated prompts; for each attack, greedy-generate before attack, "
                         "after attack, and after the selective code, and write <out>_outputs.{md,tex}. "
                         "Use 'default' for a built-in set of three.")
    ap.add_argument("--demo-tokens", type=int, default=60)
    ap.add_argument("--ppl", type=int, default=1,
                    help="1 = also report WikiText-2 perplexity (uses ecc_llm_adapter)")
    ap.add_argument("--ppl-tokens", type=int, default=20000)
    ap.add_argument("--multi-gpu", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    # attack
    ap.add_argument("--attack", choices=["genbfa", "fixed", "cwaware"], default="genbfa")
    ap.add_argument("--cw-counts", default="1,2,5,10,20,50,100",
                    help="cwaware: number of codewords to overload (t+1 flips each)")
    ap.add_argument("--cw-probe", type=int, default=400,
                    help="cwaware: how many top-scored codewords to measure individually")
    ap.add_argument("--cw-flips", type=int, default=0,
                    help="cwaware: flips per codeword (0 = t+1, the minimum that defeats BCH)")
    ap.add_argument("--adaptive", action="store_true",
                    help="attacker's loss is measured after the selective code correction")
    ap.add_argument("--layer-filter", default="attn,mlp")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-prompts", type=int, default=4, help="MMLU prompts in the attack loss")
    ap.add_argument("--loss-batch", type=int, default=4)
    ap.add_argument("--sens-pcts", default="0.001,0.01,0.1,1,5")
    ap.add_argument("--layer-thresh", type=float, default=4.0)
    ap.add_argument("--n-seed-layers", type=int, default=1)
    ap.add_argument("--pop-pct", type=float, default=2.0)
    ap.add_argument("--loss-thresh", type=float, default=7.0)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--init-sols", type=int, default=50)
    ap.add_argument("--per-gen", type=int, default=40)
    ap.add_argument("--patience", type=int, default=0, help="stop if best size unchanged for N gens (0=off)")
    ap.add_argument("--fixed-flips", default="3,10,28,100,500")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_llm/attnbreaker")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"torch_dtype": torch.float16 if args.fp16 else torch.float32}
    if args.multi_gpu:
        kw["device_map"] = "auto"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if not args.multi_gpu:
        model = model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # --- data: MMLU eval rows + the attacker's astronomy prompts --------------
    rows = M.load_mmlu(args.n_questions)
    bench = {}
    if args.benchmarks:
        import selcode_benchmarks as BM
        for b in [x.strip() for x in args.benchmarks.split(",") if x.strip()]:
            try:
                bench[b] = BM.load(b, args.n_questions)
                print(f"[data] {b}: {len(bench[b])} questions", flush=True)
            except Exception as e:
                print(f"[data] {b} skipped: {e}", flush=True)
    def bench_eval():
        import selcode_benchmarks as BM
        return {b: BM.accuracy(model, tok, rb, dev) for b, rb in bench.items()}
    atk_rows = M.load_mmlu(args.n_prompts, subjects=["astronomy"])
    batches = AB.build_loss_batches(tok, atk_rows, args.n_prompts, args.loss_batch, dev)
    print(f"[data] MMLU eval {len(rows)} q, attack loss on {args.n_prompts} astronomy prompts", flush=True)
    ppl_te = None
    if args.ppl:
        import ecc_llm_adapter as LA
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        ppl_te = LA.TextEval(tok("\n\n".join(ds["text"]), return_tensors="pt"), args.ppl_tokens)
        print(f"[data] WikiText-2 perplexity on {min(ppl_te.enc['input_ids'].size(1), args.ppl_tokens):,} tokens", flush=True)

    def ppl_now():
        return LA._perplexity(model, ppl_te, dev) if ppl_te is not None else float("nan")

    DEMO_DEFAULT = ["What is the tallest mountain in the world?",
                    "What is the boiling point of water in Celsius?",
                    "Elaborate on the theory of relativity."]
    demo = (DEMO_DEFAULT if args.demo_prompts.strip() == "default"
            else [q.strip() for q in args.demo_prompts.split("|") if q.strip()])

    @torch.no_grad()
    def generate(prompt):
        if getattr(tok, "chat_template", None):
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True, return_tensors="pt").to(dev)
            enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        else:
            enc = tok(prompt, return_tensors="pt").to(dev)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        out = model.generate(**enc, max_new_tokens=args.demo_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return " ".join(txt.split())

    def esc(t):
        return (t.replace("\\", "/").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
                 .replace("#", "\\#").replace("_", "\\_").replace("{", "(").replace("}", ")"))

    if demo:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    demo_rows = []                                     # (tag, prompt, before, after, selcode)
    demo_clean = {q: generate(q) for q in demo} if demo else {}

    # --- INT8 layers, same collection rule as ecc_llm_mmlu ----------------------
    excl = [e for e in args.exclude.split(",") if e]
    layers = {}
    for name, mod in model.named_modules():
        if (isinstance(mod, nn.Linear) or mod.__class__.__name__ == "Conv1D") and hasattr(mod, "weight"):
            if any(e in name for e in excl):
                continue
            q, s, shp = quantize_any(mod.weight, args.bits, args.per_channel)
            layers[name] = [q, s, shp, mod]
    if args.max_layers:
        layers = dict(sorted(layers.items(), key=lambda kv: -kv[1][0].size)[:args.max_layers])
    tot = sum(v[0].size for v in layers.values())
    clean = {n: v[0].copy() for n, v in layers.items()}

    def write(name, q_int):
        v = layers[name]
        w = torch.from_numpy(dequant_any(q_int, v[1], v[2])).to(v[3].weight.device, v[3].weight.dtype)
        with torch.no_grad():
            v[3].weight.copy_(w.reshape(v[3].weight.shape))

    def write_all(ints):
        for n in layers:
            write(n, ints[n])

    write_all(clean)
    base_acc = M.accuracy(model, tok, rows, dev)
    base_ppl = ppl_now()
    base_bench = bench_eval() if bench else {}
    print(f"[base] INT{args.bits} MMLU = {base_acc:.2f}%  PPL = {base_ppl:.3f}  "
          f"({len(layers)} layers, {tot:,} weights)", flush=True)
    for b in bench:
        print(f"[base] {b} = {base_bench[b]:.2f}%", flush=True)

    # --- attacker: gradients + scores (needed by 'grad'/'probe' proxies too) ---
    ab = AB.AttentionBreaker(model, layers, write, clean, batches, device=dev,
                             alpha=args.alpha, seed=args.seed,
                             layer_filter=[f for f in args.layer_filter.split(",") if f])
    t0 = time.time()
    ab.compute_gradients()
    print(f"[ab] gradients + scores for {len(ab.cands)} attn/mlp layers ({time.time()-t0:.0f}s)", flush=True)

    pcts = [float(p) for p in args.sens_pcts.split(",") if p]
    sens_rows, seeds, ranked = ab.layer_sensitivity(pcts, args.layer_thresh)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + "_layer_sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "pct", "flips", "attack_loss"])
        for r in sens_rows:
            w.writerow(r)
    print(f"[ab] seed layers (loss >= {args.layer_thresh}, fewest flips first):")
    for n, (k, l, p) in ranked[:10]:
        print(f"      {n}: {k:,} flips ({p}%) -> loss {l:.3f}")
    if not seeds:
        sys.exit("[ab] no layer crossed --layer-thresh; lower it or raise --sens-pcts")

    # --- the selective code layer selection --------------------------------------------------
    if args.sens_proxy == "l1":
        score = {n: float(np.sum(np.abs(dequant_any(clean[n], layers[n][1], layers[n][2])))) for n in layers}
    elif args.sens_proxy == "grad":
        score = {n: float(np.sum(np.abs(ab.grads[n]) * np.abs(dequant_any(clean[n], layers[n][1], layers[n][2]).reshape(-1))))
                 if n in ab.grads else 0.0 for n in layers}
    else:  # probe: max attack loss seen for that layer at the smallest pct
        probe = {}
        for (n, p, k, l) in sens_rows:
            probe[n] = max(probe.get(n, 0.0), l / max(k, 1))
        score = {n: probe.get(n, 0.0) for n in layers}
    order = sorted(score, key=lambda n: -score[n])
    sens, cov = set(), 0
    for n in order:
        if cov + layers[n][0].size > args.sens_frac * tot and sens:
            break
        sens.add(n); cov += layers[n][0].size
    print(f"[protect] proxy={args.sens_proxy}: {len(sens)} layers = {cov:,} wts = {100*cov/tot:.1f}%", flush=True)
    print(f"[protect] seed layer {seeds[0]} is {'INSIDE' if seeds[0] in sens else 'OUTSIDE'} the protected set")

    code = ENG.XLayerECC(strong_t=args.strong_t, light_t=args.light_t, bch_m=args.bch_m,
                         delta=args.delta, host_key=args.host_key)
    code.host_bits = args.host_bits; code.plane_bits = args.plane_bits
    code.bit_width = int(args.bits); code.embed_mode = args.embed; code.stdm_L = args.stdm_L
    Lf = args.stdm_L if args.embed == "stdm" else 1
    t0 = time.time()
    selcode = SelCodeLLM(code, layers, clean, sens, score, args.plane_bits, args.host_bits, Lf,
                     interleave=bool(args.interleave), seed=args.seed)
    print(f"[selcode] grouping={'INTERLEAVED' if args.interleave else 'ADJACENT'}")
    G_any = next(iter(selcode.parity.values()))[2] if selcode.parity else 0
    print(f"[selcode] codeword: G={G_any} weights x K={args.plane_bits} planes, BCH t={selcode.t} "
          f"-> sign-only attacks {'CANNOT' if G_any <= selcode.t else 'CAN'} overload a codeword "
          f"(G {'<=' if G_any <= selcode.t else '>'} t)", flush=True)
    if G_any > selcode.t:
        print("[selcode] WARNING: G > t. A sign-bit attacker can defeat a codeword with G flips. "
              "Raise --strong-t or --bch-m, or raise --plane-bits, to restore G <= t.", flush=True)
    print(f"[selcode] parity {len(selcode.payload):,} bits in {len(selcode.seg)} carriers, "
          f"clean parity BER {selcode.parity_ber_clean*100:.3f}%, overflow={'YES' if selcode.overflow else 'no'} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # --- adaptive attacker: loss measured on the corrected model --------------
    if args.adaptive:
        cur = {"sol": None}
        base_loss_fn = ab.loss_fn

        def defended_loss():
            flips = ab._group(cur["sol"]) if cur["sol"] else {}
            att = ab.attacked_ints(cur["sol"] or [])
            rec = selcode.correct(att, flips)
            for n in flips:
                write(n, rec[n])
            l = base_loss_fn()
            return l

        def loss_of(sol):
            cur["sol"] = sol
            grp = ab._group(sol)
            l = defended_loss()
            ab.n_loss_evals += 1
            for n in grp:
                write(n, clean[n])
            return l
        ab.loss_of = loss_of
        print("[ab] ADAPTIVE: attacker's loss is measured after the selective code correction", flush=True)

    # --- build solutions to evaluate --------------------------------------------
    pop = ab.population_space(seeds[:args.n_seed_layers], args.pop_pct)
    print(f"[ab] population space: {len(pop):,} weights from {seeds[:args.n_seed_layers]}", flush=True)
    solutions = []                                     # (tag, solution)
    if args.attack == "genbfa":
        best, hist = ab.genbfa(pop, args.loss_thresh, args.gens, args.init_sols,
                               args.per_gen, args.patience)
        with open(args.out + "_genbfa_progress.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["generation", "n_flips", "loss", "fitness"])
            for r in hist:
                w.writerow(r)
        with open(args.out + "_critical_weights.json", "w") as f:
            json.dump([{"layer": n, "index": int(i)} for n, i in best], f, indent=1)
        solutions.append((f"genbfa_{len(best)}", best))
    elif args.attack == "fixed":
        for N in [int(x) for x in args.fixed_flips.split(",") if x]:
            solutions.append((f"fixed_{N}", [(n, i) for (n, i, _s) in pop[:N]]))
    else:
        # codeword-aware: the attacker knows G (weights per codeword) and t, and
        # overloads whole codewords with F = t+1 bit errors. Candidates are the
        # codewords of the SEED layers that contain population-space weights
        # (the same weights the real attack uses); each candidate's damage is
        # MEASURED (true attack loss with that one codeword broken), not scored,
        # so the ranking is comparable. Then the top-N are combined.
        F = args.cw_flips if args.cw_flips > 0 else selcode.t + 1
        pop_by_layer = {}
        for (n, i, sc_) in pop:
            pop_by_layer.setdefault(n, []).append((i, sc_))
        cw_cands = []                                   # (score_sum, layer, cw, entries)
        for n, items in pop_by_layer.items():
            if n not in selcode.parity:
                print(f"[cwaware] seed layer {n} is unprotected; use --attack fixed there")
                continue
            G = selcode.parity[n][2]
            sc = ab.scores[n]
            per_plane = []; left = F; plane = 7
            while left > 0 and plane >= 8 - args.plane_bits:
                k = min(left, G); per_plane.append((plane, k)); left -= k; plane -= 1
            if left > 0:
                continue
            knows = bool(args.attacker_knows_key) or not args.interleave
            cws = {}
            for (i, sc_) in items:
                c = int(selcode.pos_of(n, i) // G) if knows else int(i // G)
                cws.setdefault(c, 0.0); cws[c] += sc_
            for c, ssum in cws.items():
                base = (selcode.members(n, c) if knows
                        else np.arange(c * G, min((c + 1) * G, sc.size)))
                if base.size < G:
                    continue
                base = base[np.argsort(-sc[base])]      # best weights of the codeword first
                entries = []
                for (p, k) in per_plane:
                    entries += [(n, int(i), p) for i in base[:k]]
                cw_cands.append((ssum, n, int(c), entries))
        cw_cands.sort(key=lambda r: -r[0])
        cw_cands = cw_cands[:args.cw_probe]
        # measure the true loss of each candidate codeword on its own
        cw_rank = []
        for (ssum, n, c, entries) in cw_cands:
            l = ab.loss_of(entries)
            cw_rank.append((l, n, c, entries))
        cw_rank.sort(key=lambda r: -r[0])
        G0 = selcode.parity[seeds[0]][2] if seeds[0] in selcode.parity else "?"
        print(f"[cwaware] attacker {'KNOWS' if knows else 'does NOT know'} the codeword layout")
        print(f"[cwaware] {len(cw_rank):,} candidate codewords in seed layers measured, "
              f"{F} bit errors each (t={selcode.t}, G={G0}); best single codeword loss="
              f"{cw_rank[0][0] if cw_rank else float('nan'):.3f}", flush=True)
        with open(args.out + "_codeword_rank.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["layer", "codeword", "single_cw_loss"])
            for (l, n, c, _e) in cw_rank:
                w.writerow([n, c, f"{l:.4f}"])
        # GREEDY ACCUMULATION: add codewords in ranked order, keep each only if
        # the combined attack loss rises. Damage is then monotone in the number
        # of codewords, and the attacker is stronger than a plain top-N cut.
        counts = sorted(int(x) for x in args.cw_counts.split(",") if x)
        kept, cur_loss, cur_sol = [], ab.loss_of([]), []
        ci = 0
        for (l1, n, c, entries) in cw_rank:
            if len(kept) >= counts[-1]:
                break
            trial = cur_sol + entries
            l = ab.loss_of(trial)
            if l > cur_loss:
                kept.append((l, n, c, entries)); cur_sol = trial; cur_loss = l
                if len(kept) in counts:
                    solutions.append((f"cwaware_{len(kept)}cw_{len(cur_sol)}", list(cur_sol)))
        print(f"[cwaware] greedy kept {len(kept)} of {len(cw_rank)} codewords; "
              f"final loss {cur_loss:.3f} with {len(cur_sol)} bit errors", flush=True)
        for N in counts:
            if N > len(kept) and kept:
                print(f"[cwaware] only {len(kept)} useful codewords found; N={N} not reached")

    # --- evaluate each solution: unprotected vs the selective code-corrected ----------------
    fout = open(args.out + ".csv", "w", newline="")
    wcsv = csv.writer(fout)
    wcsv.writerow(["tag", "n_flips", "attack_loss", "base_mmlu", "unprotected_mmlu",
                   "protected_mmlu", "base_ppl", "unprotected_ppl", "protected_ppl",
                   "flips_in_protected_layers", "max_flips_per_codeword",
                   "bch_t", "correctable_by_code", "adaptive", "sens_proxy", "loss_evals",
                   "interleave", "attacker_knows_key"])
    for tag, sol in solutions:
        flips = ab._group(sol)
        att = ab.attacked_ints(sol)
        write_all(att)
        loss = AB.lm_loss(model, batches)
        un = M.accuracy(model, tok, rows, dev); un_ppl = ppl_now()
        un_bench = bench_eval() if bench else {}
        demo_att = {q: generate(q) for q in demo} if demo else {}
        rec = selcode.correct(att, flips)
        n_wrong = sum(int((rec[n] != clean[n]).sum()) for n in layers)
        write_all(rec)
        pr = M.accuracy(model, tok, rows, dev); pr_ppl = ppl_now()
        pr_bench = bench_eval() if bench else {}
        demo_san = {q: generate(q) for q in demo} if demo else {}
        print(f"[{tag}] weights still differing from clean after the selective code: {n_wrong}", flush=True)
        for q in demo:
            demo_rows.append((tag, q, demo_clean[q], demo_att[q], demo_san[q]))
        write_all(clean)
        stats = selcode.codeword_stats(flips)
        inside = sum(r[2] for r in stats if r[1])          # bit errors inside protected layers
        mx = max([r[4] for r in stats if r[1]] + [0])
        ok = all(r[5] for r in stats if r[1]) and inside == len(sol)
        print(f"[{tag}] flips={len(sol)} loss={loss:.3f} | MMLU base={base_acc:.2f} "
              f"unprotected={un:.2f} protected={pr:.2f} | PPL base={base_ppl:.2f} "
              f"unprotected={un_ppl:.2f} protected={pr_ppl:.2f} | {inside}/{len(sol)} inside protected "
              f"layers, max {mx} flips/codeword (t={selcode.t}) -> "
              f"{'correctable' if ok else 'NOT fully correctable'}", flush=True)
        wcsv.writerow([tag, len(sol), f"{loss:.4f}", f"{base_acc:.2f}", f"{un:.2f}", f"{pr:.2f}",
                       f"{base_ppl:.3f}", f"{un_ppl:.3f}", f"{pr_ppl:.3f}", inside, mx, selcode.t, int(ok), int(args.adaptive), args.sens_proxy,
                       ab.n_loss_evals, int(args.interleave), int(args.attacker_knows_key)]); fout.flush()
        for b in bench:
            print(f"[{tag}] {b}: base={base_bench.get(b, float('nan')):.2f} "
                  f"unprot={un_bench[b]:.2f} prot={pr_bench[b]:.2f}", flush=True)
        with open(args.out + f"_{tag}_flips.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["layer", "index", "protected_layer", "plane"])
            for e in sol:
                w.writerow([e[0], int(e[1]), int(e[0] in sens), int(e[2]) if len(e) > 2 else 7])
    fout.close()
    if demo_rows:
        with open(args.out + "_outputs.md", "w") as f:
            f.write("| Attack | Prompt | Before attack | After attack | After the selective code |\n|---|---|---|---|---|\n")
            for (tg, q, b, a, sn) in demo_rows:
                f.write(f"| {tg} | {q} | {b} | {a} | {sn} |\n")
        with open(args.out + "_outputs.tex", "w") as f:
            f.write("% needs \\usepackage[table]{xcolor} and booktabs\n"
                    "\\begin{table*}[t]\\centering\\small\n"
                    "\\caption{Model outputs before the attack, after the attack, and after the selective code correction (" + args.model + "). "
                    "Green: clean or recovered; red: attacked.}\n"
                    "\\begin{tabular}{p{0.16\\linewidth}"
                    ">{\\columncolor{green!10}}p{0.26\\linewidth}"
                    ">{\\columncolor{red!10}}p{0.26\\linewidth}"
                    ">{\\columncolor{green!10}}p{0.26\\linewidth}}\n"
                    "\\toprule\n\\textbf{Prompt} & \\textbf{Before attack} & \\textbf{After attack} & \\textbf{After the selective code} \\\\\n\\midrule\n")
            last = None
            for (tg, q, b, a, sn) in demo_rows:
                if last is not None and tg != last:
                    f.write("\\midrule\n")
                last = tg
                f.write(f"{esc(q)} & {esc(b)} & {esc(a)} & {esc(sn)} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
        print(f"[demo] wrote {args.out}_outputs.md and _outputs.tex")
        for (tg, q, b, a, sn) in demo_rows:
            print(f"[demo:{tg}] Q: {q}\n   before: {b[:120]}\n   attack: {a[:120]}\n   selcode : {sn[:120]}")
    print(f"[done] wrote {args.out}.csv")


if __name__ == "__main__":
    main()
