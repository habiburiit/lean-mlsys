#!/usr/bin/env python
"""
lean_sensreg_llm.py  -  sensitivity-regularized fine-tuning for LLMs (the selective code / SKEW)

Same loss as the vision script, applied to a causal LM on WikiText-2:

    L = LM_CE + lambda * ( P_tail + 100 * P_clip )
    d_i    = |dL/dw_i| * 128 * max|W_layer| / 127        (INT8 sign-bit flip damage)
    P_tail = mean of the top 1% of d_i  (double backprop, every --tail-every steps)
    P_clip = sum relu(|w| - q99_layer)^2 / n

Weights considered: all Linear weights inside the transformer blocks
(embed_tokens and lm_head excluded, as in the the selective code LLM pipeline).

Subcommands: train, hist, plot, ptq, flip   (same meaning as lean_sensreg.py;
metric is WikiText-2 perplexity, lower is better)

Usage (All_final, ANON):
  python lean_sensreg_llm.py train --model Qwen/Qwen2-0.5B --lam 10 --out ckpt_llm/qwen2_0.5b_lam10
  python lean_sensreg_llm.py hist  --model Qwen/Qwen2-0.5B --ckpt2 ckpt_llm/qwen2_0.5b_lam10 --out res_llm/qwen2_0.5b/hist_lam10
"""
import argparse, json, os, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


# ----------------------------------------------------------------- model / data
def load_model(name_or_path, dev, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=dtype, attn_implementation="eager").to(dev)
    model.config.use_cache = False
    return model, tok


def weight_params(model, exclude=("embed_tokens", "lm_head", "wte", "wpe")):
    out = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and not any(e in n for e in exclude):
            out.append((n + ".weight", m.weight))
    return out


def wikitext_blocks(tok, split, seq_len, max_tokens=None, name="wikitext-2-raw-v1"):
    from datasets import load_dataset
    ds = load_dataset("wikitext", name, split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    if max_tokens:
        ids = ids[:max_tokens]
    n = (ids.numel() // seq_len) * seq_len
    return ids[:n].view(-1, seq_len)


@torch.no_grad()
def perplexity(model, blocks, dev, bs=8):
    model.eval(); nll = 0.0; cnt = 0
    for i in range(0, blocks.size(0), bs):
        x = blocks[i:i+bs].to(dev)
        out = model(input_ids=x, labels=x)
        nll += out.loss.item() * x.numel(); cnt += x.numel()
    return math.exp(nll / cnt)


def msb_step(w):
    return 128.0 * w.detach().abs().max() / 127.0


# ----------------------------------------------------------------- penalties
def tail_penalty(loss, params, tail_frac):
    grads = torch.autograd.grad(loss, [p for _, p in params], create_graph=True)
    scores = torch.cat([(g.abs() * msb_step(p)).flatten() for g, (_, p) in zip(grads, params)])
    k = max(1, int(tail_frac * scores.numel()))
    return torch.topk(scores, k, sorted=False).values.sum() / 1e4   # sum-scaled, not mean


def clip_penalty(params, q):
    pen = 0.0
    for _, p in params:
        a = p.abs()
        flat = a.detach().flatten()
        thr = torch.quantile(flat[torch.randint(0, flat.numel(), (min(flat.numel(), 1_000_000),), device=flat.device)], q)
        pen = pen + F.relu(a - thr).pow(2).sum() / a.numel()
    return pen


# ----------------------------------------------------------------- hessian diagonal (Hutchinson) + hybrid score
def hutch_diag(model, params, x, probes=2):
    """diag(H) estimate: E[z * (H z)], z Rademacher; one double-backprop per probe. Returns list of detached tensors."""
    plist = [p for _, p in params]
    acc = [torch.zeros_like(p) for p in plist]
    for _ in range(probes):
        loss = model(input_ids=x, labels=x).loss
        grads = torch.autograd.grad(loss, plist, create_graph=True)
        zs = [torch.randint_like(p, 2) * 2 - 1 for p in plist]
        gz = sum((g * z).sum() for g, z in zip(grads, zs))
        hz = torch.autograd.grad(gz, plist)
        for a_, h, z in zip(acc, hz, zs): a_ += (h * z).detach()
        model.zero_grad(set_to_none=True)
    return [a_ / probes for a_ in acc]


def damage(g, p, hd=None, alpha=0.0):
    """per-weight sign-bit-flip damage. first-order: |g|*dw ; hybrid: |g*dw + 0.5*H_ii*dw^2| (+ alpha*|w|)"""
    dw = msb_step(p)
    if hd is None:
        d = g.abs() * dw
    else:
        d = (g * dw + 0.5 * hd * dw * dw).abs()
    if alpha > 0: d = d + alpha * p.detach().abs()
    return d


# ----------------------------------------------------------------- score
def score_model(model, tok, params, dev, seq_len, probe_blocks, score="first", probes=2):
    """per-weight damage averaged over probe blocks from the TEST split (first-order or hybrid)"""
    blocks = wikitext_blocks(tok, "test", seq_len)[:probe_blocks]
    model.eval(); acc = None; n = 0
    hd = hutch_diag(model, params, blocks[:1].to(dev), probes) if score == "hybrid" else None
    for i in range(blocks.size(0)):
        x = blocks[i:i+1].to(dev)
        loss = model(input_ids=x, labels=x).loss
        grads = torch.autograd.grad(loss, [p for _, p in params])
        s = torch.cat([damage(g, p, None if hd is None else hd[j]).flatten().cpu()
                       for j, (g, (_, p)) in enumerate(zip(grads, params))])
        acc = s if acc is None else acc + s; n += 1
        model.zero_grad(set_to_none=True)
    return (acc / n).numpy()


# ----------------------------------------------------------------- train
def cmd_train(a):
    dev = "cuda"
    torch.manual_seed(a.seed)
    model, tok = load_model(a.init or a.model, dev)
    params = weight_params(model)
    print(f"[model] {a.model}  regularized weights: {sum(p.numel() for _, p in params)/1e6:.1f}M in {len(params)} linears")
    train_blocks = wikitext_blocks(tok, "train", a.seq_len, a.max_train_tokens)
    test_blocks = wikitext_blocks(tok, "test", a.seq_len, a.max_eval_tokens)
    print(f"[data] train blocks {train_blocks.size(0)} x {a.seq_len}, eval tokens {test_blocks.numel()}")
    print(f"[init] ppl = {perplexity(model, test_blocks, dev):.3f}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    steps = a.epochs * (train_blocks.size(0) // a.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1))
    it = 0; hd = None
    plist = [p for _, p in params]; sizes = [p.numel() for p in plist]; offs = np.cumsum([0] + sizes)
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(train_blocks.size(0)); t0 = time.time(); ce_s = pen_s = 0.0; nb = 0
        for b in range(0, train_blocks.size(0) - a.bs + 1, a.bs):
            x = train_blocks[perm[b:b+a.bs]].to(dev)
            if a.score == "hybrid" and (hd is None or it % a.hutch_every == 0):
                hd = hutch_diag(model, params, x[:1], a.hutch_probes)
            ce = model(input_ids=x, labels=x).loss
            pen = torch.zeros((), device=dev)
            if a.reg in ("tail", "both") and it % a.tail_every == 0:
                pen = pen + tail_penalty(ce, params, a.tail_frac) * a.tail_every
            if a.reg in ("clip", "both", "flip"):
                pen = pen + a.clip_w * clip_penalty(params, a.clip_q)
            loss = ce + a.lam * pen
            opt.zero_grad(set_to_none=True); loss.backward()
            if a.reg == "flip":
                # flip-aware: pick the currently most damaging weights (by score), apply their sign-bit flips
                # as a detached perturbation, add the loss under those flips (weight mu). One extra fwd+bwd.
                with torch.no_grad():
                    sc = torch.cat([damage(p.grad, p, None if hd is None else hd[j]).flatten()
                                    for j, p in enumerate(plist)])
                    kpool = max(a.flip_m, int(a.tail_frac * sc.numel()))
                    pool = torch.topk(sc, kpool, sorted=False).indices
                    sel = pool[torch.randperm(pool.numel(), device=dev)[:a.flip_m]]
                    deltas = []
                    for j, p in enumerate(plist):
                        idx = sel[(sel >= offs[j]) & (sel < offs[j+1])] - offs[j]
                        if idx.numel() == 0: deltas.append(None); continue
                        flat = p.data.view(-1); dw = msb_step(p)
                        d = torch.where(flat[idx] >= 0, -dw, dw)           # sign-bit flip direction
                        flat[idx] += d; deltas.append((idx, d))
                ce_flip = model(input_ids=x, labels=x).loss
                (a.mu * ce_flip).backward()
                with torch.no_grad():
                    for p, dd in zip(plist, deltas):
                        if dd is not None: p.data.view(-1)[dd[0]] -= dd[1]
                pen = pen + ce_flip.detach()
            opt.step(); sched.step(); it += 1
            ce_s += ce.item(); pen_s += pen.item(); nb += 1
            if it % a.log_every == 0:
                print(f"  it {it}/{steps} ce={ce_s/nb:.3f} pen={pen_s/nb:.3e} {time.time()-t0:.0f}s", flush=True)
        ppl = perplexity(model, test_blocks, dev)
        print(f"ep {ep+1}/{a.epochs}  ce={ce_s/nb:.3f}  pen={pen_s/nb:.3e}  ppl={ppl:.3f}  {time.time()-t0:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"[saved] {a.out}  final ppl {ppl:.3f}")


# ----------------------------------------------------------------- hist / plot
def _scores_for(path, a, dev):
    model, tok = load_model(path, dev); params = weight_params(model)
    s = score_model(model, tok, params, dev, a.seq_len, a.probe_blocks, a.score, a.hutch_probes)
    ppl = perplexity(model, wikitext_blocks(tok, "test", a.seq_len, a.max_eval_tokens), dev)
    del model; torch.cuda.empty_cache()
    return s, ppl


def cmd_hist(a):
    dev = "cuda"
    s1, p1 = _scores_for(a.model, a, dev)
    hi = float(np.quantile(s1, 0.99)); mid = float(np.quantile(s1, 0.91))
    def bins3(s):
        h = float((s >= hi).mean()); m = float(((s >= mid) & (s < hi)).mean())
        return {"HIGH": 100*h, "MID": 100*m, "LOW": 100*(1-h-m)}
    out = {"model": a.model, "ppl": p1, "n_weights": int(s1.size), "thr_high": hi, "thr_mid": mid,
           "tail_mass_top1pct": float(np.sort(s1)[-int(0.01*s1.size):].sum()/s1.sum()),
           "bins_pct": {"baseline": bins3(s1)}}
    if a.ckpt2:
        s2, p2 = _scores_for(a.ckpt2, a, dev)
        out["ckpt2"] = a.ckpt2; out["ppl2"] = p2
        out["tail_mass_top1pct2"] = float(np.sort(s2)[-int(0.01*s2.size):].sum()/s2.sum())
        out["bins_pct"]["regularized"] = bins3(s2)
        out["ratio_high"] = out["bins_pct"]["regularized"]["HIGH"] / max(out["bins_pct"]["baseline"]["HIGH"], 1e-9)
    print("\nBin occupancy (% of weights, fixed thresholds from baseline):")
    print(f"{'':14s}{'HIGH':>8s}{'MID':>8s}{'LOW':>8s}{'PPL':>9s}")
    for name, b in out["bins_pct"].items():
        print(f"{name:14s}{b['HIGH']:8.3f}{b['MID']:8.3f}{b['LOW']:8.3f}{(p1 if name=='baseline' else p2):9.3f}")
    os.makedirs(a.out, exist_ok=True); json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 3.2)); names = list(out["bins_pct"]); w = 0.8/len(names)
        for i, nm in enumerate(names):
            v = [out["bins_pct"][nm][k] for k in ("HIGH", "MID", "LOW")]
            bars = ax.bar(np.arange(3)+i*w, v, w, label=nm)
            for b_, vv in zip(bars, v): ax.text(b_.get_x()+b_.get_width()/2, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(np.arange(3)+w*(len(names)-1)/2); ax.set_xticklabels(["HIGH", "MID", "LOW"]); ax.set_yscale("log")
        ax.set_ylabel("% of weights"); ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(a.out, "bins_hml.png"), dpi=200)
    except Exception as e:
        print("[plot skipped]", e)


def cmd_plot(a):
    dev = "cuda"
    labels = a.labels or [os.path.basename(c.rstrip("/")) for c in a.ckpts]
    scores, ppls = [], []
    for c in a.ckpts:
        s, p = _scores_for(c, a, dev); scores.append(s); ppls.append(p); print(f"[scored] {c} ppl={p:.3f}")
    smax = float(np.quantile(scores[0], 0.9999))
    rel = [np.clip(100.0*s/smax, 1e-2, 100) for s in scores]
    edges = np.logspace(-2, 2, a.nbins+1)
    counts = [np.histogram(r, edges)[0] for r in rel]
    print(f"\n{'bin':>4s} {'sens range':>16s} " + "".join(f"{l[:14]:>16s}" for l in labels))
    for i in range(a.nbins):
        print(f"{i+1:4d} {edges[i]:7.2f}-{edges[i+1]:<8.2f} " + "".join(f"{int(c[i]):16d}" for c in counts))
    print("\nweights in top 3 bins: " + ", ".join(f"{l}={int(c[-3:].sum())}" for l, c in zip(labels, counts)))
    os.makedirs(a.out, exist_ok=True)
    json.dump({"labels": labels, "ppl": ppls, "edges": edges.tolist(), "counts": [c.tolist() for c in counts]},
              open(os.path.join(a.out, "quantized_hist.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.6)); n = len(counts); w = 0.8/n; x = np.arange(a.nbins)
        for i, (c, l, p) in enumerate(zip(counts, labels, ppls)):
            ax.bar(x+i*w, np.maximum(c, 0.5), w, label=f"{l} (ppl {p:.2f})")
        ax.set_yscale("log"); ax.set_ylabel("number of weights"); ax.set_xlabel("quantized sensitivity bin (log, 100 = baseline max)")
        ax.set_xticks(x+w*(n-1)/2); ax.set_xticklabels([str(i+1) for i in range(a.nbins)], fontsize=7); ax.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(a.out, "quantized_sens_hist.png"), dpi=200)
    except Exception as e:
        print("[plot skipped]", e)


# ----------------------------------------------------------------- ptq / flip
def quantize_(params, bits, per_channel=False):
    qmax = 2**(bits-1) - 1; store = []
    with torch.no_grad():
        for n, p in params:
            if per_channel:
                scale = p.abs().amax(dim=1, keepdim=True) / qmax
            else:
                scale = p.abs().max() / qmax
            scale = torch.clamp(scale, min=1e-12)
            q = torch.round(p / scale).clamp(-qmax-1, qmax)
            p.copy_(q * scale); store.append((n, p, q, scale))
    return store


def cmd_ptq(a):
    dev = "cuda"; labels = a.labels or [os.path.basename(c.rstrip("/")) for c in a.ckpts]
    bits_list = [int(b) for b in a.bits.split(",")]; res = {}
    print(f"\n{'model':16s}{'fp32':>9s}" + "".join(f"{'int'+str(b):>9s}" for b in bits_list) + f"   (WikiText-2 ppl, {'per-channel' if a.per_channel else 'per-tensor'})")
    for c, l in zip(a.ckpts, labels):
        model, tok = load_model(c, dev); params = weight_params(model)
        tb = wikitext_blocks(tok, "test", a.seq_len, a.max_eval_tokens)
        row = {"fp32": perplexity(model, tb, dev)}
        base = {n: p.detach().clone() for n, p in params}
        for b in bits_list:
            with torch.no_grad():
                for n, p in params: p.copy_(base[n])
            quantize_(params, b, a.per_channel); row[f"int{b}"] = perplexity(model, tb, dev)
        res[l] = row; print(f"{l:16s}{row['fp32']:9.2f}" + "".join(f"{row['int'+str(b)]:9.2f}" for b in bits_list), flush=True)
        del model; torch.cuda.empty_cache()
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(os.path.join(a.out, "ptq.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for l in labels: ax.plot(bits_list, [res[l][f"int{b}"] for b in bits_list], marker="o", label=l)
        ax.set_yscale("log"); ax.invert_xaxis(); ax.set_xlabel("weight bits"); ax.set_ylabel("perplexity (log)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(a.out, "ptq_ppl.png"), dpi=200)
    except Exception as e:
        print("[plot skipped]", e)


def cmd_flip(a):
    dev = "cuda"; labels = a.labels or [os.path.basename(c.rstrip("/")) for c in a.ckpts]
    nflips = [int(x) for x in a.nflips.split(",")]; res = {}
    print(f"\n{'model':16s}" + "".join(f"{'n='+str(n):>10s}" for n in nflips) + "   (WikiText-2 ppl after targeted INT8 sign-bit flips)")
    for c, l in zip(a.ckpts, labels):
        model, tok = load_model(c, dev); params = weight_params(model)
        tb = wikitext_blocks(tok, "test", a.seq_len, a.max_eval_tokens)
        s_all = score_model(model, tok, params, dev, a.seq_len, a.probe_blocks, a.score, a.hutch_probes)
        order = torch.from_numpy(np.argsort(-s_all))
        base = {n: p.detach().clone() for n, p in params}
        row = {}
        for n_ in nflips:
            with torch.no_grad():
                for n, p in params: p.copy_(base[n])
            store = quantize_(params, 8); idx = order[:n_].sort().values; off = 0
            with torch.no_grad():
                for _, p, q, scale in store:
                    sel = idx[(idx >= off) & (idx < off + q.numel())] - off
                    if sel.numel():
                        qf = q.flatten().clone()
                        qf[sel] = torch.where(qf[sel] >= 0, qf[sel] - 128, qf[sel] + 128)
                        p.copy_((qf * scale).view_as(p))   # per-tensor scale (scalar)
                    off += q.numel()
            row[str(n_)] = perplexity(model, tb, dev)
        res[l] = row; print(f"{l:16s}" + "".join(f"{row[str(n_)]:10.2f}" for n_ in nflips), flush=True)
        del model; torch.cuda.empty_cache()
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(os.path.join(a.out, "flip.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for l in labels: ax.plot(nflips, [res[l][str(n_)] for n_ in nflips], marker="o", label=l)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("targeted sign-bit flips (INT8, no ECC)"); ax.set_ylabel("perplexity (log)")
        ax.legend(fontsize=7); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(a.out, "flip_targeted.png"), dpi=200)
    except Exception as e:
        print("[plot skipped]", e)


# ----------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for s in ("train", "hist", "plot", "ptq", "flip"):
        p = sub.add_parser(s)
        p.add_argument("--model", default="Qwen/Qwen2-0.5B", help="HF name or local path of the BASELINE")
        p.add_argument("--seq-len", type=int, default=512)
        p.add_argument("--max-eval-tokens", type=int, default=40000)
        p.add_argument("--probe-blocks", type=int, default=16)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--score", choices=["first", "hybrid"], default="first", help="damage score: first-order or hybrid (with Hessian diagonal)")
        p.add_argument("--hutch-probes", type=int, default=2)
    t = sub.choices["train"]
    t.add_argument("--init", default="", help="start from this checkpoint instead of --model")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--bs", type=int, default=4)
    t.add_argument("--lr", type=float, default=5e-6)
    t.add_argument("--max-train-tokens", type=int, default=1_000_000)
    t.add_argument("--reg", choices=["tail", "clip", "both", "flip", "none"], default="both",
                   help="flip = flip-aware training (loss under actual sign-bit flips of the top weights)")
    t.add_argument("--mu", type=float, default=1.0, help="weight of the flipped-loss term (reg=flip)")
    t.add_argument("--flip-m", type=int, default=64, help="number of weights flipped per step (reg=flip)")
    t.add_argument("--hutch-every", type=int, default=50, help="refresh Hessian diagonal every N steps (score=hybrid)")
    t.add_argument("--lam", type=float, default=10.0)
    t.add_argument("--tail-frac", type=float, default=0.01)
    t.add_argument("--tail-every", type=int, default=4)
    t.add_argument("--clip-q", type=float, default=0.99)
    t.add_argument("--clip-w", type=float, default=100.0)
    t.add_argument("--log-every", type=int, default=50)
    h = sub.choices["hist"]; h.add_argument("--ckpt2", default=""); h.add_argument("--out", default="res_llm/hist")
    pl = sub.choices["plot"]; pl.add_argument("--ckpts", nargs="+", required=True); pl.add_argument("--labels", nargs="*"); pl.add_argument("--nbins", type=int, default=20); pl.add_argument("--out", default="res_llm/plot")
    q = sub.choices["ptq"]; q.add_argument("--ckpts", nargs="+", required=True); q.add_argument("--labels", nargs="*"); q.add_argument("--bits", default="8,6,5,4"); q.add_argument("--per-channel", action="store_true"); q.add_argument("--out", default="res_llm/ptq")
    f = sub.choices["flip"]; f.add_argument("--ckpts", nargs="+", required=True); f.add_argument("--labels", nargs="*"); f.add_argument("--nflips", default="1,2,3,5,10,20,50"); f.add_argument("--out", default="res_llm/flip")
    a = ap.parse_args()
    {"train": cmd_train, "hist": cmd_hist, "plot": cmd_plot, "ptq": cmd_ptq, "flip": cmd_flip}[a.cmd](a)
