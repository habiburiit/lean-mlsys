#!/usr/bin/env python
"""
lean_sensreg.py  -  sensitivity-regularized training for the selective code

Two subcommands:

  train   fine-tune a CIFAR-10 ResNet-18 with a regularizer that shrinks the
          high-sensitivity tail of the weights, save a plain state_dict
  hist    score every weight of one or two checkpoints with the INT8 MSB-flip
          damage proxy, bin them, save histogram PNG + JSON summary

Per-weight damage proxy (first order, same spirit as run_binned_protect.py):
    s_i = |g_i| * dW_msb_i
    dW_msb_i = 128 * scale_layer,  scale_layer = max|W_layer| / 127   (per-tensor INT8)
i.e. how much the loss moves if the sign bit of weight i flips.

Regularizers (--reg):
    tail   penalize the mean of the top-k proxy scores (double backprop, direct
           attack on the tail).  k = --tail-frac of all weights.
    clip   penalize |w| above the per-layer --clip-q quantile.  Shrinks the
           INT8 scale, so every MSB flip moves the weight less (Stutz et al.
           MLSys 2021 style).  Cheap, no double backprop.
    both   tail + clip

Usage (in All_final, ANON env):
    python lean_sensreg.py train --init resnet18_cifar10.pth --reg both \
        --lam 1e-3 --epochs 10 --out resnet18_c10_sensreg.pth
    python lean_sensreg.py hist --ckpt resnet18_cifar10.pth \
        --ckpt2 resnet18_c10_sensreg.pth --out sens_hist
"""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T


# ----------------------------------------------------------------- model / data
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "imagenet": 1000}
_DS = "cifar10"   # set from --dataset at startup


def build_model(arch="resnet18", num_classes=None):
    num_classes = num_classes or NUM_CLASSES[_DS]
    if _DS == "imagenet":
        # standard torchvision ResNet; pretrained weights are the "baseline"
        return getattr(torchvision.models, arch)(weights="DEFAULT")
    # CIFAR-style ResNet (3x3 stem, no maxpool).  If your the selective code checkpoint
    # comes from a different definition, import that class here instead.
    m = getattr(torchvision.models, arch)(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def load_state_flex(model, path):
    if path in ("", "pretrained"):
        return model   # ImageNet: torchvision weights already loaded in build_model
    sd = torch.load(path, map_location="cpu")
    for k in ("state_dict", "model", "net", "model_state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)} "
              f"(first missing: {missing[:3]})")
    return model


IMAGENET_ROOT = "./imagenet-val"


def cifar_loaders(root, bs, workers=4):
    """returns (train_loader, test_loader) for the dataset set by --dataset"""
    if _DS in ("cifar10", "cifar100"):
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        if _DS == "cifar100":
            mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize(mean, std)])
        te = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        cls = torchvision.datasets.CIFAR10 if _DS == "cifar10" else torchvision.datasets.CIFAR100
        train = cls(root, True, tr, download=True)
        test = cls(root, False, te, download=True)
        return (torch.utils.data.DataLoader(train, bs, shuffle=True, num_workers=workers, pin_memory=True),
                torch.utils.data.DataLoader(test, 512, shuffle=False, num_workers=workers, pin_memory=True))
    # imagenet: only the val set is available, so split it 40k fine-tune / 10k held-out (fixed seed)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tr = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize(mean, std)])
    te = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])
    full_tr = torchvision.datasets.ImageFolder(IMAGENET_ROOT, tr)
    full_te = torchvision.datasets.ImageFolder(IMAGENET_ROOT, te)
    n = len(full_tr); g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g).tolist()
    n_test = min(10000, n // 5)
    te_idx, tr_idx = perm[:n_test], perm[n_test:]
    train = torch.utils.data.Subset(full_tr, tr_idx); test = torch.utils.data.Subset(full_te, te_idx)
    return (torch.utils.data.DataLoader(train, bs, shuffle=True, num_workers=workers, pin_memory=True),
            torch.utils.data.DataLoader(test, 128, shuffle=False, num_workers=workers, pin_memory=True))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); c = n = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        c += (model(x).argmax(1) == y).sum().item(); n += y.numel()
    return 100.0 * c / n


def weight_params(model):
    # conv / linear weights only: what gets quantized and protected
    return [(n, p) for n, p in model.named_parameters()
            if p.dim() >= 2 and p.requires_grad]


# ----------------------------------------------------------------- proxy
def msb_step(w):
    # dW for a sign-bit flip under per-tensor symmetric INT8
    return 128.0 * w.detach().abs().max() / 127.0


def tail_penalty(loss, params, tail_frac):
    """mean of the top-k damage proxies, differentiable through the gradient."""
    grads = torch.autograd.grad(loss, [p for _, p in params], create_graph=True)
    scores = torch.cat([(g.abs() * msb_step(p)).flatten()
                        for g, (_, p) in zip(grads, params)])
    k = max(1, int(tail_frac * scores.numel()))
    top = torch.topk(scores, k, sorted=False).values
    return top.mean()


def clip_penalty(params, q):
    """L2 push on |w| above the per-layer q-quantile (shrinks the INT8 range)."""
    pen = 0.0
    for _, p in params:
        a = p.abs()
        thr = torch.quantile(a.detach().flatten(), q)
        pen = pen + F.relu(a - thr).pow(2).sum() / a.numel()
    return pen


# ----------------------------------------------------------------- train
def cmd_train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    tl, vl = cifar_loaders(a.data, a.bs, a.workers)
    model = build_model(a.arch).to(dev)
    if a.init:
        load_state_flex(model, a.init)
        print(f"[init] clean acc = {evaluate(model, vl, dev):.2f}")
    params = weight_params(model)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl))

    for ep in range(a.epochs):
        model.train(); t0 = time.time(); ce_sum = pen_sum = 0.0
        for it, (x, y) in enumerate(tl):
            x, y = x.to(dev), y.to(dev)
            ce = F.cross_entropy(model(x), y)
            pen = torch.zeros((), device=dev)
            if a.reg in ("tail", "both") and it % a.tail_every == 0:
                pen = pen + tail_penalty(ce, params, a.tail_frac) * a.tail_every
            if a.reg in ("clip", "both"):
                pen = pen + a.clip_w * clip_penalty(params, a.clip_q)
            loss = ce + a.lam * pen
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            ce_sum += ce.item(); pen_sum += pen.item()
        acc = evaluate(model, vl, dev)
        print(f"ep {ep+1}/{a.epochs}  ce={ce_sum/len(tl):.3f}  pen={pen_sum/len(tl):.3e}  "
              f"acc={acc:.2f}  {time.time()-t0:.0f}s", flush=True)

    torch.save(model.state_dict(), a.out)
    print(f"[saved] {a.out}  final acc {acc:.2f}")


# ----------------------------------------------------------------- hist
def score_checkpoint(path, a, dev):
    """per-weight proxy scores on a fixed probe batch set (same batches for both ckpts)"""
    model = build_model(a.arch).to(dev)
    load_state_flex(model, path); model.eval()
    _, vl = cifar_loaders(a.data, 128, a.workers)
    params = weight_params(model)
    acc = torch.zeros(sum(p.numel() for _, p in params))
    seen = 0
    for x, y in vl:
        if seen >= a.probe_batches: break
        x, y = x.to(dev), y.to(dev)
        loss = F.cross_entropy(model(x), y)
        grads = torch.autograd.grad(loss, [p for _, p in params])
        acc += torch.cat([(g.abs() * msb_step(p)).flatten().cpu()
                          for g, (_, p) in zip(grads, params)])
        seen += 1
    acc /= seen
    clean = evaluate(model, vl, dev)
    score_checkpoint.layers = [(n, p.numel()) for n, p in params]   # side info for per-layer views
    return acc.numpy(), clean


def cmd_hist(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s1, acc1 = score_checkpoint(a.ckpt, a, dev)
    out = {"ckpt": a.ckpt, "clean_acc": acc1, "n_weights": int(s1.size)}
    # absolute thresholds from the BASELINE: top-1% and top-9% score boundaries
    hi = float(np.quantile(s1, 0.99)); mid = float(np.quantile(s1, 0.91))
    out["thr_high"] = hi; out["thr_mid"] = mid
    out["frac_above_high"] = float((s1 >= hi).mean())
    out["frac_above_mid"] = float((s1 >= mid).mean())
    out["tail_mass_top1pct"] = float(np.sort(s1)[-int(0.01*s1.size):].sum() / s1.sum())
    series = [("baseline", s1)]
    if a.ckpt2:
        s2, acc2 = score_checkpoint(a.ckpt2, a, dev)
        out["ckpt2"] = a.ckpt2; out["clean_acc2"] = acc2
        out["frac_above_high2"] = float((s2 >= hi).mean())
        out["frac_above_mid2"] = float((s2 >= mid).mean())
        out["tail_mass_top1pct2"] = float(np.sort(s2)[-int(0.01*s2.size):].sum() / s2.sum())
        out["ratio_high"] = out["frac_above_high2"] / max(out["frac_above_high"], 1e-12)
        series.append(("regularized", s2))

    # log-spaced bins shared by both models
    lo = max(min(s.min() for _, s in series), 1e-12)
    hi_edge = max(s.max() for _, s in series)
    edges = np.logspace(np.log10(lo), np.log10(hi_edge), a.bins + 1)
    hist = {}
    for name, s in series:
        h, _ = np.histogram(np.clip(s, lo, hi_edge), edges)
        hist[name] = (100.0 * h / s.size).tolist()
    out["bin_edges"] = edges.tolist(); out["hist_pct"] = hist

    # HIGH / MID / LOW bin occupancy against the baseline's fixed thresholds
    def bins3(s):
        h = float((s >= hi).mean()); m = float(((s >= mid) & (s < hi)).mean())
        return {"HIGH": 100*h, "MID": 100*m, "LOW": 100*(1-h-m)}
    out["bins_pct"] = {name: bins3(s) for name, s in series}
    print("\nBin occupancy (% of weights, fixed thresholds from baseline):")
    print(f"{'':14s}{'HIGH':>8s}{'MID':>8s}{'LOW':>8s}")
    for name, b in out["bins_pct"].items():
        print(f"{name:14s}{b['HIGH']:8.3f}{b['MID']:8.3f}{b['LOW']:8.3f}")

    # per-layer breakdown: HIGH / MID counts per layer, before vs after
    layers = score_checkpoint.layers; off = 0; per = []
    for (name, n) in layers:
        row = {"layer": name, "n": n}
        for tag, s in series:
            seg = s[off:off+n]
            row[f"high_{tag}"] = int((seg >= hi).sum()); row[f"mid_{tag}"] = int(((seg >= mid) & (seg < hi)).sum())
        per.append(row); off += n
    out["per_layer"] = per
    print("\nPer-layer HIGH / MID weight counts (fixed baseline thresholds):")
    hdr = f"{'layer':40s}{'n':>10s}" + "".join(f"{'HIGH_'+t:>12s}{'MID_'+t:>12s}" for t, _ in series)
    print(hdr)
    for r in per:
        if any(r[f"high_{t}"] > 0 or r[f"mid_{t}"] > 0 for t, _ in series):
            print(f"{r['layer'][:40]:40s}{r['n']:10d}" + "".join(f"{r['high_'+t]:12d}{r['mid_'+t]:12d}" for t, _ in series))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        # per-layer figure: HIGH count per layer, before vs after (log y)
        fig, ax = plt.subplots(figsize=(max(7, 0.22*len(per)), 3.6)); x = np.arange(len(per)); w = 0.8/len(series)
        for i, (tag, _) in enumerate(series):
            ax.bar(x + i*w, np.maximum([r[f"high_{tag}"] for r in per], 0.5), w, label=f"{tag} HIGH")
        ax.set_yscale("log"); ax.set_ylabel("HIGH-bin weights in layer (log)"); ax.set_xticks(x + w*(len(series)-1)/2)
        ax.set_xticklabels([r["layer"].replace(".weight", "") for r in per], rotation=90, fontsize=5)
        ax.legend(fontsize=7); ax.set_title("Where the HIGH-bin weights live, per layer"); plt.tight_layout()
        plt.savefig(os.path.join(a.out, "per_layer_high.png"), dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        names = list(out["bins_pct"]); w = 0.8 / len(names)
        for i, nm in enumerate(names):
            vals = [out["bins_pct"][nm][k] for k in ("HIGH", "MID", "LOW")]
            bars = ax.bar(np.arange(3) + i*w, vals, w, label=nm)
            for b_, v in zip(bars, vals):
                ax.text(b_.get_x()+b_.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(np.arange(3) + w*(len(names)-1)/2); ax.set_xticklabels(["HIGH", "MID", "LOW"])
        ax.set_ylabel("% of weights"); ax.set_yscale("log"); ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(a.out, "bins_hml.png"), dpi=200); plt.close(fig)
    except Exception as e:
        print("[bin plot skipped]", e)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3.4))
        c = np.sqrt(edges[:-1] * edges[1:])
        for name, s in series:
            plt.step(c, hist[name], where="mid", label=f"{name}")
        plt.axvline(mid, ls="--", c="gray", lw=0.8); plt.axvline(hi, ls="--", c="k", lw=0.8)
        plt.xscale("log"); plt.xlabel("MSB-flip damage proxy  |g|*dW"); plt.ylabel("% of weights")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(a.out, "sens_hist.png"), dpi=200)
    except Exception as e:
        print("[plot skipped]", e)

    print(json.dumps({k: v for k, v in out.items() if k not in ("bin_edges", "hist_pct")}, indent=2))


# ----------------------------------------------------------------- plot (multi-model quantized histogram)
def cmd_plot(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels = a.labels if a.labels else [os.path.basename(c).replace(".pth", "") for c in a.ckpts]
    assert len(labels) == len(a.ckpts), "--labels must match --ckpts"
    scores, accs = [], []
    for c in a.ckpts:
        s, acc = score_checkpoint(c, a, dev); scores.append(s); accs.append(acc)
        print(f"[scored] {c}  clean_acc={acc:.2f}")
    # relative sensitivity: 0..100 with 100 = the BASELINE's (first ckpt) top value.
    # baseline's 99.99th pct is used as 100 so a handful of extreme outliers does not squash the axis.
    smax = float(np.quantile(scores[0], 0.9999))
    rel = [np.clip(100.0 * s / smax, 0, 100) for s in scores]
    if a.scale == "log":
        edges = np.logspace(-2, 2, a.nbins + 1)          # 0.01 .. 100 relative
        rel = [np.clip(r, 1e-2, 100) for r in rel]
    else:
        edges = np.linspace(0, 100, a.nbins + 1)
    counts = [np.histogram(r, edges)[0] for r in rel]

    # table: bin range, then count per model
    hdr = f"{'bin':>4s} {'sens range':>16s} " + "".join(f"{l[:14]:>16s}" for l in labels)
    print("\n" + hdr)
    for i in range(a.nbins):
        row = f"{i+1:4d} {edges[i]:7.2f}-{edges[i+1]:<8.2f} " + "".join(f"{int(c[i]):16d}" for c in counts)
        print(row)
    top = [int(c[-a.top_bins:].sum()) for c in counts]
    print(f"\nweights in top {a.top_bins} bins: " + ", ".join(f"{l}={t}" for l, t in zip(labels, top)))

    os.makedirs(a.out, exist_ok=True)
    json.dump({"labels": labels, "clean_acc": accs, "edges": edges.tolist(),
               "counts": [c.tolist() for c in counts], "smax": smax},
              open(os.path.join(a.out, "quantized_hist.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.6))
        n = len(counts); w = 0.8 / n; x = np.arange(a.nbins)
        for i, (c, l, acc) in enumerate(zip(counts, labels, accs)):
            ax.bar(x + i*w, np.maximum(c, 0.5), w, label=f"{l} ({acc:.2f}%)")
        ax.set_yscale("log"); ax.set_ylabel("number of weights")
        ax.set_xlabel(f"quantized sensitivity bin ({'log' if a.scale=='log' else 'linear'}, 100 = baseline max)")
        ax.set_xticks(x + w*(n-1)/2); ax.set_xticklabels([str(i+1) for i in range(a.nbins)], fontsize=7)
        ax.legend(fontsize=7); plt.tight_layout()
        plt.savefig(os.path.join(a.out, "quantized_sens_hist.png"), dpi=200); plt.close(fig)
        print(f"[saved] {a.out}/quantized_sens_hist.png")
    except Exception as e:
        print("[plot skipped]", e)


# ----------------------------------------------------------------- ptq (quantization loss, baseline vs regularized)
def quantize_model_(model, bits):
    """in-place per-tensor symmetric PTQ of all conv/linear weights; returns (q_ints, scales) per layer"""
    qmax = 2 ** (bits - 1) - 1; store = []
    with torch.no_grad():
        for n, p in weight_params(model):
            scale = p.abs().max() / qmax
            q = torch.round(p / scale).clamp(-qmax - 1, qmax)
            p.copy_(q * scale); store.append((n, p, q, scale))
    return store


def cmd_ptq(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels = a.labels if a.labels else [os.path.basename(c).replace(".pth", "") for c in a.ckpts]
    _, vl = cifar_loaders(a.data, 128, a.workers)
    bits_list = [int(b) for b in a.bits.split(",")]
    res = {}
    print(f"\n{'model':16s}{'fp32':>8s}" + "".join(f"{'int'+str(b):>8s}" for b in bits_list))
    for c, l in zip(a.ckpts, labels):
        model = build_model(a.arch).to(dev); load_state_flex(model, c)
        row = {"fp32": evaluate(model, vl, dev)}
        for b in bits_list:
            m2 = build_model(a.arch).to(dev); load_state_flex(m2, c); quantize_model_(m2, b)
            row[f"int{b}"] = evaluate(m2, vl, dev)
        res[l] = row
        print(f"{l:16s}{row['fp32']:8.2f}" + "".join(f"{row['int'+str(b)]:8.2f}" for b in bits_list))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "ptq.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for l in labels:
            ax.plot(bits_list, [res[l]["fp32"] - res[l][f"int{b}"] for b in bits_list], marker="o", label=l)
        ax.set_xlabel("weight bits (per-tensor PTQ)"); ax.set_ylabel("accuracy loss vs fp32 (%)")
        ax.invert_xaxis(); ax.legend(fontsize=7); ax.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(a.out, "ptq_loss.png"), dpi=200); plt.close(fig)
        print(f"[saved] {a.out}/ptq_loss.png")
    except Exception as e:
        print("[plot skipped]", e)


# ----------------------------------------------------------------- flip (random MSB flips, no ECC)
def cmd_flip(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels = a.labels if a.labels else [os.path.basename(c).replace(".pth", "") for c in a.ckpts]
    _, vl = cifar_loaders(a.data, 128, a.workers)
    nflips = [int(x) for x in a.nflips.split(",")]
    res = {}
    print(f"\n{'model':16s}" + "".join(f"{'n='+str(n):>10s}" for n in nflips) + f"   ({a.mode} INT8 sign-bit flips)")
    for c, l in zip(a.ckpts, labels):
        accs = []
        if a.mode == "targeted":
            s_all, _ = score_checkpoint(c, a, dev)              # per-weight damage on THIS model
            order = torch.from_numpy(np.argsort(-s_all))       # most damaging first
        for n in nflips:
            per_seed = []
            for seed in range(a.seeds if a.mode == "random" else 1):
                model = build_model(a.arch).to(dev); load_state_flex(model, c)
                store = quantize_model_(model, 8)
                total = sum(q.numel() for _, _, q, _ in store)
                if a.mode == "targeted":
                    idx = order[:n].sort().values
                else:
                    g = torch.Generator().manual_seed(seed * 1000 + n)
                    idx = torch.randperm(total, generator=g)[:n].sort().values
                off = 0
                with torch.no_grad():
                    for _, p, q, scale in store:
                        sel = idx[(idx >= off) & (idx < off + q.numel())] - off
                        if sel.numel():
                            qf = q.flatten().clone()
                            qf[sel] = torch.where(qf[sel] >= 0, qf[sel] - 128, qf[sel] + 128)  # flip sign bit
                            p.copy_((qf * scale).view_as(p))
                        off += q.numel()
                per_seed.append(evaluate(model, vl, dev))
            accs.append(float(np.mean(per_seed)))
        res[l] = dict(zip([str(n) for n in nflips], accs))
        print(f"{l:16s}" + "".join(f"{v:10.2f}" for v in accs))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "flip.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for l in labels:
            ax.plot(nflips, [res[l][str(n)] for n in nflips], marker="o", label=l)
        ax.set_xscale("log"); ax.set_xlabel(f"{a.mode} sign-bit flips (INT8, no ECC)"); ax.set_ylabel("accuracy (%)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(a.out, f"flip_{a.mode}.png"), dpi=200); plt.close(fig)
        print(f"[saved] {a.out}/flip_{a.mode}.png")
    except Exception as e:
        print("[plot skipped]", e)


# ----------------------------------------------------------------- knob (lambda -> tail size / accuracy)
def cmd_knob(a):
    lams = [float(x) for x in a.lams.split(",")]
    rows = []
    for lam, d in zip(lams, a.dirs):
        s = json.load(open(os.path.join(d, "summary.json")))
        rows.append((lam, s["bins_pct"]["regularized"]["HIGH"], s["bins_pct"]["regularized"]["MID"],
                     100 * s["tail_mass_top1pct2"], s["clean_acc2"]))
        base = (s["bins_pct"]["baseline"]["HIGH"], s["bins_pct"]["baseline"]["MID"],
                100 * s["tail_mass_top1pct"], s["clean_acc"])
    print(f"\n{'lambda':>8s}{'HIGH%':>8s}{'MID%':>8s}{'tail%':>8s}{'acc':>8s}")
    print(f"{'0 (base)':>8s}{base[0]:8.3f}{base[1]:8.3f}{base[2]:8.2f}{base[3]:8.2f}")
    for r in rows:
        print(f"{r[0]:8g}{r[1]:8.3f}{r[2]:8.3f}{r[3]:8.2f}{r[4]:8.2f}")
    os.makedirs(a.out, exist_ok=True)
    json.dump({"baseline": base, "rows": rows}, open(os.path.join(a.out, "knob.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x = [0] + [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(x, [base[0]] + [r[1] for r in rows], marker="o", label="HIGH bin (% of weights)")
        ax.plot(x, [base[2]] + [r[3] for r in rows], marker="s", label="damage carried by top 1% (%)")
        ax.set_xscale("symlog", linthresh=1); ax.set_xlabel("regularizer strength lambda")
        ax.set_ylabel("%"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        ax2 = ax.twinx(); ax2.plot(x, [base[3]] + [r[4] for r in rows], marker="^", color="k", ls="--", label="clean acc")
        ax2.set_ylabel("clean accuracy (%)"); ax2.legend(fontsize=7, loc="lower left")
        plt.tight_layout(); plt.savefig(os.path.join(a.out, "control_knob.png"), dpi=200); plt.close(fig)
        print(f"[saved] {a.out}/control_knob.png")
    except Exception as e:
        print("[plot skipped]", e)


# ----------------------------------------------------------------- figure (bins + accuracy, all models)
def cmd_figure(a):
    """Bars LOW/MID/HIGH per model (log) + clean-accuracy strip, from hist_lam*/summary.json files."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    cfgs = [c for c in a.configs if os.path.exists(os.path.join(a.root, c, f"hist_lam{a.lams[0]}", "summary.json"))]
    if not cfgs:
        print("no results found under", a.root); return
    titles = [c.replace("_resnet", "\nResNet-").replace("cifar10", "CIFAR-10").replace("cifar100", "CIFAR-100").replace("imagenet", "ImageNet") for c in cfgs]
    keys = ["LOW", "MID", "HIGH"]; L = a.lams
    cols = ["#b8b6b0", "#3b82f6", "#6d28d9", "#dc2626", "#059669"][:len(L)+1]
    labs = ["baseline"] + [f"lambda = {l}" for l in L]
    n = len(cfgs); w = 0.8 / (len(L)+1)
    fig, axs = plt.subplots(2, n, figsize=(2.5*n, 5.6), squeeze=False,
                            gridspec_kw={"height_ratios": [3, 1.15], "hspace": 0.35, "wspace": 0.28})
    for k, (c, t) in enumerate(zip(cfgs, titles)):
        h = {l: json.load(open(os.path.join(a.root, c, f"hist_lam{l}", "summary.json"))) for l in L}
        bins = [h[L[0]]["bins_pct"]["baseline"]] + [h[l]["bins_pct"]["regularized"] for l in L]
        accs = [h[L[0]]["clean_acc"]] + [h[l]["clean_acc2"] for l in L]
        ax = axs[0, k]; x = np.arange(3)
        for j, b in enumerate(bins):
            v = [b[kk] for kk in keys]
            ax.bar(x + (j - len(L)/2) * w, v, w, color=cols[j], label=labs[j], edgecolor="white", linewidth=0.5)
            ax.text(2 + (j - len(L)/2) * w, b["HIGH"] * 1.25, f"{b['HIGH']:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_yscale("log"); ax.set_ylim(0.05, 200)
        ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100])); ax.yaxis.set_major_formatter(FixedFormatter(["0.1%", "1%", "10%", "100%"]))
        ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=9); ax.tick_params(axis="y", labelsize=8)
        ax.set_title(t, fontsize=10, fontweight="bold"); ax.grid(axis="y", alpha=.25)
        if k == 0: ax.set_ylabel("share of weights in bin (log)", fontsize=9); ax.legend(fontsize=8, frameon=False, loc="upper right")
        a2 = axs[1, k]; xs = list(range(len(accs)))
        a2.plot(xs, accs, color="#222", lw=1.4, zorder=1)
        for j in xs: a2.scatter(j, accs[j], color=cols[j], s=55, zorder=2, edgecolor="white", linewidth=0.8)
        lo, hi = min(accs), max(accs); mid = (lo + hi) / 2; span = max(1.2, (hi - lo) * 2.2)
        a2.set_ylim(mid - span/2, mid + span/2); a2.set_xlim(-0.5, len(accs) - 0.5)
        a2.set_xticks(xs); a2.set_xticklabels(["base"] + [f"l{l}" for l in L], fontsize=8); a2.tick_params(axis="y", labelsize=7)
        for j in xs: a2.annotate(f"{accs[j]:.2f}", (j, accs[j]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
        a2.grid(axis="y", alpha=.25)
        if k == 0: a2.set_ylabel("clean acc (%)", fontsize=9)
    fig.suptitle("Controllable skewing: the HIGH-sensitivity bin empties as lambda increases (top) while clean accuracy is unchanged (bottom)", fontsize=11, y=0.995)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plt.savefig(a.out, dpi=220, bbox_inches="tight"); print(f"[saved] {a.out}")


# ----------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for s in ("train", "hist", "plot", "ptq", "flip", "knob", "figure"):
        p = sub.add_parser(s)
        p.add_argument("--arch", default="resnet18")
        p.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
        p.add_argument("--imagenet-root", default="./imagenet-val")
        p.add_argument("--data", default="./data")
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
    t = sub.choices["train"]
    t.add_argument("--init", default="", help="clean checkpoint to fine-tune from")
    t.add_argument("--out", default="resnet18_c10_sensreg.pth")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--bs", type=int, default=128)
    t.add_argument("--lr", type=float, default=0.01)
    t.add_argument("--reg", choices=["tail", "clip", "both", "none"], default="both")
    t.add_argument("--lam", type=float, default=1e-3, help="overall regularizer weight")
    t.add_argument("--tail-frac", type=float, default=0.01, help="top-k fraction penalized")
    t.add_argument("--tail-every", type=int, default=4, help="apply tail penalty every N steps")
    t.add_argument("--clip-q", type=float, default=0.99, help="per-layer |w| quantile")
    t.add_argument("--clip-w", type=float, default=100.0, help="relative weight of clip term")
    h = sub.choices["hist"]
    h.add_argument("--ckpt", required=True)
    h.add_argument("--ckpt2", default="")
    h.add_argument("--probe-batches", type=int, default=20)
    h.add_argument("--bins", type=int, default=40)
    h.add_argument("--out", default="sens_hist")
    pl = sub.choices["plot"]
    pl.add_argument("--ckpts", nargs="+", required=True, help="first one is the baseline")
    pl.add_argument("--labels", nargs="*", default=None)
    pl.add_argument("--nbins", type=int, default=20)
    pl.add_argument("--scale", choices=["linear", "log"], default="linear")
    pl.add_argument("--top-bins", type=int, default=3)
    pl.add_argument("--probe-batches", type=int, default=20)
    pl.add_argument("--out", default="sens_plot")
    for name in ("ptq", "flip"):
        q = sub.choices[name]
        q.add_argument("--ckpts", nargs="+", required=True)
        q.add_argument("--labels", nargs="*", default=None)
        q.add_argument("--out", default=name + "_out")
    sub.choices["ptq"].add_argument("--bits", default="8,6,5,4")
    sub.choices["flip"].add_argument("--nflips", default="10,30,100,300,1000")
    sub.choices["flip"].add_argument("--seeds", type=int, default=3)
    sub.choices["flip"].add_argument("--mode", choices=["random", "targeted"], default="targeted")
    sub.choices["flip"].add_argument("--probe-batches", type=int, default=20)
    kn = sub.choices["knob"]
    kn.add_argument("--dirs", nargs="+", required=True, help="hist output dirs, one per lambda")
    kn.add_argument("--lams", required=True, help="comma list matching --dirs")
    kn.add_argument("--out", default="knob_out")
    fg = sub.choices["figure"]
    fg.add_argument("--root", default="sensreg_results")
    fg.add_argument("--configs", nargs="+", default=["cifar10_resnet18", "cifar10_resnet50", "cifar100_resnet18", "cifar100_resnet50", "imagenet_resnet18", "imagenet_resnet50"])
    fg.add_argument("--lams", nargs="+", type=int, default=[10, 30])
    fg.add_argument("--out", default="sensreg_results/fig_bins_accuracy.png")
    a = ap.parse_args()
    if hasattr(a, "dataset"):
        _DS = a.dataset; IMAGENET_ROOT = a.imagenet_root
    {"train": cmd_train, "hist": cmd_hist, "plot": cmd_plot,
     "ptq": cmd_ptq, "flip": cmd_flip, "knob": cmd_knob, "figure": cmd_figure}[a.cmd](a)
