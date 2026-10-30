#!/usr/bin/env python
"""lean_hist.py - the HIGH/MID/LOW comparison of `lean_sensreg.py hist`, done through the imported module so that
lean_vgg's VGG registration is honoured. Prints the same 'Bin occupancy' block and writes summary.json.
usage: python lean_hist.py --dataset cifar10 --arch vgg16_bn --ckpt A.pth --ckpt2 B.pth --out DIR
"""
import argparse, json, os, numpy as np, torch
import lean_sensreg as S
try:
    import lean_vgg
except Exception as e:
    print(f"[warn] lean_vgg not importable: {e}")

_grad = torch.autograd.grad
def _grad_allow_unused(outputs, inputs, *args, **kwargs):
    kwargs.setdefault("allow_unused", True); gs = _grad(outputs, inputs, *args, **kwargs)
    ins = inputs if isinstance(inputs, (list, tuple)) else [inputs]
    return tuple(g if g is not None else torch.zeros_like(p) for g, p in zip(gs, ins))
torch.autograd.grad = _grad_allow_unused

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--arch", default="resnet18")
ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--imagenet-root", default="./imagenet-val"); ap.add_argument("--probe-batches", type=int, default=20)
ap.add_argument("--ckpt", required=True); ap.add_argument("--ckpt2", required=True); ap.add_argument("--out", default="hist_out")
a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = S.build_model(a.arch); n = sum(p.numel() for _, p in S.weight_params(m)); print(f"[arch] {a.arch}: {n/1e6:.1f}M weights")
s1, acc1 = S.score_checkpoint(a.ckpt, a, dev); s2, acc2 = S.score_checkpoint(a.ckpt2, a, dev)
hi = float(np.quantile(s1, 0.99)); mid = float(np.quantile(s1, 0.91))
def bins3(s):
    h = float((s >= hi).mean()); mm = float(((s >= mid) & (s < hi)).mean()); return {"HIGH": 100*h, "MID": 100*mm, "LOW": 100*(1-h-mm)}
out = {"ckpt": a.ckpt, "ckpt2": a.ckpt2, "clean_acc": acc1, "clean_acc2": acc2, "n_weights": int(s1.size), "thr_high": hi, "thr_mid": mid,
       "tail_mass_top1pct": float(np.sort(s1)[-int(0.01*s1.size):].sum()/s1.sum()), "tail_mass_top1pct2": float(np.sort(s2)[-int(0.01*s2.size):].sum()/s2.sum()),
       "bins_pct": {"baseline": bins3(s1), "regularized": bins3(s2)}}
out["ratio_high"] = out["bins_pct"]["regularized"]["HIGH"] / out["bins_pct"]["baseline"]["HIGH"]
print("\nBin occupancy (% of weights, fixed thresholds from baseline):")
print(f"{'':14s}{'HIGH':>8s}{'MID':>8s}{'LOW':>8s}")
for name, b in out["bins_pct"].items(): print(f"{name:14s}{b['HIGH']:8.3f}{b['MID']:8.3f}{b['LOW']:8.3f}")
print(f"clean_acc {acc1:.2f} -> clean_acc2 {acc2:.2f}   tail_mass_top1pct {out['tail_mass_top1pct']:.4f} -> {out['tail_mass_top1pct2']:.4f}   ratio_high {out['ratio_high']:.3f}")
os.makedirs(a.out, exist_ok=True); json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
