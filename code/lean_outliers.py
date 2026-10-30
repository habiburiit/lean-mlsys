#!/usr/bin/env python
"""
lean_outliers.py  -  per-layer outlier ratio of the baseline (max|W| / q99|W|) next to the per-layer HIGH-count
reduction after LEAN.  Scores both checkpoints itself (same probe batches as hist).  SEPARATE file; imports lean_sensreg unchanged.

usage:
  CUDA_VISIBLE_DEVICES=2 python lean_outliers.py --dataset cifar10 --arch resnet18 \
     --ckpt artifacts/models/cifar10/resnet18/model_float32.pth --ckpt2 artifacts/models/cifar10/resnet18/model_float32_sensreg_lam40.pth \
     --out sensreg_results/cifar10_resnet18/outliers_lam40.json
"""
import argparse, json, numpy as np, torch
import lean_sensreg as S
try:
    import lean_vgg
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--arch", default="resnet18")
ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=0)
ap.add_argument("--imagenet-root", default="./imagenet-val"); ap.add_argument("--probe-batches", type=int, default=20)
ap.add_argument("--ckpt", required=True); ap.add_argument("--ckpt2", required=True)
ap.add_argument("--out", default="outliers.json")
a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
dev = "cuda" if torch.cuda.is_available() else "cpu"

layers = [(n, p.numel()) for n, p in S.weight_params(S.build_model(a.arch))]
s1, acc1 = S.score_checkpoint(a.ckpt, a, dev)
s2, acc2 = S.score_checkpoint(a.ckpt2, a, dev)
hi = float(np.quantile(s1, 0.99))
print(f"baseline acc {acc1:.2f}  LEAN acc {acc2:.2f}  HIGH threshold {hi:.3e}  total HIGH {int((s1>=hi).sum())} -> {int((s2>=hi).sum())}")

model = S.build_model(a.arch); S.load_state_flex(model, a.ckpt); model.eval()
ratio = {}
with torch.no_grad():
    for n, p in S.weight_params(model):
        w = p.abs().flatten().float()
        q99 = torch.quantile(w[torch.randperm(w.numel())[:min(w.numel(), 2_000_000)]], 0.99).item()
        ratio[n] = w.max().item() / max(q99, 1e-12)

rows = []; off = 0
print(f"\n{'layer':40s}{'n':>10s}{'max/q99':>9s}{'HIGH_base':>11s}{'HIGH_lean':>11s}{'reduction':>10s}")
for n, cnt in layers:
    hb = int((s1[off:off+cnt] >= hi).sum()); hl = int((s2[off:off+cnt] >= hi).sum()); off += cnt
    red = hb / max(hl, 1)
    rows.append({"layer": n, "n": cnt, "outlier_ratio": ratio[n], "high_base": hb, "high_lean": hl, "reduction": red})
    print(f"{n[:40]:40s}{cnt:10d}{ratio[n]:9.2f}{hb:11d}{hl:11d}{red:10.1f}")
json.dump(rows, open(a.out, "w"), indent=1)
print(f"[saved] {a.out}")
