#!/usr/bin/env python
"""
lean_archs.py  -  LEAN on more ImageNet backbones (MobileNet-V2, EfficientNet-B0, ViT-B/16, Swin-T, RegNet-Y-16GF,
Wide-ResNet-50-2, any torchvision model name) WITHOUT modifying lean_sensreg.py: like lean_vgg.py, it swaps in an
arch-aware build_model at import time, then either
  base   : saves the torchvision pretrained weights as artifacts/models/imagenet/<arch>/model_float32.pth
  train  : runs lean_sensreg.cmd_train (the LEAN fine-tune, same options as `lean_sensreg.py train`)
  share  : HIGH / MID shares of a checkpoint against thresholds fixed from a baseline checkpoint (held-out batches),
           like Table I of the paper (damage |g_i| x INT8 sign-flip step)
  run    : launches ANY other script (lean_tbfa.py, lean_pbs_codec.py, lean_hist.py, ...) with the patch active:
           python lean_archs.py run lean_tbfa.py --dataset imagenet --arch vit_b_16 ...
SEPARATE file.  `import lean_archs` from another script also activates the patch.

usage:
  python lean_archs.py base  --arch mobilenet_v2
  python lean_archs.py train --arch mobilenet_v2 --dataset imagenet --init artifacts/models/imagenet/mobilenet_v2/model_float32.pth \
         --reg both --lam 40 --lr 0.0005 --bs 32 --epochs 10 --out artifacts/models/imagenet/mobilenet_v2/model_float32_sensreg_lam40.pth
  python lean_archs.py share --arch mobilenet_v2 --dataset imagenet --base <model_float32.pth> --ckpt <lam40.pth>
"""
import argparse, os, runpy, sys
import torch, torch.nn.functional as F, torchvision
import lean_sensreg as S
try:
    import lean_vgg                      # keeps vgg16_bn working too
except Exception:
    pass

_orig_build = S.build_model
# the LEAN term needs double backward; the fused attention kernels (flash / memory-efficient SDPA) used by ViT and Swin have no
# double-backward, so force the math kernel for transformers (same reason the LLM path uses attn_implementation="eager")
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
TV = {"mobilenet_v2", "mobilenet_v3_large", "efficientnet_b0", "efficientnet_b1", "vit_b_16", "vit_b_32", "swin_t", "swin_s", "swin_v2_t",
      "regnet_y_16gf", "regnet_y_8gf", "wide_resnet50_2", "resnext50_32x4d", "convnext_tiny", "densenet121", "maxvit_t"}


def build_model_ext(arch="resnet18", num_classes=None):
    if arch in TV:
        return torchvision.models.get_model(arch, weights=None, num_classes=1000 if num_classes is None else num_classes)
    try:
        return _orig_build(arch, num_classes)
    except TypeError:
        return _orig_build(arch)


S.build_model = build_model_ext


def cmd_base(a):
    m = torchvision.models.get_model(a.arch, weights="DEFAULT"); d = f"artifacts/models/{a.dataset}/{a.arch}"; os.makedirs(d, exist_ok=True)
    torch.save(m.state_dict(), f"{d}/model_float32.pth"); print(f"[saved] {d}/model_float32.pth  ({sum(p.numel() for p in m.parameters())/1e6:.1f}M params)")


def cmd_share(a):
    S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root; dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, vl = S.cifar_loaders(a.data, 128, a.workers)
    def damage(ck):
        m = S.build_model(a.arch).to(dev); S.load_state_flex(m, ck); params = S.weight_params(m); plist = [p for _, p in params]; m.eval()
        acc = S.evaluate(m, vl, dev); dsum = None; n = 0
        for x, y in vl:
            if n >= a.batches: break
            x, y = x.to(dev), y.to(dev); g = torch.autograd.grad(F.cross_entropy(m(x), y), plist)
            d = torch.cat([(gi.abs() * S.msb_step(p)).flatten().cpu() for gi, p in zip(g, plist)]); dsum = d if dsum is None else dsum + d; n += 1
        del m; torch.cuda.empty_cache(); return dsum / n, acc
    db, accb = damage(a.base); srt = torch.sort(db).values; th_h = srt[int(0.99 * (srt.numel() - 1))]; th_m = srt[int(0.91 * (srt.numel() - 1))]; del srt   # quantile() caps at 16M elements
    print(f"{'ckpt':40s}{'acc':>7s}{'HIGH %':>9s}{'MID %':>9s}{'top1% share':>13s}")
    for ck, d, acc in [(a.base, db, accb)] + [(c,) + damage(c) for c in a.ckpt]:
        k = max(1, int(0.01 * d.numel())); share = torch.topk(d, k).values.sum() / d.sum()
        print(f"{os.path.basename(ck):40s}{acc:7.2f}{100*(d>=th_h).float().mean():9.3f}{100*((d>=th_m)&(d<th_h)).float().mean():9.2f}{100*share:13.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("base"); b.add_argument("--arch", required=True); b.add_argument("--dataset", default="imagenet")
    t = sub.add_parser("train")
    for k, v in [("--arch", "resnet18"), ("--dataset", "imagenet"), ("--imagenet-root", "./imagenet-val"), ("--data", "./data"), ("--init", None), ("--out", None), ("--reg", "both")]:
        t.add_argument(k, default=v)
    for k, v in [("--workers", 4), ("--seed", 0), ("--epochs", 10), ("--bs", 32), ("--tail-every", 4)]: t.add_argument(k, type=int, default=v)
    for k, v in [("--lr", 0.0005), ("--lam", 0.0), ("--tail-frac", 0.01), ("--clip-q", 0.99), ("--clip-w", 100.0)]: t.add_argument(k, type=float, default=v)
    s = sub.add_parser("share"); s.add_argument("--arch", required=True); s.add_argument("--dataset", default="imagenet"); s.add_argument("--imagenet-root", default="./imagenet-val")
    s.add_argument("--data", default="./data"); s.add_argument("--workers", type=int, default=4); s.add_argument("--batches", type=int, default=20)
    s.add_argument("--base", required=True); s.add_argument("--ckpt", nargs="+", required=True)
    r = sub.add_parser("run"); r.add_argument("script"); r.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.cmd == "base": cmd_base(a)
    elif a.cmd == "train":
        S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root      # the globals lean_sensreg's own main sets
        S.cmd_train(a)
    elif a.cmd == "share": cmd_share(a)
    else:
        sys.argv = [a.script] + a.rest; runpy.run_path(a.script, run_name="__main__")


if __name__ == "__main__":
    main()
