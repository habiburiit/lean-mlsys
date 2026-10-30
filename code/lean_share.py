#!/usr/bin/env python
"""
lean_share.py  -  scale-free concentration of the weight-sensitivity distribution: the share of total first-order
damage carried by the top 1% of weights, baseline vs LEAN, for the nine vision models (new column of Table I).
A uniform rescaling of every weight's flip step leaves this share unchanged, so a smaller share means the distribution
was reshaped, not only scaled.  Also prints the share of the top 1% by |g_i| alone (no flip step at all).
Damage d_i = |g_i| x INT8 sign-flip step, averaged over --batches held-out batches, as in plot_sens_curves.py.
SEPARATE file; imports lean_sensreg (and lean_vgg for VGG) unchanged.  No training.

usage:  CUDA_VISIBLE_DEVICES=1 python -u lean_share.py --imagenet-root <path> 2>&1 | tee sensreg_results/share_table.log
"""
import argparse, json, numpy as np, torch, torch.nn.functional as F
import lean_sensreg as S
try:
    import lean_vgg
except Exception:
    pass

CFG = [  # dataset, arch, lambda, label
    ("cifar10", "resnet18", 40, "C10 ResNet-18"), ("cifar10", "resnet50", 40, "C10 ResNet-50"), ("cifar10", "vgg16_bn", 40, "C10 VGG-16"),
    ("cifar100", "resnet18", 40, "C100 ResNet-18"), ("cifar100", "resnet50", 40, "C100 ResNet-50"), ("cifar100", "vgg16_bn", 40, "C100 VGG-16"),
    ("imagenet", "resnet18", 30, "ImNet ResNet-18"), ("imagenet", "resnet50", 40, "ImNet ResNet-50"), ("imagenet", "vgg16_bn", 100, "ImNet VGG-16")]


def shares(dataset, arch, ckpt, batches, data, workers, imroot, dev):
    S._DS = dataset; S.IMAGENET_ROOT = imroot
    _, vl = S.cifar_loaders(data, 128, workers)
    model = S.build_model(arch).to(dev); S.load_state_flex(model, ckpt); params = S.weight_params(model)
    plist = [p for _, p in params]; dsum = gsum = None; n = 0; model.eval()
    for x, y in vl:
        if n >= batches: break
        x, y = x.to(dev), y.to(dev); loss = F.cross_entropy(model(x), y)
        g = torch.autograd.grad(loss, plist)
        d = torch.cat([(gi.abs() * S.msb_step(p)).flatten().cpu() for gi, p in zip(g, plist)])
        ga = torch.cat([gi.abs().flatten().cpu() for gi in g])
        dsum = d if dsum is None else dsum + d; gsum = ga if gsum is None else gsum + ga; n += 1
        model.zero_grad(set_to_none=True)
    d = (dsum / n).numpy(); ga = (gsum / n).numpy(); k = max(1, int(0.01 * d.size))
    sd = np.sort(d)[-k:].sum() / d.sum(); sg = np.sort(ga)[-k:].sum() / ga.sum()
    del model; torch.cuda.empty_cache()
    return 100 * sd, 100 * sg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imagenet-root", default="./imagenet-val"); ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--out", default="sensreg_results/share_table.json")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
    rows = []
    print(f"{'model':18s}{'top1% damage share: base':>26s}{'LEAN':>8s}{'top1% |g| share: base':>24s}{'LEAN':>8s}", flush=True)
    for ds, arch, lam, lab in CFG:
        D = f"artifacts/models/{ds}/{arch}"
        try:
            b = shares(ds, arch, f"{D}/model_float32.pth", a.batches, a.data, a.workers, a.imagenet_root, dev)
            l = shares(ds, arch, f"{D}/model_float32_sensreg_lam{lam}.pth", a.batches, a.data, a.workers, a.imagenet_root, dev)
        except Exception as e:
            print(f"{lab:18s}  skipped: {e}", flush=True); continue
        rows.append(dict(model=lab, lam=lam, share_base=b[0], share_lean=l[0], gshare_base=b[1], gshare_lean=l[1]))
        print(f"{lab:18s}{b[0]:26.1f}{l[0]:8.1f}{b[1]:24.1f}{l[1]:8.1f}", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1); print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
