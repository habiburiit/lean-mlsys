#!/usr/bin/env python
"""
lean_vgg.py  -  VGG-16 support for the sensitivity-skewing pipeline WITHOUT touching lean_sensreg.py.
Imports lean_sensreg (and plot_sens_curves) and swaps in a VGG-aware build_model at run time.

  CIFAR-10 / CIFAR-100 : torchvision vgg16_bn with a CIFAR head (global average pool + one linear layer, 14.7 M weights).
                         No pretrained CIFAR VGG exists, so `base` trains it from scratch (one run, ~30 min on one GPU).
  ImageNet             : torchvision vgg16_bn pretrained (baseline = "pretrained", like the ResNets).

Subcommands (all flags of the underlying lean_sensreg commands are kept):
  base    train the CIFAR baseline from scratch        python lean_vgg.py base --dataset cifar10 --epochs 60 --out <pth>
  lam     lambda fine-tune (= lean_sensreg.py train)  python lean_vgg.py lam  --dataset cifar10 --init <base> --lam 40 --out <pth>
  hist    HIGH/MID/LOW bins (= lean_sensreg.py hist)  python lean_vgg.py hist --dataset cifar10 --ckpt <base> --ckpt2 <lam> --out <dir>
  curves  sensitivity curves (= plot_sens_curves.py)   python lean_vgg.py curves --out <dir> --pairs <dataset:vgg16_bn:ckpt:label> ...
"""
import argparse, sys, types
import torch, torch.nn as nn, torchvision
import lean_sensreg as S

_orig_build = S.build_model


def build_model_vgg(arch="resnet18", num_classes=None):
    if not arch.startswith("vgg"):
        return _orig_build(arch, num_classes)
    num_classes = num_classes or S.NUM_CLASSES[S._DS]
    if S._DS == "imagenet":
        return getattr(torchvision.models, arch)(weights="DEFAULT")
    m = getattr(torchvision.models, arch)(num_classes=num_classes)
    m.avgpool = nn.AdaptiveAvgPool2d(1)                      # 32x32 input -> 1x1 after 5 pools
    m.classifier = nn.Linear(512, num_classes)               # CIFAR head
    return m


S.build_model = build_model_vgg      # every S.* command now understands vgg16_bn


def clip_penalty_big(params, q):
    """same as S.clip_penalty, but the per-layer quantile uses kthvalue: torch.quantile refuses tensors
    above 16 M elements and VGG-16's first fc layer has 103 M weights."""
    import torch.nn.functional as F
    pen = 0.0
    for _, p in params:
        a = p.abs(); flat = a.detach().flatten()
        k = max(1, min(flat.numel(), int(q * flat.numel())))
        thr = flat.kthvalue(k).values
        pen = pen + F.relu(a - thr).pow(2).sum() / a.numel()
    return pen


S.clip_penalty = clip_penalty_big


def cmd_base(a):
    import time, torch.nn.functional as F
    S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    model = S.build_model(a.arch).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl))
    best = 0.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(model(x), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        acc = S.evaluate(model, vl, dev)
        print(f"ep {ep+1}/{a.epochs}  loss={tot/len(tl):.3f}  acc={acc:.2f}  {time.time()-t0:.0f}s", flush=True)
        if acc >= best: best = acc; torch.save(model.state_dict(), a.out)
    print(f"[saved] {a.out}  best clean acc {best:.2f}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for s in ("base", "lam", "hist", "curves"):
        p = sub.add_parser(s)
        p.add_argument("--arch", default="vgg16_bn")
        p.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
        p.add_argument("--imagenet-root", default="./imagenet-val"); p.add_argument("--data", default="./data")
        p.add_argument("--workers", type=int, default=4); p.add_argument("--seed", type=int, default=0)
    b = sub.choices["base"]
    b.add_argument("--out", required=True); b.add_argument("--epochs", type=int, default=60)
    b.add_argument("--bs", type=int, default=128); b.add_argument("--lr", type=float, default=0.05)
    t = sub.choices["lam"]
    t.add_argument("--init", default=""); t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=10); t.add_argument("--bs", type=int, default=128)
    t.add_argument("--lr", type=float, default=0.002); t.add_argument("--reg", choices=["tail", "clip", "both", "none"], default="both")
    t.add_argument("--lam", type=float, default=40.0); t.add_argument("--tail-frac", type=float, default=0.01)
    t.add_argument("--tail-every", type=int, default=4); t.add_argument("--clip-q", type=float, default=0.99)
    t.add_argument("--clip-w", type=float, default=100.0)
    h = sub.choices["hist"]
    h.add_argument("--ckpt", required=True); h.add_argument("--ckpt2", default="")
    h.add_argument("--probe-batches", type=int, default=20); h.add_argument("--bins", type=int, default=40); h.add_argument("--out", default="sens_hist")
    c = sub.choices["curves"]
    c.add_argument("--pairs", nargs="+", required=True); c.add_argument("--nbins", type=int, default=40)
    c.add_argument("--probe-batches", type=int, default=20); c.add_argument("--norm", action="store_true"); c.add_argument("--out", default="sens_curves")
    a = ap.parse_args()
    S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    if a.cmd == "base": cmd_base(a)
    elif a.cmd == "lam": S.cmd_train(a)
    elif a.cmd == "hist": S.cmd_hist(a)
    else:
        import plot_sens_curves as P          # P.S is the same (patched) module object
        sys.argv = ["plot_sens_curves.py", "--out", a.out, "--nbins", str(a.nbins), "--probe-batches", str(a.probe_batches),
                    "--data", a.data, "--workers", str(a.workers), "--imagenet-root", a.imagenet_root] + (["--norm"] if a.norm else []) + ["--pairs"] + a.pairs
        P.main()


if __name__ == "__main__":
    main()
