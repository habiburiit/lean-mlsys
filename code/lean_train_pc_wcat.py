#!/usr/bin/env python
"""
lean_train_pc_wcat.py  -  two published training-time bit-flip defenses as baselines, trained under OUR conditions
(same data, epochs, lr, init as our lambda models). SEPARATE file; imports from lean_sensreg.py.

--method pc    : Piece-wise Clustering (He, Rakin et al., CVPR 2020)
                 L = CE + lam * sum_l ( ||W_l+ - mean(W_l+)||^2 + ||W_l- - mean(W_l-)||^2 )   (their lam = 1e-3 on CIFAR)
--method wcat  : Weight-Clipping-Aware Training (Chitsaz et al., TMLR 2023)
                 per-layer learnable clip range c_l; weights clipped to [-c_l, c_l] and fake-quantized to 8 bits with
                 step c_l/127 (STE); L = CE + lam * sum_l c_l   (pushes the quantization range down)   (lam ~ 1e-2)

Usage:
  python lean_train_pc_wcat.py --method pc   --dataset cifar10 --arch resnet18 --init <fp32.pth> --lam 1e-3 --epochs 10 --out <..._pc.pth>
  python lean_train_pc_wcat.py --method wcat --dataset cifar10 --arch resnet18 --init <fp32.pth> --lam 1e-2 --epochs 10 --out <..._wcat.pth>
"""
import argparse, time, torch, torch.nn as nn, torch.nn.functional as F
import lean_sensreg as S


class ClipQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, c):
        ctx.save_for_backward(w, c); step = c / 127.0
        return torch.round(torch.clamp(w, -c, c) / step) * step
    @staticmethod
    def backward(ctx, g):
        w, c = ctx.saved_tensors; inside = (w.abs() <= c).float()
        return g * inside, (g * torch.sign(w) * (1 - inside)).sum().view_as(c)


def pc_penalty(params):
    pen = 0.0
    for _, p in params:
        w = p.flatten(); pos = w[w > 0]; neg = w[w < 0]
        if pos.numel() > 1: pen = pen + ((pos - pos.mean()) ** 2).sum()
        if neg.numel() > 1: pen = pen + ((neg - neg.mean()) ** 2).sum()
    return pen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["pc", "wcat"], required=True)
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--lam", type=float, default=None, help="penalty weight (default: pc 1e-3, wcat 1e-2)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    lam = a.lam if a.lam is not None else (1e-3 if a.method == "pc" else 1e-2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.init)
    print(f"[init] clean acc = {S.evaluate(model, vl, dev):.2f}   baseline method={a.method}, lam={lam}")
    params = S.weight_params(model); mods = {n.replace(".weight", ""): m for n, m in model.named_modules()}
    clips = {}
    if a.method == "wcat":
        for n, p in params:
            clips[n] = nn.Parameter(p.detach().abs().max().clamp(min=1e-6).reshape(1).to(dev))
        def make_fwd(mod, name):
            def fwd(x):
                w = ClipQuant.apply(mod.weight, clips[name])
                if isinstance(mod, nn.Conv2d): return F.conv2d(x, w, mod.bias, mod.stride, mod.padding, mod.dilation, mod.groups)
                return F.linear(x, w, mod.bias)
            mod.forward = fwd
        for n, _ in params: make_fwd(mods[n.replace(".weight", "")], n)
        opt = torch.optim.SGD([{"params": model.parameters()}, {"params": list(clips.values()), "weight_decay": 0.0}], lr=a.lr, momentum=0.9, weight_decay=5e-4)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl))
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); ce_s = pen_s = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            ce = F.cross_entropy(model(x), y)
            pen = pc_penalty(params) if a.method == "pc" else sum(c.sum() for c in clips.values())
            loss = ce + lam * pen
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); ce_s += ce.item(); pen_s += float(pen)
        acc = S.evaluate(model, vl, dev); nb = len(tl)
        extra = f"  mean clip={sum(float(c) for c in clips.values())/len(clips):.3f}" if clips else ""
        print(f"ep {ep+1}/{a.epochs}  ce={ce_s/nb:.3f}  pen={pen_s/nb:.3e}  acc={acc:.2f}{extra}  {time.time()-t0:.0f}s", flush=True)
    if a.method == "wcat":                       # bake the learned clipping into the stored weights so the INT8 pipeline sees it
        with torch.no_grad():
            for n, p in params: p.copy_(torch.clamp(p, -clips[n].item(), clips[n].item()))
    torch.save(model.state_dict(), a.out); print(f"[saved] {a.out}  final acc {acc:.2f}")


if __name__ == "__main__":
    main()
