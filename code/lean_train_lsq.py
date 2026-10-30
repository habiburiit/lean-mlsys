#!/usr/bin/env python
"""
lean_train_lsq.py  -  apples-to-apples baselines under OUR conditions (same data split, same 10 epochs, weights only):
  LSQ  (Esser et al., ICLR 2020): uniform b-bit weights with a LEARNED per-tensor step size (gradient scaled by 1/sqrt(N*Qp))
  KURE (Shkolnik et al., NeurIPS 2020): kurtosis regularizer pushing each layer's weight distribution toward uniform
        (target kurtosis 1.8), trained with the same fake-quant forward.  --kure 0 disables it.
SEPARATE file; imports from lean_sensreg.py and lean_train_qat.py (nothing there is changed).

Usage (ResNet-50 ImageNet):
  python lean_train_lsq.py --dataset imagenet --arch resnet50 --imagenet-root <path> --init pretrained --bits 3 --lr 0.0005 --bs 32 --epochs 10 --out <lsq3.pth>
  python lean_train_lsq.py ... --bits 3 --kure 1.0 --out <kure3.pth>        (KURE regularizer + 3-bit QAT)
  python lean_train_lsq.py ... --bits 4 --out <lsq4.pth>
"""
import argparse, math, time, torch, torch.nn as nn, torch.nn.functional as F
import lean_sensreg as S


class LSQ(torch.autograd.Function):
    """learned step size quantizer (per tensor); gradient for the step as in the LSQ paper"""
    @staticmethod
    def forward(ctx, w, step, qn, qp, g):
        ctx.save_for_backward(w, step); ctx.other = (qn, qp, g)
        q = torch.clamp(w / step, -qn, qp); return torch.round(q) * step
    @staticmethod
    def backward(ctx, grad):
        w, step = ctx.saved_tensors; qn, qp, g = ctx.other
        q = w / step; lo = q < -qn; hi = q > qp; mid = ~(lo | hi)
        grad_w = grad * mid
        gs = torch.where(lo, torch.full_like(q, -float(qn)), torch.where(hi, torch.full_like(q, float(qp)), torch.round(q) - q))
        grad_step = (grad * gs).sum() * g
        return grad_w, grad_step.view_as(step), None, None, None


def kurtosis_penalty(params, target=1.8):
    pen = 0.0
    for _, p in params:
        w = p.flatten(); m = w.mean(); s = w.std() + 1e-12
        k = (((w - m) / s) ** 4).mean(); pen = pen + (k - target) ** 2
    return pen / len(params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--bits", type=int, default=3, help="uniform weight bit width")
    ap.add_argument("--kure", type=float, default=0.0, help="KURE kurtosis-regularizer weight (0 = plain LSQ)")
    ap.add_argument("--kure-target", type=float, default=1.8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.init)
    print(f"[init] clean acc (fp32) = {S.evaluate(model, vl, dev):.2f}   uniform {a.bits}-bit LSQ, kure={a.kure}")
    params = S.weight_params(model); mods = {n.replace(".weight", ""): m for n, m in model.named_modules()}
    qn = 2 ** (a.bits - 1); qp = 2 ** (a.bits - 1) - 1
    steps = {}
    for n, p in params:                                            # LSQ init: 2*mean|w|/sqrt(Qp)
        steps[n] = nn.Parameter((2 * p.detach().abs().mean() / math.sqrt(qp)).clamp(min=1e-8).reshape(1).to(dev))
    def make_fwd(mod, name):
        g = 1.0 / math.sqrt(mod.weight.numel() * qp)
        def fwd(x):
            w = LSQ.apply(mod.weight, steps[name], qn, qp, g)
            if isinstance(mod, nn.Conv2d): return F.conv2d(x, w, mod.bias, mod.stride, mod.padding, mod.dilation, mod.groups)
            return F.linear(x, w, mod.bias)
        mod.forward = fwd
    for n, _ in params: make_fwd(mods[n.replace(".weight", "")], n)
    opt = torch.optim.SGD([{"params": model.parameters()}, {"params": list(steps.values()), "weight_decay": 0.0}], lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl))
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); ce_s = pen_s = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            ce = F.cross_entropy(model(x), y); pen = kurtosis_penalty(params, a.kure_target) if a.kure > 0 else torch.zeros((), device=dev)
            loss = ce + a.kure * pen
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); ce_s += ce.item(); pen_s += float(pen)
        acc = S.evaluate(model, vl, dev); nb = len(tl)
        print(f"ep {ep+1}/{a.epochs}  ce={ce_s/nb:.3f}  kurt_pen={pen_s/nb:.3e}  acc@{a.bits}bit={acc:.2f}  {time.time()-t0:.0f}s", flush=True)
    torch.save({"state_dict": model.state_dict(), "steps": {k: v.detach().cpu() for k, v in steps.items()}, "bits": a.bits}, a.out)
    print(f"[saved] {a.out}  uniform {a.bits}-bit weights ({a.bits:.2f} bits/weight), kure={a.kure}, accuracy {acc:.2f}")


if __name__ == "__main__":
    main()
