#!/usr/bin/env python
"""
lean_train_qat.py  -  quantization-aware fine-tuning at the target bit width + sensitivity tail term.
SEPARATE file; imports from lean_sensreg.py (nothing there is changed).

Forward pass uses fake-quantized weights (straight-through estimator):
    MID/LOW weights : per-output-channel symmetric b-bit   (b = --tq-bits)
    HIGH weights    : per-tensor 8-bit  (the protected set; top --tail-frac by damage score, refreshed every --refresh steps)
Loss:
    L = CE(quantized model) + lambda * ( P_tail + 100 * P_clip )
lambda = 0 gives plain QAT (the fair baseline); lambda > 0 adds the sensitivity shaping on top.

Usage:
  python lean_train_qat.py --dataset cifar10 --arch resnet18 --init artifacts/models/cifar10/resnet18/model_float32.pth \
      --tq-bits 3 --lam 0  --epochs 10 --out artifacts/models/cifar10/resnet18/model_qat3_lam0.pth
  python lean_train_qat.py --dataset cifar10 --arch resnet18 --init artifacts/models/cifar10/resnet18/model_float32.pth \
      --tq-bits 3 --lam 40 --epochs 10 --out artifacts/models/cifar10/resnet18/model_qat3_lam40.pth
Evaluate afterwards with lean_trunc_sens.py --mode requant (same per-channel scheme) or the built-in --eval-bits report.
"""
import argparse, time, torch, torch.nn as nn, torch.nn.functional as F
import lean_sensreg as S


class FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, scale, qmax):
        return torch.round(w / scale).clamp(-qmax - 1, qmax) * scale
    @staticmethod
    def backward(ctx, g):
        return g, None, None                     # straight-through


def fq_weight(w, high_mask, bits):
    """MID/LOW: per-channel b-bit; HIGH: per-tensor 8-bit. high_mask: bool, same shape as w."""
    red = tuple(range(1, w.dim()))
    low = ~high_mask
    qmax = 2 ** (bits - 1) - 1
    sc = ((w.detach().abs() * low).amax(dim=red, keepdim=True) / qmax).clamp(min=1e-12)
    w_low = FakeQuant.apply(w, sc, qmax)
    sc8 = (w.detach().abs().max() / 127).clamp(min=1e-12)
    w_high = FakeQuant.apply(w, sc8, 127)
    return torch.where(high_mask, w_high, w_low)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--tq-bits", type=int, default=3, help="bit width for MID/LOW weights during training")
    ap.add_argument("--lam", type=float, default=0.0, help="sensitivity tail term weight (0 = plain QAT)")
    ap.add_argument("--tail-frac", type=float, default=0.01); ap.add_argument("--tail-every", type=int, default=2)
    ap.add_argument("--refresh", type=int, default=50, help="recompute the HIGH mask every N steps")
    ap.add_argument("--clip-q", type=float, default=0.99); ap.add_argument("--clip-w", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.init)
    print(f"[init] clean acc (fp32) = {S.evaluate(model, vl, dev):.2f}")
    params = S.weight_params(model); plist = [p for _, p in params]
    mods = {n.replace(".weight", ""): m for n, m in model.named_modules()}
    # swap each conv/linear forward to use fake-quantized weights
    masks = {n: torch.zeros_like(p, dtype=torch.bool) for n, p in params}
    def make_fwd(mod, name):
        orig = mod.forward
        def fwd(x):
            w = fq_weight(mod.weight, masks[name], a.tq_bits)
            if isinstance(mod, nn.Conv2d):
                return F.conv2d(x, w, mod.bias, mod.stride, mod.padding, mod.dilation, mod.groups)
            return F.linear(x, w, mod.bias)
        mod.forward = fwd
    for n, _ in params: make_fwd(mods[n.replace(".weight", "")], n)

    def refresh_masks(x, y):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        grads = torch.autograd.grad(loss, plist)
        d = torch.cat([(g.abs() * S.msb_step(p)).flatten() for g, p in zip(grads, plist)])
        k = max(1, int(a.tail_frac * d.numel())); thr = torch.topk(d, k, sorted=False).values.min()
        off = 0
        for n, p in params:
            masks[n] = (d[off:off + p.numel()] >= thr).view_as(p); off += p.numel()

    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl)); it = 0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); ce_s = pen_s = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            if it % a.refresh == 0: refresh_masks(x, y)
            ce = F.cross_entropy(model(x), y)
            pen = torch.zeros((), device=dev)
            if a.lam > 0:
                if it % a.tail_every == 0:
                    pen = pen + a.tail_every * S.tail_penalty(ce, params, a.tail_frac)
                pen = pen + a.clip_w * S.clip_penalty(params, a.clip_q)
            loss = ce + a.lam * pen
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); it += 1
            ce_s += ce.item(); pen_s += pen.item()
        acc_q = S.evaluate(model, vl, dev)                      # evaluated WITH fake quant (MID/LOW at tq-bits)
        nb = len(tl); hi = sum(m.sum().item() for m in masks.values()) / sum(p.numel() for p in plist)
        print(f"ep {ep+1}/{a.epochs}  ce={ce_s/nb:.3f}  pen={pen_s/nb:.3e}  acc@{a.tq_bits}bit={acc_q:.2f}  HIGH={100*hi:.2f}%  {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), a.out)
    print(f"[saved] {a.out}  accuracy at {a.tq_bits} bits (MID/LOW) with HIGH at 8 bits: {acc_q:.2f}")


if __name__ == "__main__":
    main()
