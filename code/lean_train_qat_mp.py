#!/usr/bin/env python
"""
lean_train_qat_mp.py  -  sensitivity-guided MIXED-PRECISION quantization-aware training.
SEPARATE file; imports from lean_sensreg.py and lean_train_qat.py (nothing there is changed).

Bits per bin (all per-output-channel symmetric for MID/LOW, per-tensor 8-bit for HIGH):
    HIGH : 8 bits (fixed)         top  --tail-frac  of weights by damage score
    MID  : --mid-bits             next --mid-frac   of weights
    LOW  : --low-bits             the rest
Bins are refreshed every --refresh steps from the current damage score. The λ tail term (--lam) rides on top.
Prints the average bits per weight so results can be plotted as accuracy vs bits (compare with KURE 3-bit / 4-bit).

Usage (ImageNet ResNet-50):
  python lean_train_qat_mp.py --dataset imagenet --arch resnet50 --imagenet-root <path> --init pretrained \
      --mid-bits 6 --low-bits 3 --lam 40 --lr 0.0005 --bs 32 --epochs 10 --out artifacts/models/imagenet/resnet50/model_mp_6_3_lam40.pth
"""
import argparse, time, torch, torch.nn as nn, torch.nn.functional as F
import lean_sensreg as S
from lean_train_qat import FakeQuant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--mid-bits", type=int, default=6); ap.add_argument("--low-bits", type=int, default=3)
    ap.add_argument("--tail-frac", type=float, default=0.01, help="HIGH share (8 bits)")
    ap.add_argument("--mid-frac", type=float, default=0.08, help="MID share (mid-bits), measured on the CURRENT model")
    ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--tail-every", type=int, default=2)
    ap.add_argument("--refresh", type=int, default=50)
    ap.add_argument("--clip-q", type=float, default=0.99); ap.add_argument("--clip-w", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.init)
    print(f"[init] clean acc (fp32) = {S.evaluate(model, vl, dev):.2f}   bits HIGH/MID/LOW = 8/{a.mid_bits}/{a.low_bits}")
    params = S.weight_params(model); plist = [p for _, p in params]; n_total = sum(p.numel() for p in plist)
    mods = {n.replace(".weight", ""): m for n, m in model.named_modules()}
    hmask = {n: torch.zeros_like(p, dtype=torch.bool) for n, p in params}
    mmask = {n: torch.zeros_like(p, dtype=torch.bool) for n, p in params}
    qm_mid = 2 ** (a.mid_bits - 1) - 1; qm_low = 2 ** (a.low_bits - 1) - 1

    def fq(w, name):
        hi, mi = hmask[name], mmask[name]; lo = ~(hi | mi); red = tuple(range(1, w.dim()))
        sc_l = ((w.detach().abs() * lo).amax(dim=red, keepdim=True) / qm_low).clamp(min=1e-12)
        sc_m = ((w.detach().abs() * mi).amax(dim=red, keepdim=True) / qm_mid).clamp(min=1e-12)
        sc_h = (w.detach().abs().max() / 127).clamp(min=1e-12)
        w_l = FakeQuant.apply(w, sc_l, qm_low); w_m = FakeQuant.apply(w, sc_m, qm_mid); w_h = FakeQuant.apply(w, sc_h, 127)
        return torch.where(hi, w_h, torch.where(mi, w_m, w_l))

    def make_fwd(mod, name):
        def fwd(x):
            w = fq(mod.weight, name)
            if isinstance(mod, nn.Conv2d): return F.conv2d(x, w, mod.bias, mod.stride, mod.padding, mod.dilation, mod.groups)
            return F.linear(x, w, mod.bias)
        mod.forward = fwd
    for n, _ in params: make_fwd(mods[n.replace(".weight", "")], n)

    def refresh(x, y):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y); grads = torch.autograd.grad(loss, plist)
        d = torch.cat([(g.abs() * S.msb_step(p)).flatten() for g, p in zip(grads, plist)])
        kh = max(1, int(a.tail_frac * d.numel())); km = max(1, int((a.tail_frac + a.mid_frac) * d.numel()))
        th = torch.topk(d, kh, sorted=False).values.min(); tm = torch.topk(d, km, sorted=False).values.min(); off = 0
        for n, p in params:
            seg = d[off:off + p.numel()].view_as(p); off += p.numel()
            hmask[n] = seg >= th; mmask[n] = (seg >= tm) & (seg < th)

    def avg_bits():
        nh = sum(m.sum().item() for m in hmask.values()); nm = sum(m.sum().item() for m in mmask.values())
        return (8 * nh + a.mid_bits * nm + a.low_bits * (n_total - nh - nm)) / n_total

    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl)); it = 0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); ce_s = pen_s = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            if it % a.refresh == 0: refresh(x, y)
            ce = F.cross_entropy(model(x), y); pen = torch.zeros((), device=dev)
            if a.lam > 0:
                if it % a.tail_every == 0: pen = pen + a.tail_every * S.tail_penalty(ce, params, a.tail_frac)
                pen = pen + a.clip_w * S.clip_penalty(params, a.clip_q)
            loss = ce + a.lam * pen
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); it += 1
            ce_s += ce.item(); pen_s += pen.item()
        acc_q = S.evaluate(model, vl, dev); nb = len(tl)
        print(f"ep {ep+1}/{a.epochs}  ce={ce_s/nb:.3f}  pen={pen_s/nb:.3e}  acc={acc_q:.2f}  avg_bits={avg_bits():.3f}  {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), a.out)
    print(f"[saved] {a.out}  HIGH/MID/LOW = 8/{a.mid_bits}/{a.low_bits} bits, avg {avg_bits():.3f} bits/weight, accuracy {acc_q:.2f}")


if __name__ == "__main__":
    main()
