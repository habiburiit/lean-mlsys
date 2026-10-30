#!/usr/bin/env python
"""
lean_train_qat_mp5.py  -  same as mp4 but the budget is enforced by a self-checking loop (recomputed from the tiers
themselves, demote until avg <= budget, warn if not). Use this one; mp4 left over-budget tiers in some runs.
mp4 note: same as mp3 PLUS a hard budget: if the 8/6-bit tiers overflow the budget, the least
sensitive 6-bit weights are demoted to 3 bits, then (if still over) the least sensitive 8-bit weights to 6, until the
average-bit budget holds. The 4-bit tier is filled with any spare budget as before. Average bits always == --budget.
SEPARATE file; imports from lean_sensreg.py and lean_train_qat.py (nothing there is changed).

Difference from mp/mp2: bin sizes are NOT fixed percentages. Bits follow the sensitivity against FIXED thresholds
taken once from a reference model (--ref, default: the --init model), and a 4-bit tier is filled greedily with the
next most sensitive weights until the average-bit budget is met:
    8 bits : damage >= HIGH threshold of the reference
    6 bits : damage >= MID  threshold of the reference
    4 bits : next most sensitive weights, as many as the budget allows
    3 bits : the rest
So a model whose sensitivity was skewed by lambda (fewer weights above the thresholds) gets more weights at 4 bits
for the SAME budget. Thresholds and tiers are refreshed every --refresh steps (frozen for the last --freeze-last
epochs) and the final assignment is saved to <out>.bins.pt (readable by lean_eval_mp.py: tiers 8 and 6 map to its
HIGH/MID; the 4-bit tier is stored under "mid4").

Usage (ImageNet ResNet-50, same budget as the 8/6/3 fixed-share run = 3.29 bits/weight):
  python lean_train_qat_mp3.py --dataset imagenet --arch resnet50 --imagenet-root <path> --init pretrained \
      --budget 3.29 --lam 40 --lr 0.0005 --bs 32 --epochs 10 --out artifacts/models/imagenet/resnet50/model_mp3_b329_lam40.pth
"""
import argparse, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import lean_sensreg as S
from lean_train_qat import FakeQuant

TIERS = {"t8": (8, 127), "t6": (6, 31), "t4": (4, 7), "t3": (3, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", required=True); ap.add_argument("--ref", default="", help="reference model for thresholds (default = --init)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--budget", type=float, default=3.29, help="target average bits per weight")
    ap.add_argument("--tail-frac", type=float, default=0.01); ap.add_argument("--mid-frac", type=float, default=0.08)
    ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--tail-every", type=int, default=2)
    ap.add_argument("--refresh", type=int, default=50); ap.add_argument("--freeze-last", type=int, default=3)
    ap.add_argument("--clip-q", type=float, default=0.99); ap.add_argument("--clip-w", type=float, default=100.0)
    ap.add_argument("--probe-batches", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)

    # ---- fixed thresholds from the reference model (damage = |g| * INT8 flip step), computed once ----
    ref = S.build_model(a.arch).to(dev); S.load_state_flex(ref, a.ref or a.init); rparams = S.weight_params(ref)
    acc_d = None; n = 0; ref.eval()
    for x, y in tl:
        if n >= a.probe_batches: break
        x, y = x.to(dev), y.to(dev); loss = F.cross_entropy(ref(x), y)
        grads = torch.autograd.grad(loss, [p for _, p in rparams])
        d = torch.cat([(g.abs() * S.msb_step(p)).flatten().cpu() for g, (_, p) in zip(grads, rparams)])
        acc_d = d if acc_d is None else acc_d + d; n += 1; ref.zero_grad(set_to_none=True)
    dref = (acc_d / n).numpy(); th8 = float(np.quantile(dref, 1 - a.tail_frac)); th6 = float(np.quantile(dref, 1 - a.tail_frac - a.mid_frac))
    del ref; torch.cuda.empty_cache()
    print(f"[ref] thresholds: 8-bit >= {th8:.3e}, 6-bit >= {th6:.3e}   budget = {a.budget} bits/weight")

    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.init)
    print(f"[init] clean acc (fp32) = {S.evaluate(model, vl, dev):.2f}")
    params = S.weight_params(model); plist = [p for _, p in params]; n_total = sum(p.numel() for p in plist)
    mods = {n.replace(".weight", ""): m for n, m in model.named_modules()}
    tier = {n: torch.full_like(p, 3, dtype=torch.int8) for n, p in params}      # 8 / 6 / 4 / 3

    def fq(w, name):
        t = tier[name]; red = tuple(range(1, w.dim())); out = None
        sc8 = (w.detach().abs().max() / 127).clamp(min=1e-12); w8 = FakeQuant.apply(w, sc8, 127)
        out = w8
        for b, qm in ((6, 31), (4, 7), (3, 3)):
            m = (t == b)
            sc = ((w.detach().abs() * m).amax(dim=red, keepdim=True) / qm).clamp(min=1e-12)
            out = torch.where(m, FakeQuant.apply(w, sc, qm), out)
        return out

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
        t = torch.full_like(d, 3, dtype=torch.int8)
        t[d >= th6] = 6; t[d >= th8] = 8
        def avg_of(t_):
            return (8.0 * (t_ == 8).sum() + 6.0 * (t_ == 6).sum() + 4.0 * (t_ == 4).sum() + 3.0 * (t_ == 3).sum()).item() / n_total
        # hard budget: demote least sensitive 6-bit weights to 3, then 8-bit to 6, until avg <= budget (self-checked)
        for _ in range(6):
            over_bits = (avg_of(t) - a.budget) * n_total
            if over_bits <= 0: break
            idx6 = torch.where(t == 6)[0]
            if idx6.numel() > 0:
                k = int(min(idx6.numel(), over_bits // 3 + 1)); t[idx6[torch.argsort(d[idx6])[:k]]] = 3; continue
            idx8 = torch.where(t == 8)[0]
            if idx8.numel() > 0:
                k = int(min(idx8.numel(), over_bits // 2 + 1)); t[idx8[torch.argsort(d[idx8])[:k]]] = 6; continue
            break
        # fill 4-bit tier greedily with the next most sensitive weights until the budget is met
        spare_bits = (a.budget - avg_of(t)) * n_total
        rest = torch.where(t == 3)[0]; n4 = int(max(0, min(rest.numel(), spare_bits // 1)))
        if n4 > 0:
            t[rest[torch.argsort(d[rest], descending=True)[:n4]]] = 4
        if avg_of(t) > a.budget + 1e-3:
            print(f"[warn] budget not met at refresh: avg {avg_of(t):.3f} > {a.budget}", flush=True)
        off = 0
        for n_, p in params:
            tier[n_] = t[off:off + p.numel()].view_as(p); off += p.numel()

    def tier_stats():
        c = {b: sum(int((t == b).sum()) for t in tier.values()) for b in (8, 6, 4, 3)}
        avg = sum(b * c[b] for b in c) / n_total
        return avg, {b: 100 * c[b] / n_total for b in c}

    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl)); it = 0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); ce_s = pen_s = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            if it % a.refresh == 0 and ep < a.epochs - a.freeze_last: refresh(x, y)
            ce = F.cross_entropy(model(x), y); pen = torch.zeros((), device=dev)
            if a.lam > 0:
                if it % a.tail_every == 0: pen = pen + a.tail_every * S.tail_penalty(ce, params, a.tail_frac)
                pen = pen + a.clip_w * S.clip_penalty(params, a.clip_q)
            loss = ce + a.lam * pen
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); it += 1
            ce_s += ce.item(); pen_s += pen.item()
        acc_q = S.evaluate(model, vl, dev); nb = len(tl); avg, sh = tier_stats()
        print(f"ep {ep+1}/{a.epochs}  ce={ce_s/nb:.3f}  pen={pen_s/nb:.3e}  acc={acc_q:.2f}  avg_bits={avg:.3f}  "
              f"tiers 8/6/4/3 = {sh[8]:.2f}/{sh[6]:.2f}/{sh[4]:.1f}/{sh[3]:.1f}%  {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), a.out)
    torch.save({"high": {n: (t == 8).cpu() for n, t in tier.items()}, "mid": {n: (t == 6).cpu() for n, t in tier.items()},
                "mid4": {n: (t == 4).cpu() for n, t in tier.items()}, "tier": {n: t.cpu() for n, t in tier.items()},
                "budget": a.budget, "th8": th8, "th6": th6}, a.out + ".bins.pt")
    avg, sh = tier_stats()
    print(f"[saved] {a.out}.bins.pt")
    print(f"[saved] {a.out}  budget {a.budget}: avg {avg:.3f} bits/weight, tiers 8/6/4/3 = {sh[8]:.2f}/{sh[6]:.2f}/{sh[4]:.1f}/{sh[3]:.1f}%, accuracy {acc_q:.2f}")


if __name__ == "__main__":
    main()
