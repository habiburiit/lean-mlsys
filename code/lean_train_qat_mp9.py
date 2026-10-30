#!/usr/bin/env python
"""
lean_train_qat_mp9.py  -  mp8 with ONE change: the lambda tail penalty acts on a WIDER set than the protected tier:
the top --pen-frac of all weights by deployed damage (default 3%, the top 1% included), while the 8-bit tier stays --tail-frac (1%).
At low bits the weights just below the protected tier are still damaging; mp8 only pushed on the top 1%. --pen-frac 0 = mp8.

mp8 doc: mp6 (budgeted 8/6/4/3 tiers, format-aware lambda) with the two fixes needed for the
3.3-bit LEAN model to survive bit flips. SEPARATE file; nothing in mp5 / mp6 / lean_sensreg /
lean_train_qat / lean_eval_mp is changed. All mp6 flags are kept with the same defaults.

Why the 3.3-bit model died at 100 flips (pbs_mp logs, R18 CIFAR-10): the 6/4/3-bit tiers use a per-channel
MAX-based scale, so one sign flip in the 3-bit tier moves the weight by 4*max|w|/3, as large as an INT8 sign flip,
for 91% of the weights. And the tiers were ranked by INT8 damage of the float model, so after QAT only 14% of the
most damaging deployed weights were in the protected 8-bit tier.

Fix 1  --qclip Q   (default 0.999): the scale of the 6/4/3-bit tiers is the Q-quantile of |w| in that tier
                   (per output channel) instead of the max. The few weights above it saturate (QAT adapts).
                   Every sign flip in those tiers shrinks by max/quantile (typically 2 to 4x) and 3-bit accuracy
                   usually improves, because the grid is no longer set by one outlier. --qclip 1.0 = mp6 behaviour.
Fix 2  --rank deployed (default): tiers are re-assigned by DEPLOYED damage |g| * (own sign-flip step), top
                   --tail-frac -> 8 bits, next --mid-frac -> 6 bits, then the same budget demotion / 4-bit fill as
                   mp5/mp6. So the 8-bit tier IS the most damaging 1% of the deployed model. --rank int8 = mp6 behaviour.
The exact per-weight scales are saved in <out>.bins.pt ("scale") so lean_pbs_mp8.py deploys the same grid.

Usage (CIFAR-10 ResNet-18, 3.29 bits, one fine-tune from the baseline):
  python lean_train_qat_mp9.py --dataset cifar10 --arch resnet18 --init artifacts/models/cifar10/resnet18/model_float32.pth \
      --budget 3.29 --lam 40 --qclip 0.999 --out artifacts/models/cifar10/resnet18/model_mp9_b329_lam100_q99.pth
"""
import argparse, math, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import lean_sensreg as S
try:
    import lean_vgg          # patches S.build_model for vgg* and the >16M-element quantile; harmless for ResNets
except Exception as _e:
    print(f"[warn] lean_vgg not importable ({_e}); vgg archs unavailable", flush=True)
from lean_train_qat import FakeQuant

LOW_TIERS = ((6, 31), (4, 7), (3, 3))


def tier_scale(wabs, m, qm, q):
    """per-output-channel scale for the weights in mask m: Q-quantile (q<1) or max (q>=1) of |w| over the tier members."""
    C = wabs.shape[0]; a = (wabs * m).reshape(C, -1)
    if q >= 1.0:
        s = a.amax(dim=1)
    else:
        n = a.shape[1]; cnt = m.reshape(C, -1).sum(dim=1)                        # members per channel
        srt = a.sort(dim=1).values                                              # zeros (non-members) sort first
        k = (n - cnt + torch.ceil(q * cnt.float()).long() - 1).clamp(0, n - 1)
        s = srt.gather(1, k.view(-1, 1)).squeeze(1)
        s = torch.where(cnt > 0, s, a.amax(dim=1))
    s = (s / qm).clamp(min=1e-12)
    return s.view(C, *([1] * (wabs.dim() - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", required=True); ap.add_argument("--ref", default="", help="reference model for int8 thresholds (default = --init)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--budget", type=float, default=3.29, help="target average bits per weight")
    ap.add_argument("--tail-frac", type=float, default=0.01); ap.add_argument("--mid-frac", type=float, default=0.08)
    ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--tail-every", type=int, default=2)
    ap.add_argument("--refresh", type=int, default=50); ap.add_argument("--freeze-last", type=int, default=3)
    ap.add_argument("--clip-q", type=float, default=0.99); ap.add_argument("--clip-w", type=float, default=100.0)
    ap.add_argument("--probe-batches", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qclip", type=float, default=0.999, help="quantile of |w| used as the 6/4/3-bit tier scale (1.0 = max, like mp6)")
    ap.add_argument("--rank", choices=["deployed", "int8"], default="deployed", help="how tiers are ranked at every refresh")
    ap.add_argument("--pen-frac", type=float, default=0.03, help="penalise the top pen-frac of ALL weights by deployed damage (tier size stays --tail-frac; 0 = mp8 behaviour)")
    ap.add_argument("--pen-scope", choices=["all", "uncovered"], default="all", help="all: top pen-frac including the 8-bit tier; uncovered: top pen-frac outside the 8-bit tier")
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)

    # ---- fixed INT8 thresholds from the reference model (used only with --rank int8; printed for comparison) ----
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
    print(f"[ref] int8 thresholds: 8-bit >= {th8:.3e}, 6-bit >= {th6:.3e}   budget = {a.budget} bits/weight   "
          f"qclip = {a.qclip}   rank = {a.rank}   pen-frac = {a.pen_frac} ({a.pen_scope})", flush=True)

    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.init)
    print(f"[init] clean acc (fp32) = {S.evaluate(model, vl, dev):.2f}", flush=True)
    params = S.weight_params(model); plist = [p for _, p in params]; n_total = sum(p.numel() for p in plist)
    mods = {n.replace(".weight", ""): m for n, m in model.named_modules()}
    tier = {n: torch.full_like(p, 3, dtype=torch.int8) for n, p in params}      # 8 / 6 / 4 / 3

    def scales(w, name):
        """per-weight scale in the deployed format (detached): 8-bit per-tensor max, low tiers per-channel qclip quantile"""
        t = tier[name]; wd = w.detach(); wabs = wd.abs()
        sc = torch.full_like(wd, float(wabs.max() / 127.0)).clamp(min=1e-12)
        for b, qm in LOW_TIERS:
            m = (t == b)
            if m.any():
                sc = torch.where(m, tier_scale(wabs, m, qm, a.qclip).expand_as(wd), sc)
        return sc

    def fq(w, name):
        t = tier[name]; sc = scales(w, name)
        out = FakeQuant.apply(w, sc, 127)
        for b, qm in LOW_TIERS:
            m = (t == b)
            if m.any(): out = torch.where(m, FakeQuant.apply(w, sc, qm), out)
        return out

    def make_fwd(mod, name):
        def fwd(x):
            w = fq(mod.weight, name)
            if isinstance(mod, nn.Conv2d): return F.conv2d(x, w, mod.bias, mod.stride, mod.padding, mod.dilation, mod.groups)
            return F.linear(x, w, mod.bias)
        mod.forward = fwd
    for n, _ in params: make_fwd(mods[n.replace(".weight", "")], n)

    def deployed_step(w, name):
        """sign-bit flip magnitude of every weight in its own deployed format = 2^(b-1) * its scale (detached)"""
        t = tier[name]; sc = scales(w, name)
        fm = torch.full_like(sc, 128.0)
        for b, _ in LOW_TIERS: fm = torch.where(t == b, float(2 ** (b - 1)), fm)
        return fm * sc

    def tail_penalty_fmt(loss, tail_frac):
        grads = torch.autograd.grad(loss, plist, create_graph=True)
        scores = torch.cat([(g.abs() * deployed_step(p, n)).flatten() for g, (n, p) in zip(grads, params)])
        if a.pen_frac > 0:                                   # wider penalty set than the protected tier
            if a.pen_scope == "uncovered":
                unprot = torch.cat([(tier[n] != 8).flatten() for n, _ in params])
                scores = scores * unprot
            k = max(1, int(a.pen_frac * scores.numel()))
        else:
            k = max(1, int(tail_frac * scores.numel()))
        return torch.topk(scores, k, sorted=False).values.mean()

    def refresh(x, y):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y); grads = torch.autograd.grad(loss, plist)
        t = torch.full(( n_total,), 3, dtype=torch.int8, device=dev)
        if a.rank == "int8":
            d = torch.cat([(g.abs() * S.msb_step(p)).flatten() for g, p in zip(grads, plist)])
            t[d >= th6] = 6; t[d >= th8] = 8
        else:
            d = torch.cat([(g.abs() * deployed_step(p, n_)).flatten() for g, (n_, p) in zip(grads, params)])
            k8 = max(1, int(a.tail_frac * n_total)); k6 = max(1, int((a.tail_frac + a.mid_frac) * n_total))
            top = torch.topk(d, k6).indices; t[top] = 6; t[top[:k8]] = 8
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
                if it % a.tail_every == 0: pen = pen + a.tail_every * tail_penalty_fmt(ce, a.tail_frac)
                pen = pen + a.clip_w * S.clip_penalty(params, a.clip_q)
            loss = ce + a.lam * pen
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); it += 1
            ce_s += ce.item(); pen_s += pen.item()
        acc_q = S.evaluate(model, vl, dev); nb = len(tl); avg, sh = tier_stats()
        print(f"ep {ep+1}/{a.epochs}  ce={ce_s/nb:.3f}  pen={pen_s/nb:.3e}  acc={acc_q:.2f}  avg_bits={avg:.3f}  "
              f"tiers 8/6/4/3 = {sh[8]:.2f}/{sh[6]:.2f}/{sh[4]:.1f}/{sh[3]:.1f}%  {time.time()-t0:.0f}s", flush=True)
    # ---- how concentrated is the deployed damage now (same measure the attacker uses) ----
    model.eval(); model.zero_grad(set_to_none=True); dsum = None; nb_ = 0
    for x, y in tl:
        if nb_ >= a.probe_batches: break
        x, y = x.to(dev), y.to(dev); loss = F.cross_entropy(model(x), y); grads = torch.autograd.grad(loss, plist)
        d = torch.cat([(g.abs() * deployed_step(p, n_)).flatten() for g, (n_, p) in zip(grads, params)])
        dsum = d if dsum is None else dsum + d; nb_ += 1
    d = dsum / nb_; k = max(1, int(a.tail_frac * n_total)); top = torch.topk(d, k).indices
    tier_all = torch.cat([tier[n_].flatten() for n_, _ in params])
    share = 100.0 * d[top].sum().item() / d.sum().item(); in8 = 100.0 * float((tier_all[top] == 8).sum()) / k
    print(f"[deployed] top {100*a.tail_frac:.0f}% by deployed damage carry {share:.1f}% of total damage; {in8:.1f}% of them are in the 8-bit tier", flush=True)
    d_un = d * (tier_all != 8); k2 = max(1, int(max(a.pen_frac, a.tail_frac) * n_total))
    print(f"[deployed] outside the 8-bit tier: top {100*max(a.pen_frac, a.tail_frac):.0f}% carry {100.0*torch.topk(d_un, k2).values.sum().item()/d.sum().item():.1f}% of total damage; "
          f"max uncovered damage / max covered damage = {d_un.max().item()/d[tier_all==8].max().clamp(min=1e-30).item():.3f}", flush=True)

    torch.save(model.state_dict(), a.out)
    torch.save({"high": {n: (t == 8).cpu() for n, t in tier.items()}, "mid": {n: (t == 6).cpu() for n, t in tier.items()},
                "mid4": {n: (t == 4).cpu() for n, t in tier.items()}, "tier": {n: t.cpu() for n, t in tier.items()},
                "scale": {n: scales(p, n).cpu() for n, p in params},
                "budget": a.budget, "th8": th8, "th6": th6, "qclip": a.qclip, "rank": a.rank}, a.out + ".bins.pt")
    avg, sh = tier_stats()
    print(f"[saved] {a.out}.bins.pt (tiers + exact per-weight scales)")
    print(f"[saved] {a.out}  budget {a.budget}: avg {avg:.3f} bits/weight, tiers 8/6/4/3 = {sh[8]:.2f}/{sh[6]:.2f}/{sh[4]:.1f}/{sh[3]:.1f}%, accuracy {acc_q:.2f}")


if __name__ == "__main__":
    main()
