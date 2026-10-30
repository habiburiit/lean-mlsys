#!/usr/bin/env python
"""
lean_train_qat_group.py  -  lean_train_qat_mp9.py with GROUP-WISE tiers and TOTAL bit accounting (MLSys item 1).
SEPARATE file; imports from lean_sensreg, lean_train_qat and lean_train_qat_mp9 (nothing there is changed).

What changes against mp9
  1. One bit width (8/6/4/3) per GROUP of --group consecutive weights of a tensor (default 32), not per weight.
     A group is ranked by the max (default) or the sum of the deployed damage |g| * 2^(b-1) s of its members, then the
     same budget rule as mp9 runs on groups: top --tail-frac of the weights (in groups) -> 8 bits, next --mid-frac -> 6,
     demote until the payload budget is met, fill 4 bits greedily.  The per-weight tier tensor is the group tier repeated,
     so the fake-quant forward, the lambda penalty, the saved <out>.bins.pt (tier + scale per weight) and
     lean_pbs_secded.py are exactly the ones of mp9.  Groups are taken along the flattened tensor (output channel major).
  2. The budget is given as TOTAL bits per weight (--total-bits), and the payload budget is derived from it:
        total = payload + tag + scales + parity
        tag     = 2 / G                       (2-bit tier tag per group, uncompressed)
        scales  = 16 bits x (1 per tensor + 3 per output channel) / N      (8-bit per-tensor scale, 6/4/3-bit per-channel scales)
        parity  = BCH(127,43) on the 8-bit tier: 1.95 x 8 x f8 bits/weight (as in lean_pbs_secded.py)
                + sign-plane BCH: t x ceil(log2(n+1)) / n bits/weight        (--parity both, default t=19 n=1023)
     The parity of the 8-bit tier depends on f8, which the budget demotion may change, so the payload budget is
     solved for with the nominal f8 = --tail-frac and the achieved total is printed after every epoch and at the end.
     --budget (payload only, like mp9) is still accepted when --total-bits is not given.
  3. Every epoch line and the final line print payload, tag, scale, parity and total bits per weight.

usage (ImageNet ResNet-50, total 3.5 bits/weight, groups of 32, three seeds in run_lean_group.sh):
  python -u lean_train_qat_group.py --dataset imagenet --arch resnet50 --imagenet-root <path> --init <lean_int8.pth> \
      --total-bits 3.5 --group 32 --lam 40 --qclip 0.99 --pen-frac 0.03 --lr 0.0005 --bs 32 --seed 0 --out <model.pth>
"""
import argparse, math, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import lean_sensreg as S
try:
    import lean_vgg
except Exception as _e:
    print(f"[warn] lean_vgg not importable ({_e}); vgg archs unavailable", flush=True)
from lean_train_qat import FakeQuant
from lean_train_qat_mp9 import tier_scale, LOW_TIERS


def parity_bits_per_weight(f8, mode, t, n):
    """parity cost in bits per weight, same formulas as lean_pbs_secded.py"""
    bch = 1.95 * 8.0 * f8 if mode in ("bch", "both") else 0.0
    sp = t * math.ceil(math.log2(n + 1)) / n if mode == "both" else 0.0
    return bch, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--budget", type=float, default=3.29, help="payload bits per weight (used only when --total-bits is not given)")
    ap.add_argument("--total-bits", type=float, default=0.0, help="TOTAL bits per weight incl. tags, scales and parity; payload budget is derived")
    ap.add_argument("--group", type=int, default=32, help="weights per tier group (one 2-bit tag per group)")
    ap.add_argument("--group-score", choices=["max", "sum"], default="max", help="how a group is ranked from its members' deployed damage")
    ap.add_argument("--parity", choices=["none", "bch", "both"], default="both", help="parity counted in the total: BCH on the 8-bit tier, or that plus the sign-plane code")
    ap.add_argument("--t", type=int, default=19); ap.add_argument("--n", type=int, default=1023)
    ap.add_argument("--tail-frac", type=float, default=0.01); ap.add_argument("--mid-frac", type=float, default=0.08)
    ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--tail-every", type=int, default=2)
    ap.add_argument("--refresh", type=int, default=50); ap.add_argument("--freeze-last", type=int, default=3)
    ap.add_argument("--clip-q", type=float, default=0.99); ap.add_argument("--clip-w", type=float, default=100.0)
    ap.add_argument("--probe-batches", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qclip", type=float, default=0.999)
    ap.add_argument("--pen-frac", type=float, default=0.03); ap.add_argument("--pen-scope", choices=["all", "uncovered"], default="all")
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed); np.random.seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    G = a.group

    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.init)
    print(f"[init] clean acc (fp32) = {S.evaluate(model, vl, dev):.2f}", flush=True)
    params = S.weight_params(model); plist = [p for _, p in params]; n_total = sum(p.numel() for p in plist)
    mods = {n.replace(".weight", ""): m for n, m in model.named_modules()}
    tier = {n: torch.full_like(p, 3, dtype=torch.int8) for n, p in params}

    # ---- fixed overheads (bits per weight) ----
    n_groups = sum(math.ceil(p.numel() / G) for p in plist)
    n_ch = sum(p.shape[0] for p in plist)
    tag_bpw = 2.0 * n_groups / n_total
    scale_bpw = 16.0 * (len(plist) + 3 * n_ch) / n_total
    bch0, sp0 = parity_bits_per_weight(a.tail_frac, a.parity, a.t, a.n)
    if a.total_bits > 0:
        budget = a.total_bits - tag_bpw - scale_bpw - bch0 - sp0
    else:
        budget = a.budget
    print(f"[format] N = {n_total}  groups of {G}: {n_groups}  tag = {tag_bpw:.4f}  scales = {scale_bpw:.4f}  "
          f"parity(nominal f8={a.tail_frac}) = BCH {bch0:.4f} + sign-plane {sp0:.4f} bits/weight", flush=True)
    print(f"[format] total target = {a.total_bits if a.total_bits > 0 else 'n/a'}  ->  payload budget = {budget:.4f} bits/weight   "
          f"group score = {a.group_score}  qclip = {a.qclip}  pen-frac = {a.pen_frac} ({a.pen_scope})", flush=True)
    if budget < 3.0:
        raise SystemExit(f"payload budget {budget:.3f} < 3 bits: total {a.total_bits} is not reachable with 3-bit LOW tier")

    # ---- deployed format (identical to mp9) ----
    def scales(w, name):
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
        t = tier[name]; sc = scales(w, name); fm = torch.full_like(sc, 128.0)
        for b, _ in LOW_TIERS: fm = torch.where(t == b, float(2 ** (b - 1)), fm)
        return fm * sc

    def tail_penalty_fmt(loss, tail_frac):
        grads = torch.autograd.grad(loss, plist, create_graph=True)
        scores = torch.cat([(g.abs() * deployed_step(p, n)).flatten() for g, (n, p) in zip(grads, params)])
        if a.pen_frac > 0:
            if a.pen_scope == "uncovered":
                scores = scores * torch.cat([(tier[n] != 8).flatten() for n, _ in params])
            k = max(1, int(a.pen_frac * scores.numel()))
        else:
            k = max(1, int(tail_frac * scores.numel()))
        return torch.topk(scores, k, sorted=False).values.mean()

    # ---- group bookkeeping: per-weight damage -> per-group score, per-group tier -> per-weight tier ----
    gsize = torch.cat([torch.full((math.ceil(p.numel() / G),), G, dtype=torch.float32) for p in plist]).to(dev)
    for i, p in enumerate(plist):                      # last group of a tensor may be short
        off = sum(math.ceil(q.numel() / G) for q in plist[:i]); r = p.numel() % G
        if r: gsize[off + math.ceil(p.numel() / G) - 1] = r
    gw = gsize / G                                      # group weight (fraction of a full group), for the bit count

    def group_scores(d_per_param):
        out = []
        for d in d_per_param:
            d = d.flatten(); pad = (-d.numel()) % G
            if pad: d = torch.cat([d, d.new_zeros(pad)])
            d = d.view(-1, G); out.append(d.amax(1) if a.group_score == "max" else d.sum(1))
        return torch.cat(out)

    def expand(gt):
        off = 0
        for n_, p in params:
            k = math.ceil(p.numel() / G)
            tier[n_] = gt[off:off + k].repeat_interleave(G)[:p.numel()].view_as(p); off += k

    def refresh(x, y):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y); grads = torch.autograd.grad(loss, plist)
        s = group_scores([g.abs() * deployed_step(p, n_) for g, (n_, p) in zip(grads, params)])
        gt = torch.full((n_groups,), 3, dtype=torch.int8, device=dev)
        k8 = max(1, int(round(a.tail_frac * n_groups))); k6 = max(1, int(round((a.tail_frac + a.mid_frac) * n_groups)))
        top = torch.topk(s, k6).indices; gt[top] = 6; gt[top[:k8]] = 8
        def avg_of(t_):
            return float(((t_.float()) * gw).sum().item() * G / n_total)
        for _ in range(6):                               # demote least sensitive groups until the payload budget is met
            over_bits = (avg_of(gt) - budget) * n_total
            if over_bits <= 0: break
            idx6 = torch.where(gt == 6)[0]
            if idx6.numel() > 0:
                k = int(min(idx6.numel(), over_bits // (3 * G) + 1)); gt[idx6[torch.argsort(s[idx6])[:k]]] = 3; continue
            idx8 = torch.where(gt == 8)[0]
            if idx8.numel() > 0:
                k = int(min(idx8.numel(), over_bits // (2 * G) + 1)); gt[idx8[torch.argsort(s[idx8])[:k]]] = 6; continue
            break
        spare_bits = (budget - avg_of(gt)) * n_total     # fill 4-bit groups greedily (each 3->4 upgrade costs G bits)
        rest = torch.where(gt == 3)[0]; n4 = int(max(0, min(rest.numel(), spare_bits // G)))
        if n4 > 0: gt[rest[torch.argsort(s[rest], descending=True)[:n4]]] = 4
        if avg_of(gt) > budget + 1e-3:
            print(f"[warn] payload budget not met at refresh: {avg_of(gt):.3f} > {budget:.3f}", flush=True)
        expand(gt)

    def accounting():
        c = {b: sum(int((t == b).sum()) for t in tier.values()) for b in (8, 6, 4, 3)}
        payload = sum(b * c[b] for b in c) / n_total; f8 = c[8] / n_total
        bch, sp = parity_bits_per_weight(f8, a.parity, a.t, a.n)
        total = payload + tag_bpw + scale_bpw + bch + sp
        return dict(payload=payload, tag=tag_bpw, scales=scale_bpw, bch=bch, signplane=sp, total=total,
                    share={b: 100 * c[b] / n_total for b in c})

    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl)); it = 0; acc_q = 0.0
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
            if it % 200 == 0:
                print(f"  it {it}  ce={ce_s/((it-1)%len(tl)+1):.3f}  pen={pen_s/((it-1)%len(tl)+1):.3e}  {time.time()-t0:.0f}s", flush=True)
        acc_q = S.evaluate(model, vl, dev); nb = len(tl); A = accounting(); sh = A["share"]
        print(f"ep {ep+1}/{a.epochs}  ce={ce_s/nb:.3f}  pen={pen_s/nb:.3e}  acc={acc_q:.2f}  payload={A['payload']:.3f}  "
              f"total={A['total']:.3f} b/w  tiers 8/6/4/3 = {sh[8]:.2f}/{sh[6]:.2f}/{sh[4]:.1f}/{sh[3]:.1f}%  {time.time()-t0:.0f}s", flush=True)

    torch.save(model.state_dict(), a.out)
    A = accounting()
    torch.save({"high": {n: (t == 8).cpu() for n, t in tier.items()}, "mid": {n: (t == 6).cpu() for n, t in tier.items()},
                "mid4": {n: (t == 4).cpu() for n, t in tier.items()}, "tier": {n: t.cpu() for n, t in tier.items()},
                "scale": {n: scales(p, n).cpu() for n, p in params},
                "budget": budget, "total_bits": a.total_bits, "group": G, "group_score": a.group_score,
                "accounting": A, "qclip": a.qclip, "rank": "deployed"}, a.out + ".bins.pt")
    sh = A["share"]
    print(f"[saved] {a.out}.bins.pt")
    print(f"[bits] payload {A['payload']:.3f} + tag {A['tag']:.3f} + scales {A['scales']:.3f} + BCH {A['bch']:.3f} + sign-plane {A['signplane']:.3f} "
          f"= TOTAL {A['total']:.3f} bits/weight  (target {a.total_bits if a.total_bits > 0 else 'n/a'})")
    print(f"[saved] {a.out}  group {G}: tiers 8/6/4/3 = {sh[8]:.2f}/{sh[6]:.2f}/{sh[4]:.1f}/{sh[3]:.1f}%, accuracy {acc_q:.2f}", flush=True)


if __name__ == "__main__":
    main()
