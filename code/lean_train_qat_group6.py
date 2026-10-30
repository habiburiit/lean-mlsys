#!/usr/bin/env python
"""
lean_train_qat_group6.py  -  lean_train_qat_group5.py with FIXED tier fractions.  lean_train_qat_mp5.py (the quantizer behind the storage table, model_mp5_b400_* and
model_mp3_b329_*) with GROUP-WISE tiers and TOTAL bit accounting.  SEPARATE file; nothing in mp5 is changed.
lean_train_qat_group.py is the same idea on top of mp9 (the low-bit security mode, qclip + deployed ranking), which
costs about 5 points of accuracy against mp5 at the same bits, so it is NOT the right base for the storage comparison.

Difference from group5: the 8-bit and 6-bit tiers are the top --tail-frac and next --mid-frac of GROUPS by group score
(top-k), not every group that contains a weight above mp5's fixed thresholds.  With thresholds one weight above the
cut pulled its whole group to 8 bits, so the 8-bit tier grew from 1% to 5.6% of the weights and the BCH parity with it
(1.07 instead of 0.34 bits/weight); with top-k the tier is exactly --tail-frac and the parity in the budget is exact.
The thresholds are still computed and printed for reference.

Against mp5, the only changes:
  1. One bit width (8/6/4/3) per GROUP of --group consecutive weights (default 32) with a 2-bit tag.  A group's score is
     the max (default) or the sum of its members' INT8 damage |g| * step; groups with score >= th8 get 8 bits, >= th6 get
     6 bits (the fixed reference thresholds of mp5), then the same self-checking budget demotion and greedy 4-bit fill,
     now on groups.  The per-weight tier is the group tier repeated, so the fake-quant forward, the lambda term and the
     saved <out>.bins.pt are exactly mp5's.
  2. --total-bits T: the payload budget is T minus tags (2/G), scales (16 bits per tensor + 3 per output channel) and
     parity (BCH on the 8-bit tier 1.95 x 8 x f8, plus the t=19 n=1023 sign-plane code when --parity both; the formulas of
     lean_pbs_secded.py).  f8 is taken as the nominal --tail-frac for the budget; the achieved total is printed.
     --budget (payload only) still works when --total-bits is not given.
  3. Every epoch line prints payload and total bits/weight; the final [bits] line is the table entry.

usage (ImageNet ResNet-50, the storage-table recipe, total 4.0 bits/weight):
  python -u lean_train_qat_group6.py --dataset imagenet --arch resnet50 --imagenet-root <path> --init pretrained \
      --total-bits 4.0 --group 32 --lam 0 --lr 0.0005 --bs 32 --seed 0 --out <model.pth>
"""
import argparse, math, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import lean_sensreg as S
from lean_train_qat import FakeQuant


def parity_bits_per_weight(f8, mode, t, n):
    bch = 1.95 * 8.0 * f8 if mode in ("bch", "both") else 0.0
    sp = t * math.ceil(math.log2(n + 1)) / n if mode == "both" else 0.0
    return bch, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", required=True); ap.add_argument("--ref", default="", help="reference model for thresholds (default = --init)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--budget", type=float, default=3.29, help="payload bits per weight (used only when --total-bits is not given)")
    ap.add_argument("--total-bits", type=float, default=0.0, help="TOTAL bits per weight incl. tags, scales and parity")
    ap.add_argument("--group", type=int, default=32); ap.add_argument("--group-score", choices=["max", "sum"], default="max")
    ap.add_argument("--parity", choices=["none", "bch", "both"], default="both")
    ap.add_argument("--t", type=int, default=19); ap.add_argument("--n", type=int, default=1023)
    ap.add_argument("--tail-frac", type=float, default=0.01); ap.add_argument("--mid-frac", type=float, default=0.08)
    ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--tail-every", type=int, default=2)
    ap.add_argument("--refresh", type=int, default=50); ap.add_argument("--freeze-last", type=int, default=3)
    ap.add_argument("--clip-q", type=float, default=0.99); ap.add_argument("--clip-w", type=float, default=100.0)
    ap.add_argument("--probe-batches", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed); np.random.seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    G = a.group

    # ---- fixed thresholds from the reference model (as mp5) ----
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

    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.init)
    print(f"[init] clean acc (fp32) = {S.evaluate(model, vl, dev):.2f}", flush=True)
    params = S.weight_params(model); plist = [p for _, p in params]; n_total = sum(p.numel() for p in plist)
    mods = {n.replace(".weight", ""): m for n, m in model.named_modules()}
    tier = {n: torch.full_like(p, 3, dtype=torch.int8) for n, p in params}

    # ---- fixed overheads (bits per weight) and the payload budget ----
    n_groups = sum(math.ceil(p.numel() / G) for p in plist); n_ch = sum(p.shape[0] for p in plist)
    tag_bpw = 2.0 * n_groups / n_total; scale_bpw = 16.0 * (len(plist) + 3 * n_ch) / n_total
    bch0, sp0 = parity_bits_per_weight(a.tail_frac, a.parity, a.t, a.n)
    budget = (a.total_bits - tag_bpw - scale_bpw - bch0 - sp0) if a.total_bits > 0 else a.budget
    print(f"[ref] thresholds: 8-bit >= {th8:.3e}, 6-bit >= {th6:.3e}", flush=True)
    print(f"[format] N = {n_total}  groups of {G}: {n_groups}  tag = {tag_bpw:.4f}  scales = {scale_bpw:.4f}  "
          f"parity(nominal f8={a.tail_frac}) = BCH {bch0:.4f} + sign-plane {sp0:.4f} bits/weight", flush=True)
    print(f"[format] total target = {a.total_bits if a.total_bits > 0 else 'n/a'}  ->  payload budget = {budget:.4f} bits/weight   group score = {a.group_score}", flush=True)
    if budget < 3.0:
        raise SystemExit(f"payload budget {budget:.3f} < 3 bits: total {a.total_bits} is not reachable with a 3-bit LOW tier")

    # ---- fake quant exactly as mp5: 8-bit per-tensor max, 6/4/3 per-channel max over the tier members ----
    def fq(w, name):
        t = tier[name]; red = tuple(range(1, w.dim()))
        sc8 = (w.detach().abs().max() / 127).clamp(min=1e-12); out = FakeQuant.apply(w, sc8, 127)
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

    # ---- group bookkeeping ----
    gsize = []
    for p in plist:
        k = math.ceil(p.numel() / G); s = torch.full((k,), float(G)); r = p.numel() % G
        if r: s[-1] = r
        gsize.append(s)
    gw = (torch.cat(gsize) / G).to(dev)                         # fraction of a full group, for the bit count

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
        s = group_scores([g.abs() * S.msb_step(p) for g, p in zip(grads, plist)])
        gt = torch.full((n_groups,), 3, dtype=torch.int8, device=dev)
        k8 = max(1, int(round(a.tail_frac * n_groups))); k6 = max(1, int(round((a.tail_frac + a.mid_frac) * n_groups)))
        top = torch.topk(s, k6).indices; gt[top] = 6; gt[top[:k8]] = 8
        def avg_of(t_):
            return float((t_.float() * gw).sum().item() * G / n_total)
        for _ in range(6):
            over_bits = (avg_of(gt) - budget) * n_total
            if over_bits <= 0: break
            idx6 = torch.where(gt == 6)[0]
            if idx6.numel() > 0:
                k = int(min(idx6.numel(), over_bits // (3 * G) + 1)); gt[idx6[torch.argsort(s[idx6])[:k]]] = 3; continue
            idx8 = torch.where(gt == 8)[0]
            if idx8.numel() > 0:
                k = int(min(idx8.numel(), over_bits // (2 * G) + 1)); gt[idx8[torch.argsort(s[idx8])[:k]]] = 6; continue
            break
        spare_bits = (budget - avg_of(gt)) * n_total
        rest = torch.where(gt == 3)[0]; n4 = int(max(0, min(rest.numel(), spare_bits // G)))
        if n4 > 0: gt[rest[torch.argsort(s[rest], descending=True)[:n4]]] = 4
        if avg_of(gt) > budget + 1e-3:
            print(f"[warn] payload budget not met at refresh: {avg_of(gt):.3f} > {budget:.3f}", flush=True)
        expand(gt)

    def accounting():
        c = {b: sum(int((t == b).sum()) for t in tier.values()) for b in (8, 6, 4, 3)}
        payload = sum(b * c[b] for b in c) / n_total; f8 = c[8] / n_total
        bch, sp = parity_bits_per_weight(f8, a.parity, a.t, a.n)
        return dict(payload=payload, tag=tag_bpw, scales=scale_bpw, bch=bch, signplane=sp,
                    total=payload + tag_bpw + scale_bpw + bch + sp, share={b: 100 * c[b] / n_total for b in c})

    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl)); it = 0; acc_q = 0.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); ce_s = pen_s = 0.0; seen = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            if it % a.refresh == 0 and ep < a.epochs - a.freeze_last: refresh(x, y)
            ce = F.cross_entropy(model(x), y); pen = torch.zeros((), device=dev)
            if a.lam > 0:
                if it % a.tail_every == 0: pen = pen + a.tail_every * S.tail_penalty(ce, params, a.tail_frac)
                pen = pen + a.clip_w * S.clip_penalty(params, a.clip_q)
            loss = ce + a.lam * pen
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); it += 1
            ce_s += ce.item(); pen_s += pen.item(); seen += 1
            if it % 200 == 0:
                print(f"  it {it}  ce={ce_s/seen:.3f}  pen={pen_s/seen:.3e}  {time.time()-t0:.0f}s", flush=True)
        acc_q = S.evaluate(model, vl, dev); nb = len(tl); A = accounting(); sh = A["share"]
        print(f"ep {ep+1}/{a.epochs}  ce={ce_s/nb:.3f}  pen={pen_s/nb:.3e}  acc={acc_q:.2f}  payload={A['payload']:.3f}  "
              f"total={A['total']:.3f} b/w  tiers 8/6/4/3 = {sh[8]:.2f}/{sh[6]:.2f}/{sh[4]:.1f}/{sh[3]:.1f}%  {time.time()-t0:.0f}s", flush=True)

    torch.save(model.state_dict(), a.out); A = accounting(); sh = A["share"]
    torch.save({"high": {n: (t == 8).cpu() for n, t in tier.items()}, "mid": {n: (t == 6).cpu() for n, t in tier.items()},
                "mid4": {n: (t == 4).cpu() for n, t in tier.items()}, "tier": {n: t.cpu() for n, t in tier.items()},
                "budget": budget, "total_bits": a.total_bits, "group": G, "group_score": a.group_score,
                "accounting": A, "th8": th8, "th6": th6}, a.out + ".bins.pt")
    print(f"[saved] {a.out}.bins.pt")
    print(f"[bits] payload {A['payload']:.3f} + tag {A['tag']:.3f} + scales {A['scales']:.3f} + BCH {A['bch']:.3f} + sign-plane {A['signplane']:.3f} "
          f"= TOTAL {A['total']:.3f} bits/weight  (target {a.total_bits if a.total_bits > 0 else 'n/a'})")
    print(f"[saved] {a.out}  group {G}: tiers 8/6/4/3 = {sh[8]:.2f}/{sh[6]:.2f}/{sh[4]:.1f}/{sh[3]:.1f}%, accuracy {acc_q:.2f}", flush=True)


if __name__ == "__main__":
    main()
