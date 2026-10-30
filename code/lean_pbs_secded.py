#!/usr/bin/env python
"""
lean_pbs_secded.py  -  PBS on a mixed-precision (mp8 / mp3..mp7) model with TWO codes:
  1. BCH on the 8-bit tier (same as pbs_mp.py --protect high): flips there are always corrected.
  2. SECDED (Hamming 72,64) on the SIGN-BIT PLANE of every other weight: 64 consecutive sign bits + 8 parity bits.
     Cost = 8/64 = 0.125 bit per covered weight. One flipped sign bit inside a 64-weight group is corrected;
     two or more are detected but not corrected (the worst case for the defender: the flips take effect).
Attacker: white-box and adaptive. It knows that a single flip in a fresh group is corrected, so each step it
searches for the best PAIR of sign flips inside one group (ranked by the sum of their first-order gains, the real
loss of the top candidates is evaluated, the worst pair is committed, 2 flips of budget). Groups that already hold
2 flips are open: a single extra flip there costs 1 and takes effect. This is the same progressive bit search as
pbs_mp.py, with the pair move added. Flip budgets are counted the way the paper counts them (number of bits flipped).
Weights are flattened per layer and grouped 64 at a time, so a group never spans two layers.

SEPARATE file; imports lean_pbs_mp8 (deployment on saved scales), pbs_mp, lean_eval_mp, lean_sensreg: nothing changed.

  python lean_pbs_secded.py --dataset cifar10 --arch resnet18 --ckpt artifacts/models/cifar10/resnet18/model_mp8_b329_lam100_q99.pth \
      --flips 10,25,50,100,200,300,500
  --secded off  reproduces pbs_mp.py --protect high (BCH only) for a same-script comparison.
"""
import argparse, csv, os, time
import numpy as np, torch, torch.nn.functional as F
import lean_sensreg as S
try:
    import lean_vgg          # patches S.build_model for vgg* and the >16M-element quantile; harmless for ResNets
except Exception as _e:
    print(f"[warn] lean_vgg not importable ({_e}); vgg archs unavailable", flush=True)
import lean_eval_mp as EM
import lean_pbs_mp8            # installs the mp8-aware build_mp
from pbs_mp import attack_loss

GROUP = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--label", default=None)
    ap.add_argument("--mid-bits", type=int, default=6); ap.add_argument("--low-bits", type=int, default=3)
    ap.add_argument("--tail-frac", type=float, default=0.01); ap.add_argument("--mid-frac", type=float, default=0.08)
    ap.add_argument("--secded", choices=["on", "off"], default="on")
    ap.add_argument("--interleave", choices=["on", "off"], default="on", help="on: the 64 sign bits of a codeword are spread over random weights of the whole network (address interleaving); off: 64 consecutive weights")
    ap.add_argument("--t", type=int, default=1, help="correction capability of the sign-plane code: t=1 with n=64 is SECDED(72,64); otherwise a BCH code over n sign bits with t*log2(n+1) parity bits")
    ap.add_argument("--promote", type=float, default=0.0, help="at deployment, move the top PROMOTE fraction of weights by deployed damage (outside the 8-bit tier) into the 8-bit tier under the BCH code, keeping their deployed values (the function is unchanged); no retraining, costs ~5 bits per promoted weight plus their BCH parity")
    ap.add_argument("--n", type=int, default=0, help="codeword length in sign bits (0 = 64 for t=1, 127 otherwise); 255, 511, 1023 give cheaper parity per weight for the same t")
    ap.add_argument("--flips", default="10,25,50,100,200,300,500")
    ap.add_argument("--topk", type=int, default=50, help="candidate moves (pairs or singles) evaluated per step")
    ap.add_argument("--attack-batches", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab = a.label or os.path.basename(a.ckpt).replace(".pth", "")
    flips_list = [int(x) for x in a.flips.split(",")]; max_flips = max(flips_list)

    _, vl = S.cifar_loaders(a.data, 128, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.ckpt); params = S.weight_params(model)
    saved = torch.load(a.ckpt + ".bins.pt", map_location="cpu")
    w_float = [p.detach().clone() for _, p in params]                  # float weights, before in-place quantization
    store = EM.build_mp(params, None, a, saved); model.eval()
    clean = S.evaluate(model, vl, dev)
    plist = [p for _, p in params]; n_total = sum(p.numel() for p in plist)
    q_all = torch.cat([q.flatten() for _, _, q, *_ in store]); sc_all = torch.cat([sc.flatten() for _, _, _, sc, *_ in store])
    fm_all = torch.cat([fm.flatten() for _, _, _, _, fm, *_ in store])
    tier_all = torch.cat([saved["tier"][n].flatten().to(torch.int64) for n, _ in params]).to(q_all.device)
    avg_bits = sum((torch.log2(fm).sum() + fm.numel()).item() for _, _, _, _, fm, _, _ in store) / n_total
    prot = tier_all == 8                                        # BCH-corrected weights
    if a.promote > 0:
        # deployment-time promotion: rank uncovered weights by deployed damage on the attack batches, move the top
        # fraction to the per-tensor 8-bit grid (from their float value) and under the BCH code. Nothing is trained.
        pb = []
        for x, y in vl:
            pb.append((x.to(dev), y.to(dev)))
            if len(pb) >= a.attack_batches: break
        # gradient accumulated in chunks of 32 images so that VGG-size models fit in memory
        model.zero_grad(set_to_none=True); nimg = sum(x.shape[0] for x, _ in pb)
        for x, y in pb:
            for i0 in range(0, x.shape[0], 32):
                xs_, ys_ = x[i0:i0+32], y[i0:i0+32]
                (F.cross_entropy(model(xs_), ys_, reduction="sum") / nimg).backward()
        g0 = torch.cat([p_.grad.abs().flatten() for p_ in plist]); model.zero_grad(set_to_none=True)
        dmg = g0 * fm_all * sc_all; dmg[prot] = -1.0
        k = int(a.promote * n_total); idx = torch.topk(dmg, k).indices
        sizes_ = [p.numel() for p in plist]; offs_ = np.cumsum([0] + sizes_)
        with torch.no_grad():
            for i, (n, p) in enumerate(params):
                sel = idx[(idx >= offs_[i]) & (idx < offs_[i+1])] - offs_[i]
                if sel.numel() == 0: continue
                # keep the weight's DEPLOYED value (the model was fine-tuned with it); only re-store it on the 8-bit grid
                pf0 = p.flatten(); s8 = (w_float[i].abs().max() / 127.0).clamp(min=1e-12)
                q8 = torch.round(pf0[sel] / s8).clamp(-128, 127)
                q_all[offs_[i] + sel] = q8; sc_all[offs_[i] + sel] = s8; fm_all[offs_[i] + sel] = 128.0
                pf = p.flatten(); pf[sel] = q8 * s8; p.copy_(pf.view_as(p))
        tier_all[idx] = 8; prot = tier_all == 8
        avg_bits = (avg_bits * n_total + 5.0 * k) / n_total          # promoted weights were 3-bit; now 8
        clean_p = S.evaluate(model, vl, dev)
        print(f"[{lab}] promoted top {100*a.promote:.2f}% uncovered weights to the 8-bit tier: tier now {100*float(prot.sum())/n_total:.2f}%, "
              f"avg_bits {avg_bits:.3f}, clean {clean:.2f} -> {clean_p:.2f}", flush=True); clean = clean_p
        del pb, g0, dmg, w_float; torch.cuda.empty_cache()
    # SECDED groups: 64 consecutive weights per layer; group id global
    sizes = [p.numel() for p in plist]; offs = np.cumsum([0] + sizes)
    import math
    GROUP = a.n if a.n > 0 else (64 if a.t == 1 else 127)
    parity = 8 if (a.t == 1 and GROUP == 64) else a.t * int(math.ceil(math.log2(GROUP + 1)))   # BCH bound: m*t parity bits
    if a.interleave == "on":
        gen = torch.Generator(device="cpu").manual_seed(1234)
        perm = torch.randperm(n_total, generator=gen).to(q_all.device)           # fixed address interleaving
        gid = torch.empty(n_total, dtype=torch.int64, device=q_all.device); gid[perm] = torch.arange(n_total, device=q_all.device) // GROUP
        n_groups = (n_total + GROUP - 1) // GROUP
    else:
        gid = torch.empty(n_total, dtype=torch.int64, device=q_all.device); g0 = 0
        for i, sz in enumerate(sizes):
            ng = (sz + GROUP - 1) // GROUP
            gid[offs[i]:offs[i+1]] = g0 + torch.arange(sz, device=q_all.device) // GROUP; g0 += ng
        n_groups = g0
    covered = (~prot); NEED = a.t + 1                                              # flips per codeword before they take effect
    secded_bits = parity * n_groups / n_total if a.secded == "on" else 0.0
    bch_bits = 1.95 * 8.0 * float(prot.sum()) / n_total          # BCH(127,43) parity on the 8-bit tier, as in the paper
    print(f"[{lab}] clean={clean:.2f}  avg_bits={avg_bits:.3f}  BCH on 8-bit tier ({100*float(prot.sum())/n_total:.2f}%)  "
          f"sign-plane code={a.secded} t={a.t} interleave={a.interleave} ({100*float(covered.sum())/n_total:.1f}% of weights, {n_groups} groups of {GROUP})  "
          f"parity: BCH {bch_bits:.3f} + SECDED {secded_bits:.3f} bits/weight -> total {avg_bits+bch_bits+secded_bits:.3f}", flush=True)

    batches = []
    for x, y in vl:
        batches.append((x.to(dev), y.to(dev)))
        if len(batches) >= a.attack_batches: break

    def write(q_flat):
        with torch.no_grad():
            for i, p in enumerate(plist):
                p.copy_((q_flat[offs[i]:offs[i+1]] * sc_all[offs[i]:offs[i+1]]).view_as(p))

    def flip_code(q_flat, idxs):
        qf = q_flat.clone()
        for idx in idxs:
            v = qf[idx]; m = fm_all[idx]; qf[idx] = torch.where(v >= 0, v - m, v + m)
        return qf

    gcount = torch.zeros(n_groups, dtype=torch.int64, device=q_all.device)   # flips committed per group
    flipped = torch.zeros(n_total, dtype=torch.bool, device=q_all.device)
    q_cur = q_all.clone(); results = {}; t0 = time.time(); nflips = 0; last_loss = attack_loss(model, batches)

    while nflips < max_flips:
        model.zero_grad(set_to_none=True)                       # one batch at a time: half the activation memory
        for x, y in batches: (F.cross_entropy(model(x), y) / len(batches)).backward()
        g = torch.cat([p_.grad.flatten() for p_ in plist]); model.zero_grad(set_to_none=True)
        delta = torch.where(q_cur >= 0, -fm_all, fm_all) * sc_all
        gain = g * delta; gain[prot] = -float("inf"); gain[flipped] = -float("inf")
        moves = []                                                   # list of (list_of_idx, cost)
        if a.secded == "off":
            for i in torch.topk(gain, a.topk).indices.tolist(): moves.append(([i], 1))
        else:
            open_grp = gcount[gid] >= NEED                            # groups already past the correction limit
            g_open = gain.clone(); g_open[~open_grp] = -float("inf")
            k1 = min(a.topk, int(open_grp.sum()))
            if k1 > 0:
                for i in torch.topk(g_open, k1).indices.tolist():
                    if g_open[i] > -float("inf"): moves.append(([i], 1))
            # pairs: best two candidates in the same group among groups with 0 or 1 flips so far
            half = gid[~open_grp]; gh = gain[~open_grp]; idx_h = torch.where(~open_grp)[0]
            # for each group: the best candidates in it, enough to reach NEED flips (strongest first; with
            # interleaving the extra members are random weights, so the attacker fills up with its best available ones)
            order = torch.argsort(gh, descending=True)
            members = {}
            for pos in order.tolist()[: 40 * a.topk]:
                if gh[pos] == -float("inf"): break
                grp = int(half[pos]); need = NEED - int(gcount[grp])
                lst = members.setdefault(grp, [])
                if len(lst) < need: lst.append((int(idx_h[pos]), float(gh[pos])))
            full = []
            for grp, lst in members.items():
                need = NEED - int(gcount[grp])
                if len(lst) < need:                                   # fill with any unflipped member of the group
                    extra = torch.where((gid == grp) & ~flipped & ~prot)[0].tolist()
                    chosen = {i for i, _ in lst}
                    for e in extra:
                        if len(lst) >= need: break
                        if e not in chosen: lst.append((e, float(gain[e]))); chosen.add(e)
                if len(lst) == need: full.append((grp, [i for i, _ in lst], sum(gg for _, gg in lst)))
            for grp, idxs, gsum in sorted(full, key=lambda x: -x[2])[: a.topk]:
                moves.append((idxs, len(idxs)))
        best, best_l, best_cost = None, -1.0, 1
        for idxs, cost in moves[: 2 * a.topk]:
            # effective weights under the code: flips in a group count only once the group holds >= 2 flips
            grp = int(gid[idxs[0]]); eff = list(idxs)
            if a.secded == "on" and int(gcount[grp]) + len(idxs) < NEED: continue     # would be corrected: no gain
            if a.secded == "on" and 0 < int(gcount[grp]) < NEED:                      # earlier corrected flips now take effect too
                eff = eff + torch.where((gid == grp) & flipped)[0].tolist()
            qf = flip_code(q_cur, eff); write(qf); l = attack_loss(model, batches)
            if l > best_l: best_l, best, best_cost = l, (idxs, eff), cost
        if best is None: print(f"   [{lab}] attacker found no effective move", flush=True); break
        idxs, eff = best
        q_cur = flip_code(q_cur, eff); write(q_cur)
        for i in idxs: flipped[i] = True
        gcount[int(gid[idxs[0]])] += len(idxs); prev = nflips; nflips += best_cost; last_loss = best_l
        for n in flips_list:
            if prev < n <= nflips:
                acc = S.evaluate(model, vl, dev); results[n] = acc
                print(f"   [{lab}] code={a.secded} t={a.t} n={GROUP} il={a.interleave} flips={n:4d}  loss={best_l:.3f}  acc={acc:6.2f}%  groups hit={int((gcount>0).sum())}  ({time.time()-t0:.0f}s)", flush=True)
    for n in flips_list:
        if n not in results: results[n] = S.evaluate(model, vl, dev)
    print(f"\n{'model':28s}{'secded':8s}{'bits+parity':>12s}" + "".join(f"{'n='+str(n):>9s}" for n in flips_list))
    print(f"{lab:28s}{a.secded:8s}{avg_bits+bch_bits+secded_bits:12.3f}" + "".join(f"{results[n]:9.2f}" for n in flips_list))
    os.makedirs("sensreg_results/pbs_mp", exist_ok=True)
    with open(f"sensreg_results/pbs_mp/{a.dataset}_{a.arch}_{lab}_secded_{a.secded}_t{a.t}_n{GROUP}_il{a.interleave}_pr{a.promote}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "secded", "bits_plus_parity", "clean"] + [f"flips_{n}" for n in flips_list])
        w.writerow([lab, a.secded, f"{avg_bits+bch_bits+secded_bits:.3f}", f"{clean:.2f}"] + [f"{results[n]:.2f}" for n in flips_list])


if __name__ == "__main__":
    main()
