#!/usr/bin/env python
"""
pbs_mp.py  -  progressive bit search (PBS) on MIXED-PRECISION models, in their deployed format.
SEPARATE file; imports lean_eval_mp.build_mp and lean_sensreg (nothing there is changed).

Why a separate attacker: pbs_core.PBSAttack and the ECC engine work on the INT8 bit planes of the
original pipeline. A mixed-precision model has 8 / 6 / 4 / 3-bit codes side by side, so the search and
the sign-bit flip have to be done per weight in that weight's own format. The algorithm is the same PBS
procedure: at every step rank candidate bits by gradient, evaluate the real loss of each candidate flip
on the attack batches, commit the most damaging one, repeat.

Protection:
  --protect none : attacker may flip any weight
  --protect high : the selective code ECC on the 8-bit tier (the HIGH bin saved in <ckpt>.bins.pt). Flips landing in that
                   tier are corrected (BCH t=12 per codeword; a PBS budget of <= 500 flips cannot exceed t in
                   any codeword), so the attacker can only damage the 6/4/3-bit tiers.
Coverage printed = share of weights in the 8-bit tier (that is what the code has to store parity for).

Usage (CIFAR-10 ResNet-18, 3.29-bit models):
  python pbs_mp.py --dataset cifar10 --arch resnet18 --ckpt artifacts/models/cifar10/resnet18/model_mp3_b329_lam40_s0.pth \
      --label mp329_lam40 --protect none --flips 10,25,50,100,200,300,500
  python pbs_mp.py ... --protect high
"""
import argparse, csv, os, time
import numpy as np, torch, torch.nn.functional as F
import lean_sensreg as S
import lean_eval_mp as EM


def attack_loss(model, batches):
    tot = 0.0
    with torch.no_grad():
        for x, y in batches: tot += F.cross_entropy(model(x), y).item()
    return tot / len(batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", required=True, help="mixed-precision checkpoint; <ckpt>.bins.pt must exist")
    ap.add_argument("--label", default=None)
    ap.add_argument("--mid-bits", type=int, default=6); ap.add_argument("--low-bits", type=int, default=3)
    ap.add_argument("--tail-frac", type=float, default=0.01); ap.add_argument("--mid-frac", type=float, default=0.08)
    ap.add_argument("--protect", choices=["none", "high", "high6", "selcode"], default="none", help="none | high = ECC on 8-bit tier | high6 = ECC on 8-bit and 6-bit tiers | selcode = ECC on the top --cov fraction by DEPLOYED-format damage (like the selective code on INT8)")
    ap.add_argument("--cov", type=float, default=0.01, help="coverage fraction for --protect selcode")
    ap.add_argument("--flips", default="10,25,50,100,200,300,500")
    ap.add_argument("--topk", type=int, default=50, help="candidate bits evaluated per PBS step")
    ap.add_argument("--attack-batches", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab = a.label or os.path.basename(a.ckpt).replace(".pth", "")
    flips_list = [int(x) for x in a.flips.split(",")]

    _, vl = S.cifar_loaders(a.data, 128, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.ckpt); params = S.weight_params(model)
    saved = torch.load(a.ckpt + ".bins.pt", map_location="cpu")
    store = EM.build_mp(params, None, a, saved)          # quantize in place, per-weight tier from training
    model.eval()
    clean = S.evaluate(model, vl, dev)
    n_total = sum(p.numel() for _, p in params)
    n_high = sum(int(h.sum()) for *_, h, _ in store)
    avg_bits = sum((torch.log2(fm).sum() + fm.numel()).item() for _, _, _, _, fm, _, _ in store) / n_total
    cov = 100.0 * n_high / n_total
    print(f"[{lab}] clean={clean:.2f}  avg_bits={avg_bits:.3f}  8-bit tier={cov:.2f}% of weights  protect={a.protect}", flush=True)

    # flat views used by the search
    plist = [p for _, p in params]
    q_all = torch.cat([q.flatten() for _, _, q, *_ in store])                 # integer codes
    sc_all = torch.cat([sc.flatten() for _, _, _, sc, *_ in store])          # scales
    fm_all = torch.cat([fm.flatten() for _, _, _, _, fm, *_ in store])       # sign-bit magnitude 2^(b-1)
    # attack batches (fixed, like pbs_core.collect_batches)
    batches = []
    for x, y in vl:
        batches.append((x.to(dev), y.to(dev)))
        if len(batches) >= a.attack_batches: break
    tier_all = torch.cat([saved["tier"][n].flatten().to(torch.int64) for n, _ in params]).to(q_all.device)
    if a.protect == "high": prot = tier_all == 8
    elif a.protect == "high6": prot = tier_all >= 6
    elif a.protect == "selcode":
        # damage in the deployed format: |g| * sign-flip step of that weight's own code, g on the attack batches
        model.zero_grad(set_to_none=True); loss0 = 0.0
        for x, y in batches: loss0 = loss0 + F.cross_entropy(model(x), y)
        g0 = torch.cat([gg.abs().flatten() for gg in torch.autograd.grad(loss0 / len(batches), plist)])
        dmg = g0 * fm_all * sc_all; k = max(1, int(a.cov * n_total))
        prot = torch.zeros_like(q_all, dtype=torch.bool); prot[torch.topk(dmg, k).indices] = True
        in8 = 100.0 * float((prot & (tier_all == 8)).sum()) / float(prot.sum())
        print(f"[{lab}] selcode top {100*a.cov:.2f}% by deployed damage: {in8:.1f}% of them are in the 8-bit tier", flush=True)
    else: prot = torch.zeros_like(q_all, dtype=torch.bool)
    cov = 100.0 * float(prot.sum()) / n_total   # ECC coverage actually used in this run
    shares = {b: 100.0 * float((tier_all == b).sum()) / n_total for b in (8, 6, 4, 3)}
    print(f"[{lab}] tiers 8/6/4/3 = {shares[8]:.2f}/{shares[6]:.2f}/{shares[4]:.1f}/{shares[3]:.1f}%   ECC coverage = {cov:.2f}%", flush=True)
    sizes = [p.numel() for p in plist]; offs = np.cumsum([0] + sizes)
    flipped = torch.zeros_like(q_all, dtype=torch.bool)

    def write(q_flat):
        with torch.no_grad():
            for i, p in enumerate(plist):
                p.copy_((q_flat[offs[i]:offs[i+1]] * sc_all[offs[i]:offs[i+1]]).view_as(p))

    def flip_code(q_flat, idx):
        qf = q_flat.clone(); v = qf[idx]; m = fm_all[idx]
        qf[idx] = torch.where(v >= 0, v - m, v + m)                           # sign-bit flip in that weight's format
        return qf

    results = {}; q_cur = q_all.clone(); t0 = time.time()
    for step in range(1, max(flips_list) + 1):
        # 1. gradient ranking of candidate sign-bit flips (first-order damage of a flip in this format)
        model.zero_grad(set_to_none=True); loss = 0.0
        for x, y in batches: loss = loss + F.cross_entropy(model(x), y)
        grads = torch.autograd.grad(loss / len(batches), plist)
        g = torch.cat([gg.flatten() for gg in grads])
        # sign flip moves the weight by -sign(q)*fm*sc ; first-order loss change = g * delta_w
        delta = torch.where(q_cur >= 0, -fm_all, fm_all) * sc_all
        gain = g * delta
        gain[prot] = -float("inf"); gain[flipped] = -float("inf")
        cand = torch.topk(gain, a.topk).indices
        # 2. evaluate the real loss of each candidate, commit the worst
        best_i, best_l = None, -1.0
        for i in cand.tolist():
            qf = flip_code(q_cur, i); write(qf); l = attack_loss(model, batches)
            if l > best_l: best_l, best_i = l, i
        q_cur = flip_code(q_cur, best_i); flipped[best_i] = True; write(q_cur)
        if step in flips_list:
            acc = S.evaluate(model, vl, dev); results[step] = acc
            print(f"   [{lab}] {a.protect:5s} flips={step:4d}  loss={best_l:.3f}  acc={acc:6.2f}%  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'model':14s}{'protect':9s}{'ECC cov':>11s}{'bits/w':>8s}" + "".join(f"{'n='+str(n):>9s}" for n in flips_list))
    print(f"{lab:14s}{a.protect:9s}{cov:10.2f}%{avg_bits:8.3f}" + "".join(f"{results[n]:9.2f}" for n in flips_list))
    os.makedirs("sensreg_results/pbs_mp", exist_ok=True)
    suffix = "_" + str(a.cov) if a.protect == "selcode" else ""
    with open(f"sensreg_results/pbs_mp/{a.dataset}_{a.arch}_{lab}_{a.protect}{suffix}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "protect", "ecc_cov_pct", "avg_bits", "clean"] + [f"flips_{n}" for n in flips_list])
        w.writerow([lab, a.protect, f"{cov:.2f}", f"{avg_bits:.3f}", f"{clean:.2f}"] + [f"{results[n]:.2f}" for n in flips_list])


if __name__ == "__main__":
    main()
