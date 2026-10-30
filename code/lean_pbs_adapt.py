#!/usr/bin/env python3
"""
lean_pbs_adapt.py  -  E10: ATTACK-IN-THE-LOOP TIER PLACEMENT on top of pbs_four (pbs_four.py is NOT modified; same modules,
same attacker, same code).  pbs_four protects the top-1% of weights ranked on the CLEAN model.  The 500-flip cliff is a chain
the attacker builds on the attacked model, and its tipping weights are outside that clean ranking.  This script closes the
loop: run the real PBS, read the weights it actually flipped, move them into the protected set (dropping the same number of
lowest-ranked protected weights, so coverage stays exactly 1%), and repeat for --rounds rounds.  Then the final placement is
evaluated at every flip count exactly as pbs_four does.  Round 0 == pbs_four.

usage:  python -u lean_pbs_adapt.py --dataset CIFAR10 --arch resnet18 --bits 8 --covs 0.01 --flips 100,200,300,500 \
            --adapt-flips 500 --rounds 3 --label e10_lam40_s0
"""
from __future__ import annotations
import os, sys, argparse, csv
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecc_protect as EP
import ecc_engine as ENG
import pbs_core as ATK
import eval_core as SE


def _flat_idx(sel, n):
    """protected positions of one layer as a sorted int64 flat index array, whatever container eval_core uses."""
    if torch.is_tensor(sel): sel = sel.detach().cpu().numpy()
    sel = np.asarray(sel)
    if sel.dtype == bool: return np.flatnonzero(sel.reshape(-1)).astype(np.int64)
    if sel.ndim > 1 and sel.shape[-1] != 1 and sel.size == n: return np.flatnonzero(sel.reshape(-1) != 0).astype(np.int64)
    return np.unique(sel.reshape(-1).astype(np.int64))


def _same_container(orig, flat, n):
    """flat index array back into the container type eval_core returned for this layer."""
    if torch.is_tensor(orig): o = orig.detach().cpu().numpy()
    else: o = np.asarray(orig)
    if o.dtype == bool or (o.ndim > 1 and o.size == n):
        m = np.zeros(n, dtype=bool); m[flat] = True; return m.reshape(o.shape)
    out = np.sort(flat).astype(o.dtype if o.dtype.kind in "iu" else np.int64)
    return torch.as_tensor(out) if torch.is_tensor(orig) else out


def _chain_from_q(q, clean_ints):
    """flat positions of the weights the attack flipped, per layer, from the attacker's returned ints."""
    chain = {}
    if isinstance(q, dict):
        for k, v in q.items():
            if k not in clean_ints: continue
            vv = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
            if vv.size != clean_ints[k].size: continue
            d = np.flatnonzero(vv.reshape(-1) != clean_ints[k].reshape(-1))
            if d.size: chain[k] = d
    return chain


def _get_flat(model, k):
    pn = dict(model.named_parameters())
    for name in (k, k + ".weight", k.replace(".weight", "")):
        if name in pn: return pn[name].detach().cpu().numpy().reshape(-1).copy()
    return None


def _chain_from_model(model, qinfo, clean_float, atk, q):
    """fallback: write the attacked ints into the model (as pbs_four does for --protect none) and diff against the clean floats.
    The model is left in the attacked state, which is what protect_correct_weights sees in pbs_four."""
    atk._write_back(q)
    chain = {}
    for k in qinfo:
        if clean_float.get(k) is None: continue
        now = _get_flat(model, k); d = np.flatnonzero(now != clean_float[k])
        if d.size: chain[k] = d
    return chain


def main():
    ap = argparse.ArgumentParser("pbs_four with attack-in-the-loop placement")
    ap.add_argument("--dataset", default="CIFAR10"); ap.add_argument("--arch", default="resnet18"); ap.add_argument("--quant", default="ptq")
    ap.add_argument("--bits", type=int, default=8); ap.add_argument("--use-pretrained", type=int, default=0)
    ap.add_argument("--imagenet-root", default="./imagenet-val"); ap.add_argument("--data-root", default="./data")
    ap.add_argument("--batch-size", type=int, default=128); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plane-bits", type=int, default=4); ap.add_argument("--strong-t", type=int, default=12)
    ap.add_argument("--bch-m", type=int, default=7); ap.add_argument("--host-bits", type=int, default=5)
    ap.add_argument("--topk", type=int, default=50); ap.add_argument("--attack-batches", type=int, default=2)
    ap.add_argument("--flips", default="100,200,300,500"); ap.add_argument("--covs", default="0.01")
    ap.add_argument("--label", default=""); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapt-flips", type=int, default=500, help="flip budget of the attack that informs the placement")
    ap.add_argument("--rounds", type=int, default=3, help="placement rounds (0 = plain pbs_four)")
    ap.add_argument("--attacker", choices=["adaptive", "fixed"], default="adaptive",
                    help="adaptive: the attacker knows the protected set and never flips a protected weight (its gradient is masked, "
                         "so protected weights get zero score); fixed: pbs_four's attacker, blind to the code (NOT meaningful for placement rounds)")
    ap.add_argument("--eval-shuffle-seed", type=int, default=-1,
                    help="E10c: the FINAL evaluation attack draws its probe batches from a reshuffled test loader with this seed, so the "
                         "attacker's data differs from the data used for the placement rounds (-1 = same batches as pbs_four)")
    args = ap.parse_args()

    flips_list = [int(x) for x in args.flips.split(",") if x.strip()]
    covs = [float(x) for x in args.covs.split(",") if x.strip()]
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    dev = torch.device(args.device)
    nc = 1000 if args.dataset.upper() == "IMAGENET" else (100 if args.dataset.upper() == "CIFAR100" else 10)

    model, qinfo = EP.load_model_and_qinfo(args, nc, dev); model.eval()
    handles = SE.build_handles(model, qinfo); TQ = EP.TQ
    _, te, _nc, _ = TQ.get_dataloaders(args.dataset, args.data_root, args.batch_size, args.workers, dist_mode=False, imagenet_root=args.imagenet_root or None)
    tot = sum(v[0].size for v in qinfo.values())
    clean_ints = {k: qinfo[k][0].copy() for k in qinfo}
    clean = SE.accuracy(model, te, dev); tag = args.label or "model"
    print(f"[{tag}] clean acc = {clean:.2f}%   protect=selcode+adapt rounds={args.rounds} adapt_flips={args.adapt_flips} attacker={args.attacker}", flush=True)

    code = ENG.XLayerECC(strong_t=args.strong_t, light_t=2, bch_m=args.bch_m, delta=0.02, host_key=12345)
    code.host_bits = args.host_bits; code.plane_bits = args.plane_bits; code.bit_width = args.bits; code.parity_rep = 1
    code.embed_mode = "stdm"; code.stdm_L = 1; code.store_mode = "xlayer"; code.ber_hosts = False

    b = ATK.collect_batches(te, dev, n_batches=args.attack_batches)
    if args.eval_shuffle_seed >= 0:                                   # attacker's own probe data for the final evaluation
        g = torch.Generator(); g.manual_seed(args.eval_shuffle_seed)
        te_eval = torch.utils.data.DataLoader(te.dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, generator=g)
        b_eval = ATK.collect_batches(te_eval, dev, n_batches=args.attack_batches)
        print(f"   [adapt] final evaluation attacker uses reshuffled probe batches (seed {args.eval_shuffle_seed}); placement rounds use the defender's batches", flush=True)
    else:
        b_eval = b
    wscore = ATK.weight_sensitivity(model, qinfo, handles, b, bits=args.bits, device=dev)
    score = {n: float(np.sum(wscore[n])) for n in wscore}
    layers = list(qinfo.keys())

    EP.apply_int_to_model(model, clean_ints, qinfo, dev)
    clean_float = {k: _get_flat(model, k) for k in qinfo}

    # ---- adaptive attacker: gradient mask on every quantized parameter; zero gradient at protected positions means the PBS
    #      candidate ranking never proposes them (the attacker "knows" the code and spends its flips elsewhere)
    pn = dict(model.named_parameters()); masks = {}
    def param_of(k):
        for name in (k, k + ".weight", k.replace(".weight", "")):
            if name in pn: return pn[name]
        return None
    if args.attacker == "adaptive":
        for k in qinfo:
            p = param_of(k)
            if p is None: raise SystemExit(f"adaptive attacker: no parameter found for layer key {k}")
            masks[k] = torch.ones(p.numel(), device=p.device)
            p.register_hook(lambda g, m=masks[k]: g * m.view_as(g))
        print(f"   [adapt] adaptive attacker: gradient masked on {len(masks)} layers (protected weights get zero attack score)", flush=True)
    def set_mask(sel_idx):
        if not masks: return
        for k in masks: masks[k].fill_(1.0)
        for k, v in sel_idx.items():
            if k in masks: masks[k][torch.as_tensor(_flat_idx(v, qinfo[k][0].size), device=masks[k].device)] = 0.0

    def attack(sel_idx, nf, batches=None):
        # identical sequence to pbs_four (selcode branch): reset -> PBS over all layers -> correct protected weights -> accuracy
        bb = b if batches is None else batches
        set_mask(sel_idx)
        EP.apply_int_to_model(model, clean_ints, qinfo, dev)
        atk = ATK.PBSAttack(model, qinfo, handles, K=args.plane_bits, bits=args.bits, topk=args.topk, device=dev, all_planes=True)
        q = atk.run(bb, layers, n_flips=nf, verbose=False)
        chain = _chain_from_q(q, clean_ints)
        if sum(v.size for v in chain.values()) == 0: chain = _chain_from_model(model, qinfo, clean_float, atk, q)
        SE.protect_correct_weights(model, qinfo, code, dev, sel_idx, score, q, clean_ints)
        a = SE.accuracy(model, te, dev); EP.apply_int_to_model(model, clean_ints, qinfo, dev)
        return a, chain

    rows = []
    for cov in covs:
        th = min(1.0, cov / 0.01); tm = max(0.0, (cov - 0.01) / 0.09)
        sel_idx, n_sel, nb = SE.select_weights_bins(wscore, qinfo, 0.01, 0.09, th, tm, 0.0, seed=args.seed)
        covpct = 100 * n_sel / tot; lbl = f"adapt {covpct:.2f}%"
        k0 = next(iter(sel_idx)); v0 = sel_idx[k0]
        print(f"   [adapt] sel_idx container: {type(v0).__name__} dtype={getattr(v0, 'dtype', None)} shape={getattr(v0, 'shape', None)} (layer {k0}, {qinfo[k0][0].size:,} weights)", flush=True)
        if isinstance(v0, dict): raise SystemExit("sel_idx values are dicts; tell me the structure printed above")
        print(f"\n=== selcode {covpct:.2f}%  (high:{nb['high']:,} mid:{nb['mid']:,})  -> attack-in-the-loop placement ===", flush=True)
        budget = sum(_flat_idx(v, qinfo[k][0].size).size for k, v in sel_idx.items())
        if args.eval_shuffle_seed >= 0:                               # BEFORE placement: clean-ranked code vs the attacker with its own data
            for nf in flips_list:
                a0, _ = attack(sel_idx, nf, b_eval)
                print(f"   [{tag}] before placement  flips={nf:5d}  acc={a0:6.2f}%   (attacker's own probe data)", flush=True)
        for r in range(args.rounds):
            a, chain = attack(sel_idx, args.adapt_flips)
            nchain = sum(v.size for v in chain.values())
            if nchain == 0: print("   [adapt] could not read the attacker's flips; stopping adaptation", flush=True); break
            # which chain positions are already protected (wasted flips) and which are free (the damage)
            free = {}
            for k, d in chain.items():
                if k not in sel_idx: free[k] = d; continue
                p = _flat_idx(sel_idx[k], qinfo[k][0].size); free[k] = d[~np.isin(d, p)]
            nfree = sum(v.size for v in free.values())
            print(f"   [adapt] round {r}: acc={a:6.2f}%  attacker flipped {nchain} weights, {nfree} outside the code -> moving them in", flush=True)
            if masks and nfree != nchain: print(f"   [adapt] WARNING: adaptive attacker still hit {nchain - nfree} protected weights (mask not effective?)", flush=True)
            if nfree == 0: break
            # drop the same number of lowest-ranked protected weights so the coverage stays the same
            cand = []
            for k, v in sel_idx.items():
                p = _flat_idx(v, qinfo[k][0].size); s = np.asarray(wscore[k]).reshape(-1)[p]
                cand.append((k, p, s))
            allk = np.concatenate([np.full(p.size, i) for i, (_, p, _) in enumerate(cand)]); alls = np.concatenate([s for _, _, s in cand]); allp = np.concatenate([p for _, p, _ in cand])
            drop_order = np.argsort(alls)[:nfree]
            drop = {}
            for j in drop_order: drop.setdefault(cand[allk[j]][0], []).append(allp[j])
            new_sel = {}
            for k, v in sel_idx.items():
                p = _flat_idx(v, qinfo[k][0].size)
                if k in drop: p = p[~np.isin(p, np.asarray(drop[k]))]
                if k in free and free[k].size: p = np.union1d(p, free[k])
                new_sel[k] = _same_container(v, p, qinfo[k][0].size)
            for k, d in free.items():
                if k not in new_sel and d.size:
                    ref = next(iter(sel_idx.values())); new_sel[k] = _same_container(ref, d, qinfo[k][0].size)
            sel_idx = new_sel
            now = sum(_flat_idx(v, qinfo[k][0].size).size for k, v in sel_idx.items())
            print(f"   [adapt] protected set: {budget:,} -> {now:,} weights", flush=True)
        accs = []
        for nf in flips_list:
            a, _ = attack(sel_idx, nf, b_eval); accs.append(a)
            print(f"   [{tag}] {lbl:16s} flips={nf:5d}  acc={a:6.2f}%" + ("   (after placement, attacker's own probe data)" if args.eval_shuffle_seed >= 0 else ""), flush=True)
        rows.append((lbl, covpct, accs))

    print(f"\n{'model':10s}{'protection':18s}{'coverage':>9s}" + "".join(f"{'n='+str(n):>9s}" for n in flips_list))
    for lbl, covpct, accs in rows: print(f"{tag:10s}{lbl:18s}{covpct:8.2f}%" + "".join(f"{a:9.2f}" for a in accs))
    os.makedirs("sensreg_results/pbs_four", exist_ok=True)
    with open(f"sensreg_results/pbs_four/{args.dataset.lower()}_{args.arch.lower()}_{tag}_adapt.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "protection", "coverage_pct"] + [f"flips_{n}" for n in flips_list])
        for lbl, covpct, accs in rows: w.writerow([tag, lbl, f"{covpct:.2f}"] + [f"{a:.2f}" for a in accs])


if __name__ == "__main__":
    main()
