#!/usr/bin/env python3
# =============================================================================
#  pbs_four.py   Real PBS (pbs_core.PBSAttack) under four protection conditions.
#  Copy of pbs.py with two additions, nothing else changed:
#     --protect  none     : no ECC (attack the whole model)
#                selcode    : coverage taken from the sensitive bins (high first, then mid), as in pbs.py
#                uniform  : SAME coverage, but taken equally from every bin (high/mid/low), i.e. not sensitivity-aware
#     --covs     comma list of coverage fractions of ALL weights, e.g. 0.0025,0.005,0.01,0.02,0.05
#  --attack-layers all (default): the SAME PBS over all layers in every condition; ECC then corrects protected weights.
#  Run it on the baseline checkpoint and on the lambda checkpoint (swap trick) to get all five bars.
#
#  USAGE:
#   python pbs_four.py --dataset CIFAR10 --arch resnet18 --bits 8 --protect selcode --covs 0.0025,0.005,0.01,0.05 --flips 20,50,100,200
#   python pbs_four.py --dataset CIFAR10 --arch resnet18 --bits 8 --protect uniform --covs 0.0025,0.005 --flips 20,50,100,200
#   python pbs_four.py --dataset CIFAR10 --arch resnet18 --bits 8 --protect none --flips 20,50,100,200
# =============================================================================
from __future__ import annotations
import os, sys, argparse, csv
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecc_protect as EP
import ecc_engine as ENG
import pbs_core as ATK
import eval_core as SE


def main():
    ap = argparse.ArgumentParser("real PBS, four protection conditions")
    ap.add_argument("--dataset", default="CIFAR10")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--quant", default="ptq")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--use-pretrained", type=int, default=0)
    ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plane-bits", type=int, default=4)
    ap.add_argument("--strong-t", type=int, default=12)
    ap.add_argument("--bch-m", type=int, default=7)
    ap.add_argument("--host-bits", type=int, default=5)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--attack-batches", type=int, default=2)
    ap.add_argument("--flips", default="20,50,100,200")
    ap.add_argument("--protect", choices=["none", "selcode", "uniform"], default="selcode")
    ap.add_argument("--covs", default="0.0025,0.005,0.01,0.05", help="coverage fractions of all weights (ignored for --protect none)")
    ap.add_argument("--label", default="", help="tag printed on every result line (e.g. baseline / lam30)")
    ap.add_argument("--attack-layers", choices=["all", "top"], default="all",
                    help="all = PBS searches every layer in every condition (same attacker everywhere, default); top = pbs.py behaviour (protected runs attack only the top band of protected layers)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    flips_list = [int(x) for x in args.flips.split(",") if x.strip()]
    covs = [float(x) for x in args.covs.split(",") if x.strip()] if args.protect != "none" else [0.0]
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    dev = torch.device(args.device)
    nc = 1000 if args.dataset.upper() == "IMAGENET" else (100 if args.dataset.upper() == "CIFAR100" else 10)

    model, qinfo = EP.load_model_and_qinfo(args, nc, dev); model.eval()
    handles = SE.build_handles(model, qinfo); TQ = EP.TQ
    _, te, _nc, _ = TQ.get_dataloaders(args.dataset, args.data_root, args.batch_size, args.workers,
                                       dist_mode=False, imagenet_root=args.imagenet_root or None)
    tot = sum(v[0].size for v in qinfo.values())
    clean_ints = {k: qinfo[k][0].copy() for k in qinfo}
    clean = SE.accuracy(model, te, dev)
    tag = args.label or "model"
    print(f"[{tag}] clean acc = {clean:.2f}%   protect={args.protect}", flush=True)

    code = ENG.XLayerECC(strong_t=args.strong_t, light_t=2, bch_m=args.bch_m, delta=0.02, host_key=12345)
    code.host_bits = args.host_bits; code.plane_bits = args.plane_bits
    code.bit_width = args.bits; code.parity_rep = 1
    code.embed_mode = "stdm"; code.stdm_L = 1; code.store_mode = "xlayer"; code.ber_hosts = False

    b = ATK.collect_batches(te, dev, n_batches=args.attack_batches)
    wscore = ATK.weight_sensitivity(model, qinfo, handles, b, bits=args.bits, device=dev)
    score = {n: float(np.sum(wscore[n])) for n in wscore}

    rows = []
    for cov in covs:
        if args.protect == "none":
            lbl = "none"; sel_idx = None; covpct = 0.0
        else:
            if args.protect == "selcode":
                # fill the high bin first (1% of weights), then the mid bin (next 9%)
                th = min(1.0, cov / 0.01); tm = max(0.0, (cov - 0.01) / 0.09); tl = 0.0
            else:
                # uniform: same fraction from every bin => cov of all weights, not sensitivity-aware
                th = tm = tl = cov
            sel_idx, n_sel, nb = SE.select_weights_bins(wscore, qinfo, 0.01, 0.09, th, tm, tl, seed=args.seed)
            covpct = 100 * n_sel / tot; lbl = f"{args.protect} {covpct:.2f}%"
            print(f"\n=== {lbl}  (high:{nb['high']:,} mid:{nb['mid']:,} low:{nb.get('low', 0):,}) ===", flush=True)
        accs = []
        for nf in flips_list:
            EP.apply_int_to_model(model, clean_ints, qinfo, dev)
            atk = ATK.PBSAttack(model, qinfo, handles, K=args.plane_bits, bits=args.bits, topk=args.topk, device=dev, all_planes=True)
            if sel_idx is None:
                q = atk.run(b, list(qinfo.keys()), n_flips=nf, verbose=False); atk._write_back(q)
            else:
                tl_layers = list(qinfo.keys()) if args.attack_layers == "all" else (ATK.pbs_targets(score, list(sel_idx.keys()), band="top") or list(sel_idx.keys()))
                q = atk.run(b, tl_layers, n_flips=nf, verbose=False)
                SE.protect_correct_weights(model, qinfo, code, dev, sel_idx, score, q, clean_ints)
            a = SE.accuracy(model, te, dev); accs.append(a)
            EP.apply_int_to_model(model, clean_ints, qinfo, dev)
            print(f"   [{tag}] {lbl:16s} flips={nf:5d}  acc={a:6.2f}%", flush=True)
        rows.append((lbl, covpct, accs))

    print(f"\n{'model':10s}{'protection':18s}{'coverage':>9s}" + "".join(f"{'n='+str(n):>9s}" for n in flips_list))
    for lbl, covpct, accs in rows:
        print(f"{tag:10s}{lbl:18s}{covpct:8.2f}%" + "".join(f"{a:9.2f}" for a in accs))
    os.makedirs("sensreg_results/pbs_four", exist_ok=True)
    with open(f"sensreg_results/pbs_four/{args.dataset.lower()}_{args.arch.lower()}_{tag}_{args.protect}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "protection", "coverage_pct"] + [f"flips_{n}" for n in flips_list])
        for lbl, covpct, accs in rows: w.writerow([tag, lbl, f"{covpct:.2f}"] + [f"{a:.2f}" for a in accs])


if __name__ == "__main__":
    main()
