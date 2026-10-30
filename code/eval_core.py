#!/usr/bin/env python3
# =============================================================================
#  eval_core.py   ONE complete the selective code evaluation script.
#  Developed By ANON, ANON, ECE Department
#
#  Protection modes (controller):
#    --protect-mode layer   : protect whole sensitive LAYERS (paper method)
#    --protect-mode weight  : protect sensitive individual WEIGHTS (per-weight)
#         with weight selection either by BINS or by top-fraction:
#           --bin-select 1  : HIGH/MID/LOW bins, controlled by
#                             --take-high --take-mid --take-low
#           --bin-select 0  : plain top --weight-frac of weights
#
#  What it runs (choose with --run):
#    --run ber    : random BER sweep, protected vs unprotected (BCH+STDM)
#    --run pbs    : PBS untargeted attack, protected vs unprotected
#    --run tbfa   : T-BFA targeted attack over source->target pairs, protected
#                   vs unprotected; for CIFAR-10 --all-pairs makes the 10x10
#                   Figure-3-style heatmaps (acc, flips, ASR) with numbers in
#                   the boxes, and tracks WHICH weights each attack flipped.
#    --run all    : ber + pbs + tbfa   (default)
#    (comma-combos allowed, e.g. --run ber,tbfa)
#
#  Everything is BCH + STDM at zero added memory.
#
#  EXAMPLES
#   # CIFAR-10, weight mode by bins, all three, all 90 T-BFA pairs:
#   python eval_core.py --dataset CIFAR10 --arch resnet18 --bits 8 \
#     --protect-mode weight --bin-select 1 --take-high 1.0 --take-mid 1.0 \
#     --plane-bits 4 --bers 0,0.03,0.06 --n-flips 50 --pbs-flips 20 \
#     --all-pairs --run all
#
#   # only T-BFA:
#   python eval_core.py ... --run tbfa
#   # only BER:
#   python eval_core.py ... --run ber
#
#   # ImageNet, a few classes:
#   python eval_core.py --dataset IMAGENET --arch resnet50 --use-pretrained 1 \
#     --imagenet-root ./imagenet-val --bits 8 --protect-mode weight \
#     --bin-select 1 --take-high 1.0 --take-mid 1.0 --plane-bits 4 \
#     --classes 3,5,100,200 --run all
# =============================================================================
from __future__ import annotations
import os, sys, argparse, csv
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecc_protect as EP
import ecc_engine as ENG
import pbs_core as ATK


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def build_handles(model, qinfo):
    h = {}
    for mn, m in model.named_modules():
        wn = f"{mn}.weight" if mn else "weight"
        if wn in qinfo and hasattr(m, "weight"):
            h[wn] = m
    return h


@torch.no_grad()
def accuracy(model, loader, device, exclude_class=-1, mb=50):
    model.eval(); n = ok = 0
    for i, (x, y) in enumerate(loader):
        if mb and i >= mb:
            break
        x, y = x.to(device), y.to(device)
        if exclude_class >= 0:
            m = (y != exclude_class)
            if not m.any():
                continue
            x, y = x[m], y[m]
        ok += int((model(x).argmax(1) == y).sum()); n += int(y.numel())
    return 100.0 * ok / max(n, 1)


@torch.no_grad()
def source_pred_hist(model, loader, device, source, nc, mb=0):
    model.eval(); h = np.zeros(nc, dtype=np.int64)
    for i, (x, y) in enumerate(loader):
        if mb and i >= mb:
            break
        x, y = x.to(device), y.to(device)
        m = (y == source)
        if m.any():
            for c in model(x[m]).argmax(1).cpu().numpy():
                h[c] += 1
    return h


# ---------------------------------------------------------------------------
# weight-selection strategies (per-weight)
# ---------------------------------------------------------------------------
def select_weights_topfrac(wscore, qinfo, frac):
    allv = np.concatenate([wscore[n].reshape(-1) for n in wscore])
    cut = float(np.quantile(allv, 1.0 - frac))
    idx, n_sel = {}, 0
    for n in wscore:
        m = wscore[n].reshape(-1) >= cut
        if m.any():
            idx[n] = np.where(m)[0]; n_sel += int(m.sum())
    return idx, n_sel


def select_weights_bins(wscore, qinfo, high_frac, mid_frac,
                        take_high, take_mid, take_low, seed=0):
    rng = np.random.default_rng(seed)
    allv = np.concatenate([wscore[n].reshape(-1) for n in wscore])
    hi_cut = float(np.quantile(allv, 1.0 - high_frac))
    mid_cut = float(np.quantile(allv, 1.0 - high_frac - mid_frac))
    idx, n_sel, nb = {}, 0, {"high": 0, "mid": 0, "low": 0}
    for n in wscore:
        v = wscore[n].reshape(-1)
        masks = [(v >= hi_cut, take_high, "high"),
                 ((v < hi_cut) & (v >= mid_cut), take_mid, "mid"),
                 (v < mid_cut, take_low, "low")]
        chosen = np.zeros(v.size, bool)
        for m, take, key in masks:
            ids = np.where(m)[0]
            if len(ids) == 0 or take <= 0:
                continue
            k = int(round(len(ids) * min(take, 1.0)))
            if k <= 0:
                continue
            pick = ids if take >= 1.0 else rng.choice(ids, size=k, replace=False)
            chosen[pick] = True; nb[key] += len(pick)
        if chosen.any():
            idx[n] = np.where(chosen)[0]; n_sel += int(chosen.sum())
    return idx, n_sel, nb


# ---------------------------------------------------------------------------
# per-weight protect + BCH-correct after external corruption/attack
# ---------------------------------------------------------------------------
def protect_correct_weights(model, qinfo, code, dev, sel_idx, score,
                            attacked_ints, clean_ints):
    K = getattr(code, "plane_bits", 4)
    B = getattr(code, "host_bits", 4)
    Lf = getattr(code, "stdm_L", 1) if getattr(code, "embed_mode", "qim") == "stdm" else 1
    _to = EP._to_ubytes; _fr = EP._from_ubytes; dq = EP.dequant_to_float
    order = sorted(sel_idx.keys())
    gathered, layout, pos = [], [], 0
    for name in order:
        ids = sel_idx[name]
        gathered.append(_to(clean_ints[name]).reshape(-1)[ids])
        layout.append((name, ids, pos, len(ids))); pos += len(ids)
    if not gathered:
        return
    gathered = np.concatenate(gathered).astype(np.uint8)
    ecc, n0, G = code.planes_encode(gathered, K, W=8)
    payload = ecc.astype(np.uint8); total = len(payload)
    host = {k: dq(clean_ints[k], qinfo[k][1], qinfo[k][2]).astype(np.float32) for k in qinfo}
    carriers = sorted(qinfo.keys(), key=lambda n: score.get(n, 0))
    seg, ptr = [], 0
    for c in carriers:
        if ptr >= total:
            break
        cap = (int(qinfo[c][0].size) // Lf) * B
        take = min(cap, total - ptr)
        if take <= 0:
            continue
        seg.append((c, ptr, take)); ptr += take
    overflow = ptr < total
    if getattr(code, "_report_storage", False):
        model_bits = sum(int(qinfo[k][0].size) for k in qinfo) * 8
        n_carriers = len(seg)
        bpw = total / max(sum(len(v) for v in sel_idx.values()), 1)
        print(f"   [storage] WEIGHT mode via STDM ({getattr(code,'embed_mode','stdm')})")
        print(f"   [storage] protected weights = {sum(len(v) for v in sel_idx.values()):,}")
        print(f"   [storage] parity = {total:,} bits hidden in {n_carriers} carrier layers")
        print(f"   [storage] {total/model_bits*100:.2f}% of model bits, "
              f"{bpw:.2f} parity bits per protected weight")
        print(f"   [storage] overflow = {'YES' if overflow else 'no'}  |  "
              f"additional memory allocated: 0", flush=True)
        code._report_storage = False   # print once
    assert ptr >= total, f"weight-mode overflow: need {total}, fit {ptr}"
    for (c, st, tk) in seg:
        host[c], _ = code.embed_store(host[c], payload[st:st + tk], B)
    reloaded = np.zeros(total, np.uint8)
    for (c, st, tk) in seg:
        reloaded[st:st + tk] = code.embed_load(host[c], tk, B)[:tk]
    gathered_att = np.concatenate(
        [_to(attacked_ints[nm]).reshape(-1)[ids] for (nm, ids, _, _) in layout]).astype(np.uint8)
    fixed = code.planes_decode(gathered_att, reloaded, K, n0, G, W=8)
    rec = {name: _to(attacked_ints[name]).reshape(-1).copy() for name in qinfo}
    for (nm, ids, st, cnt) in layout:
        rec[nm][ids] = fixed[st:st + cnt]
    rec = {name: _fr(rec[name]).reshape(qinfo[name][2]) for name in qinfo}
    EP.apply_int_to_model(model, rec, qinfo, dev)


def protect_correct_layers(model, qinfo, code, dev, protected, score,
                           attacked_ints, clean_ints):
    K = getattr(code, "plane_bits", 4)
    B = getattr(code, "host_bits", 4)
    R = getattr(code, "parity_rep", 1)
    Lf = getattr(code, "stdm_L", 1) if getattr(code, "embed_mode", "qim") == "stdm" else 1
    _to = EP._to_ubytes; _fr = EP._from_ubytes; dq = EP.dequant_to_float
    host = {k: dq(clean_ints[k], qinfo[k][1], qinfo[k][2]).astype(np.float32) for k in qinfo}
    parity_all = []
    for name in sorted(protected, key=lambda n: -score.get(n, 0)):
        ecc, n0, G = code.planes_encode(_to(clean_ints[name]), K, W=8)
        parity_all.append((name, ecc, n0, G))
    span, chunks, pos = {}, [], 0
    for (name, ecc, n0, G) in parity_all:
        rep = np.tile(ecc.astype(np.uint8), R)
        span[name] = (pos, len(ecc), R, n0, G); chunks.append(rep); pos += len(rep)
    payload = np.concatenate(chunks).astype(np.uint8) if chunks else np.zeros(0, np.uint8)
    total = len(payload)
    carriers = sorted(qinfo.keys(), key=lambda n: score.get(n, 0))
    seg, ptr = [], 0
    for c in carriers:
        if ptr >= total:
            break
        cap = (int(qinfo[c][0].size) // Lf) * B
        take = min(cap, total - ptr)
        if take <= 0:
            continue
        seg.append((c, ptr, take)); ptr += take
    overflow = ptr < total
    if getattr(code, "_report_storage", False):
        model_bits = sum(int(qinfo[k][0].size) for k in qinfo) * 8
        prot_w = sum(int(qinfo[n][0].size) for n in protected)
        insens = sum(1 for (c, _, _) in seg if c not in protected)
        ovf = len(seg) - insens
        print(f"   [storage] LAYER mode via STDM ({getattr(code,'embed_mode','stdm')})")
        print(f"   [storage] protecting {len(protected)} layers ({prot_w:,} weights)")
        print(f"   [storage] parity = {total:,} bits hidden in {len(seg)} carriers "
              f"({insens} insensitive + {ovf} sensitive-overflow)")
        print(f"   [storage] {total/model_bits*100:.2f}% of model bits, "
              f"{total/max(prot_w,1):.2f} parity bits per protected weight")
        print(f"   [storage] overflow = {'YES' if overflow else 'no'}  |  "
              f"additional memory allocated: 0", flush=True)
        code._report_storage = False
    assert ptr >= total, "layer-mode overflow"
    for (c, st, tk) in seg:
        host[c], _ = code.embed_store(host[c], payload[st:st + tk], B)
    reloaded = np.zeros(total, np.uint8)
    for (c, st, tk) in seg:
        reloaded[st:st + tk] = code.embed_load(host[c], tk, B)[:tk]
    rec = {}
    for (name, ecc, n0, G) in parity_all:
        stp, plen, Rl, _n0, _G = span[name]
        block = reloaded[stp:stp + plen * Rl].reshape(Rl, plen)
        voted = (block.sum(0) >= (Rl // 2 + 1)).astype(np.uint8) if Rl > 1 else block[0]
        rec[name] = _fr(code.planes_decode(_to(attacked_ints[name]), voted, K, n0, G, W=8)
                        ).reshape(qinfo[name][2])
    for name in qinfo:
        if name not in rec:
            rec[name] = _fr(_to(attacked_ints[name])).reshape(qinfo[name][2])
    EP.apply_int_to_model(model, rec, qinfo, dev)


def random_bit_errors(clean_ints, qinfo, target_names, ber, rng, plane_bits=4,
                      sel_idx=None, all_planes=True):
    """Return attacked_ints = clean with random bit flips. all_planes=True flips
    any of the 8 bits (realistic memory-fault model); all_planes=False restricts
    to the top plane_bits planes. In weight mode (sel_idx given) only the SELECTED
    weight indices are corrupted, so everything corrupted is also correctable."""
    _to = EP._to_ubytes; _fr = EP._from_ubytes
    att = {k: clean_ints[k].copy() for k in qinfo}
    lo_bit = 0 if all_planes else (8 - plane_bits)
    for name in target_names:
        b = _to(clean_ints[name]).reshape(-1).copy()
        if sel_idx is not None:
            idxs = sel_idx.get(name, None)
            if idxs is None or len(idxs) == 0:
                continue
            pool = np.asarray(idxs)          # only the protected weight indices
        else:
            pool = np.arange(b.size)         # whole layer
        n_bits_each = 8 if all_planes else plane_bits
        nflip = int(len(pool) * n_bits_each * ber)
        if nflip > 0:
            pick = pool[rng.integers(0, len(pool), size=nflip)]
            bp = rng.integers(lo_bit, 8, size=nflip)
            for wpos, bit in zip(pick, bp):
                b[wpos] ^= np.uint8(1 << bit)
        att[name] = _fr(b).reshape(qinfo[name][2])
    return att


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("the selective code complete evaluation")
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
    # protection controller
    ap.add_argument("--protect-mode", choices=["layer", "weight"], default="weight")
    ap.add_argument("--sens-frac", type=float, default=0.20, help="layer-mode coverage cap")
    ap.add_argument("--bin-select", type=int, default=1, help="weight mode: 1=bins, 0=top-frac")
    ap.add_argument("--weight-frac", type=float, default=0.10, help="top-frac (bin-select 0)")
    ap.add_argument("--high-frac", type=float, default=0.01)
    ap.add_argument("--mid-frac", type=float, default=0.09)
    ap.add_argument("--take-high", type=float, default=1.0)
    ap.add_argument("--take-mid", type=float, default=1.0)
    ap.add_argument("--take-low", type=float, default=0.0)
    # ecc
    ap.add_argument("--plane-bits", type=int, default=4)
    ap.add_argument("--strong-t", type=int, default=12)
    ap.add_argument("--bch-m", type=int, default=7)
    ap.add_argument("--host-bits", type=int, default=5)
    # what to run
    ap.add_argument("--run", default="all", help="all | ber | pbs | tbfa (comma-combos ok)")
    ap.add_argument("--bers", default="0,0.03,0.06")
    ap.add_argument("--n-flips", type=int, default=50, help="T-BFA max flips")
    ap.add_argument("--pbs-flips", type=int, default=20)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--attack-batches", type=int, default=2)
    ap.add_argument("--target-asr", type=float, default=99.0)
    # T-BFA pairs
    ap.add_argument("--pairs", default="3-5", help="src-tgt list, e.g. 3-5,10-20")
    ap.add_argument("--all-pairs", action="store_true", help="every src->tgt combo")
    ap.add_argument("--classes", default="", help="limit classes, e.g. 3,5,10")
    ap.add_argument("--cm-frac", type=float, default=-1, help="unused placeholder")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dev = torch.device(args.device)
    do = set(args.run.lower().replace(" ", "").split(","))
    run_ber = "all" in do or "ber" in do
    run_pbs = "all" in do or "pbs" in do
    run_tbfa = "all" in do or "tbfa" in do
    bers = [float(b) for b in args.bers.split(",") if b.strip() != ""]

    nc = 1000 if args.dataset.upper() == "IMAGENET" else (100 if args.dataset.upper() == "CIFAR100" else 10)
    model, qinfo = EP.load_model_and_qinfo(args, nc, dev)
    model.eval()
    handles = build_handles(model, qinfo)
    TQ = EP.TQ
    _, te, _nc, _ = TQ.get_dataloaders(args.dataset, args.data_root, args.batch_size,
                                       args.workers, dist_mode=False,
                                       imagenet_root=args.imagenet_root or None)
    tot = sum(v[0].size for v in qinfo.values())
    clean_ints = {k: qinfo[k][0].copy() for k in qinfo}
    # save layer sizes (total weights per layer) for layer-matched baseline analysis
    with open(f"selcode_{args.dataset.lower()}_{args.arch.lower()}_layer_sizes.csv",
              "w", newline="") as _lf:
        _w = csv.writer(_lf); _w.writerow(["layer", "num_weights"])
        for _k in qinfo:
            _w.writerow([_k, int(qinfo[_k][0].size)])
    clean = accuracy(model, te, dev)
    print(f"[base] clean acc = {clean:.2f}%  ({args.dataset}/{args.arch})", flush=True)

    code = ENG.XLayerECC(strong_t=args.strong_t, light_t=2, bch_m=args.bch_m,
                         delta=0.02, host_key=12345)
    code.host_bits = args.host_bits; code.plane_bits = args.plane_bits
    code.bit_width = args.bits; code.parity_rep = 1
    code.embed_mode = "stdm"; code.stdm_L = 1; code.store_mode = "xlayer"; code.ber_hosts = False
    code._report_storage = True   # print the [storage] block once

    # per-weight sensitivity score
    batches = ATK.collect_batches(te, dev, n_batches=args.attack_batches)
    wscore = ATK.weight_sensitivity(model, qinfo, handles, batches, bits=args.bits, device=dev)
    score = {n: float(np.sum(wscore[n])) for n in wscore}

    # ---- choose protected set ----
    sel_idx = None; protected = None
    if args.protect_mode == "weight":
        if args.bin_select:
            sel_idx, n_sel, nb = select_weights_bins(
                wscore, qinfo, args.high_frac, args.mid_frac,
                args.take_high, args.take_mid, args.take_low, seed=args.seed)
            print(f"[protect] WEIGHT/bins: {n_sel:,} weights ({100*n_sel/tot:.2f}%) "
                  f"| high:{nb['high']:,} mid:{nb['mid']:,} low:{nb['low']:,}", flush=True)
        else:
            sel_idx, n_sel = select_weights_topfrac(wscore, qinfo, args.weight_frac)
            print(f"[protect] WEIGHT/top-{args.weight_frac*100:.0f}%: {n_sel:,} weights "
                  f"({100*n_sel/tot:.2f}%)", flush=True)
        target_names = list(sel_idx.keys())
        # save the protected weight set so analysis can check inside/outside
        with open(f"selcode_{args.dataset.lower()}_{args.arch.lower()}_protected_weights.csv",
                  "w", newline="") as _pf:
            _w = csv.writer(_pf); _w.writerow(["layer", "index"])
            for _lname, _ids in sel_idx.items():
                for _ix in _ids:
                    _w.writerow([_lname, int(_ix)])
    else:
        order = sorted(qinfo, key=lambda n: -score[n])
        protected, cov = set(), 0
        for n in order:
            if cov >= args.sens_frac * tot and protected:
                break
            protected.add(n); cov += qinfo[n][0].size
        print(f"[protect] LAYER: {len(protected)} layers ({100*cov/tot:.1f}%)", flush=True)
        target_names = list(protected)

    def correct(attacked):
        if args.protect_mode == "weight":
            protect_correct_weights(model, qinfo, code, dev, sel_idx, score, attacked, clean_ints)
        else:
            protect_correct_layers(model, qinfo, code, dev, protected, score, attacked, clean_ints)

    # =====================================================================
    # BER sweep
    # =====================================================================
    if run_ber:
        print("\n[BER] random-fault sweep (protected vs unprotected)", flush=True)
        for i, ber in enumerate(bers):
            rng = np.random.default_rng(args.seed * 100003 + i)
            att = random_bit_errors(clean_ints, qinfo, target_names, ber, rng,
                                    plane_bits=args.plane_bits, sel_idx=sel_idx)
            correct(att); p = accuracy(model, te, dev)
            EP.apply_int_to_model(model, clean_ints, qinfo, dev)
            EP.apply_int_to_model(model, att, qinfo, dev); u = accuracy(model, te, dev)
            EP.apply_int_to_model(model, clean_ints, qinfo, dev)
            print(f"   BER={ber*100:5.2f}%  protected={p:6.2f}%  unprotected={u:6.2f}%", flush=True)

    # =====================================================================
    # PBS
    # =====================================================================
    if run_pbs:
        print("\n[PBS] untargeted attack (protected vs unprotected)", flush=True)
        EP.apply_int_to_model(model, clean_ints, qinfo, dev)
        p1 = ATK.PBSAttack(model, qinfo, handles, K=args.plane_bits, bits=args.bits,
                           topk=args.topk, device=dev, all_planes=True)
        b = ATK.collect_batches(te, dev, n_batches=args.attack_batches)
        qp = p1.run(b, list(qinfo.keys()), n_flips=args.pbs_flips, verbose=False)
        p1._write_back(qp); pbs_un = accuracy(model, te, dev)
        EP.apply_int_to_model(model, clean_ints, qinfo, dev)
        p2 = ATK.PBSAttack(model, qinfo, handles, K=args.plane_bits, bits=args.bits,
                           topk=args.topk, device=dev, all_planes=False)
        b = ATK.collect_batches(te, dev, n_batches=args.attack_batches)
        qp2 = p2.run(b, ATK.pbs_targets(score, target_names, band="top") or target_names,
                     n_flips=args.pbs_flips, verbose=False)
        correct(qp2); pbs_pr = accuracy(model, te, dev)
        EP.apply_int_to_model(model, clean_ints, qinfo, dev)
        eff_ber = 100.0 * args.pbs_flips / (tot * 8)
        print(f"   PBS: {args.pbs_flips} flips (effBER {eff_ber:.5f}%) | "
              f"unprotected acc={pbs_un:.2f}%  protected acc={pbs_pr:.2f}%", flush=True)

    # =====================================================================
    # T-BFA (targeted) over pairs
    # =====================================================================
    if run_tbfa:
        classes = [int(c) for c in args.classes.split(",")] if args.classes else list(range(nc))
        if args.all_pairs:
            pairs = [(s, t) for s in classes for t in classes if s != t]
        else:
            pairs = [tuple(int(z) for z in p.split("-")) for p in args.pairs.split(",")]
        print(f"\n[T-BFA] {len(pairs)} source->target pairs", flush=True)

        C = len(classes); cidx = {c: i for i, c in enumerate(classes)}
        NA = np.full((C, C), np.nan); NF = np.full((C, C), np.nan); NASR = np.full((C, C), np.nan)
        PA = np.full((C, C), np.nan); PASR = np.full((C, C), np.nan)
        flip_log = []
        tl = ATK.pbs_targets(score, target_names, band="top") or target_names

        for (s, t) in pairs:
            src, oth = ATK.split_class_batches(te, dev, s)
            # unprotected
            EP.apply_int_to_model(model, clean_ints, qinfo, dev)
            a = ATK.TBFAAttack(model, qinfo, handles, K=args.plane_bits, bits=args.bits,
                               topk=args.topk, device=dev, all_planes=True, source=s, target=t)
            qa = a.run_targeted(src, oth, list(qinfo.keys()), n_flips=args.n_flips,
                                verbose=False, target_asr=args.target_asr, asr_loader=te, device=dev)
            a._write_back(qa)
            au = accuracy(model, te, dev, exclude_class=s)
            fu = getattr(a, "flips_used", args.n_flips)
            ru = ATK.targeted_rates(model, te, dev, s, t)["asr"]
            for e in a.log:
                flip_log.append((s, t, e.get("layer"), int(e.get("index", -1)), int(e.get("plane", -1))))
            EP.apply_int_to_model(model, clean_ints, qinfo, dev)
            # protected
            a2 = ATK.TBFAAttack(model, qinfo, handles, K=args.plane_bits, bits=args.bits,
                                topk=args.topk, device=dev, all_planes=False, source=s, target=t)
            qa2 = a2.run_targeted(src, oth, tl, n_flips=args.n_flips, verbose=False,
                                  target_asr=args.target_asr, asr_loader=te, device=dev)
            correct(qa2)
            ap_ = accuracy(model, te, dev, exclude_class=s)
            rp = ATK.targeted_rates(model, te, dev, s, t)["asr"]
            EP.apply_int_to_model(model, clean_ints, qinfo, dev)
            if s in cidx and t in cidx:
                i, j = cidx[s], cidx[t]
                NA[i, j] = au; NF[i, j] = fu; NASR[i, j] = ru; PA[i, j] = ap_; PASR[i, j] = rp
            eff_ber = 100.0 * fu / (tot * 8)
            print(f"  {s}->{t}: flips={fu:.0f} (effBER {eff_ber:.5f}%) | "
                  f"UNPROT acc={au:.1f} ASR={ru:.1f} | PROT acc={ap_:.1f} ASR={rp:.1f}",
                  flush=True)

        tag = f"{args.dataset.lower()}_{args.arch.lower()}"
        # save matrices
        for nm, M in [("unprot_acc", NA), ("unprot_asr", NASR), ("unprot_flips", NF),
                      ("prot_acc", PA), ("prot_asr", PASR)]:
            with open(f"selcode_{tag}_{nm}.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow([""] + [f"t{c}" for c in classes])
                for i, s in enumerate(classes):
                    w.writerow([f"s{s}"] + ["" if np.isnan(M[i, j]) else f"{M[i,j]:.2f}" for j in range(C)])
        # flip tracking + overlap
        with open(f"selcode_{tag}_flips_detail.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["src", "tgt", "layer", "index", "plane"])
            for r in flip_log:
                w.writerow(r)
        # --- fraction of attacked weights that fall INSIDE the protected set ---
        # (proves the attacks hit the region the selective code protects -> attack-agnostic)
        if args.protect_mode == "weight":
            prot_lookup = {name: set(ids.tolist()) for name, ids in sel_idx.items()}
            inside = sum(1 for (_, _, lyr, idx, _) in flip_log
                         if lyr in prot_lookup and idx in prot_lookup[lyr])
        else:
            prot_layers = set(protected)
            inside = sum(1 for (_, _, lyr, idx, _) in flip_log if lyr in prot_layers)
        frac_inside = 100.0 * inside / max(len(flip_log), 1)
        print(f"[coverage] {inside}/{len(flip_log)} attacked bit-flips "
              f"({frac_inside:.1f}%) fall INSIDE the protected set "
              f"-> attacks concentrate where the selective code protects", flush=True)
        layer_hits = Counter(r[2] for r in flip_log)
        wkey = Counter((r[2], r[3]) for r in flip_log)
        shared = [(k, c) for k, c in wkey.items() if c > 1]
        print("\n=== FLIP TRACKING ===")
        print(f"total flips: {len(flip_log)} | distinct weights: {len(wkey)} | "
              f"common (>1 attack): {len(shared)}")
        for lname, c in layer_hits.most_common(8):
            print(f"   layer {lname}: {c} flips")
        with open(f"selcode_{tag}_flip_overlap.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["layer", "index", "num_attacks"])
            for (lname, idx), c in sorted(wkey.items(), key=lambda x: -x[1]):
                w.writerow([lname, idx, c])
        # heatmap figure with numbers in boxes
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            panels = [(NASR, "Unprotected ASR (%)", "Reds"),
                      (NA, "Unprotected acc (%)", "Reds_r"),
                      (NF, "Unprotected #flips", "viridis"),
                      (PASR, "Protected ASR (%)", "Reds"),
                      (PA, "Protected acc (%)", "Greens")]
            fig, ax = plt.subplots(1, 5, figsize=(5.4 * 5, 5.2))
            for a_, (M, title, cmap) in zip(ax, panels):
                im = a_.imshow(M, cmap=cmap); a_.set_title(title, fontsize=11, fontweight="bold")
                a_.set_xlabel("target"); a_.set_ylabel("source")
                a_.set_xticks(range(C)); a_.set_yticks(range(C))
                a_.set_xticklabels(classes, fontsize=7); a_.set_yticklabels(classes, fontsize=7)
                fs = 6 if C <= 12 else 4
                for i in range(C):
                    for j in range(C):
                        if not np.isnan(M[i, j]):
                            a_.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                                    fontsize=fs, color="black")
                plt.colorbar(im, ax=a_, fraction=0.046)
            plt.suptitle(f"the selective code vs T-BFA all-pairs  [{tag}]", fontsize=13, fontweight="bold")
            plt.tight_layout(); plt.savefig(f"selcode_{tag}.png", dpi=140)
            print(f"saved -> selcode_{tag}.png")
        except Exception as e:
            print("plot skipped:", e)
        print(f"saved -> selcode_{tag}_*.csv")

    print("\n[done]")


if __name__ == "__main__":
    main()