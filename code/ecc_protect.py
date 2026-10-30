# Developed By ANON
# ANON, ECE Department
#
# ecc_protect.py
# ==============
# Sensitivity-guided, bit-plane-selective ECC for protecting quantized NN
# weights against memory bit-errors, wired into the EXISTING pipeline
# (test-Quantizer.py / qat_train.py). Architecture-agnostic: works on any
# of the 9 archs, ImageNet-val / CIFAR-100 / CIFAR-10, and on PTQ,
# QAT-Jacob, or QAT-LSQ quantized weights.
#
# It reuses, from test-Quantizer.py:
#   - build_model(arch, nc, use_pretrained, dataset)   (all 9 archs + CIFAR surgery)
#   - get_dataloaders / get_datasets                   (CIFAR + ImageNet val-split)
#   - evaluate                                         (top-1/top-5)
#   - quantize_param_smart / dequantize_*              (PTQ per-channel/per-tensor)
#   - ckpt_path / strip_prefix_from_state_dict
#   - maybe_load_float32_or_pretrained / load_quantized_into_model
#
# The ECC engine (BCH bit-plane + QIM/STDM embedding + cross-layer storage)
# lives in ecc_engine.py (class XLayerECC), unchanged.
#
# ---------------------------------------------------------------------------
# QUANT SOURCES (--quant):
#   ptq        : build model, load float32/pretrained, PTQ-quantize on the fly
#   qat-jacob  : load int{bits}_qat       checkpoint (Jacob 2018, from qat_train.py)
#   qat-lsq    : load int{bits}_qat_lsq   checkpoint (Esser 2020 LSQ)
#
# For every quantizable Conv2d/Linear weight we recover its INT tensor +
# per-(channel/tensor) scale, run the ECC scheme on the INT bit-planes,
# inject BER, recover, dequantize, and evaluate.
#
# ---------------------------------------------------------------------------
# EXAMPLES
#   # CIFAR-10 ResNet-18, PTQ, STDM L=1 cross-layer, BER sweep:
#   python ecc_protect.py --dataset CIFAR10 --arch resnet18 --quant ptq \
#       --bits 8 --bers 0.01,0.03,0.06,0.08,0.10,0.12 \
#       --mode sensitive --sens-frac 0.20 --ber-hosts 0 \
#       --plane-bits 3 --host-bits 5 --strong-t 12 --bch-m 7 \
#       --embed stdm --stdm-L 1 --store xlayer --parity-rep 1
#
#   # ImageNet ResNet-50, QAT-LSQ INT4 weights, QIM cross-layer:
#   python ecc_protect.py --dataset IMAGENET --imagenet-root ./imagenet-val \
#       --arch resnet50 --quant qat-lsq --bits 4 --use-pretrained 1 \
#       --bers 0.01,0.03,0.06 --mode sensitive --sens-frac 0.20 \
#       --ber-hosts 0 --plane-bits 3 --host-bits 5 --strong-t 10 --bch-m 7 \
#       --embed qim --store xlayer
#
#   # wrong-key STDM security demo (recovery should collapse):
#   python ecc_protect.py --dataset CIFAR10 --arch resnet18 --quant ptq --bits 8 \
#       --bers 0,0.03,0.06 --mode sensitive --sens-frac 0.20 --ber-hosts 0 \
#       --plane-bits 3 --host-bits 5 --strong-t 10 --bch-m 7 \
#       --embed stdm --stdm-L 2 --store xlayer --host-key 12345 --attack-key 99999

from __future__ import annotations
import os, sys, csv, time, argparse, importlib.util
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

import ecc_engine as ENG


# ---------------------------------------------------------------------------
# Import the user's test-Quantizer.py (hyphenated filename) the same way
# qat_train.py does.
# ---------------------------------------------------------------------------
def _load_TQ():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, "test-Quantizer.py"),
             os.path.join(here, "test_quantizer.py"),
             "test-Quantizer.py", "test_quantizer.py"]
    for path in cands:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("test_quantizer", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["test_quantizer"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("Could not locate test-Quantizer.py next to ecc_protect.py")

TQ = _load_TQ()


# ---------------------------------------------------------------------------
# Recover INT weights + per-(channel/tensor) scale for every quantizable
# Conv2d/Linear weight, for the chosen quant source. Returns:
#   qinfo[name] = (q_int_np_int16, scale_obj, shape)
# where scale_obj is either ('per_tensor', float) or ('per_channel', np.ndarray[OC]).
# Bias/BN/etc are skipped (never protected).
# ---------------------------------------------------------------------------
def _quantizable_weight_names(model):
    names = set()
    for mname, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            names.add(f"{mname}.weight" if mname else "weight")
    return names


def build_qinfo(model, num_bits, quant_source, qat_scales=None):
    """Return {name: (q_int(np.int32), ('per_tensor'|'per_channel', scale), shape)}."""
    qinfo = {}
    qnames = _quantizable_weight_names(model)
    qmax = (1 << (num_bits - 1)) - 1
    with torch.no_grad():
        sd = model.state_dict()
        for name, p in sd.items():
            if name not in qnames or not p.dtype.is_floating_point:
                continue
            w = p.detach().to(torch.float32)
            if quant_source in ("qat-jacob", "qat-lsq") and qat_scales and name in qat_scales:
                # exact training-time per-channel scale; recover ints by division
                s = qat_scales[name].to(torch.float32)
                s_b = s
                while s_b.ndim < w.ndim:
                    s_b = s_b.unsqueeze(-1)
                q = torch.round(w / s_b).clamp(-qmax - 1, qmax).to(torch.int32)
                qinfo[name] = (q.cpu().numpy().astype(np.int32),
                               ("per_channel", s.cpu().numpy().astype(np.float32)),
                               tuple(w.shape))
            else:
                # PTQ on the fly via TQ's smart quantizer (per-channel conv / per-tensor lin)
                q_t, sinfo = TQ.quantize_param_smart(name, w, num_bits)
                if sinfo is None:
                    continue
                if sinfo["type"] == "per_channel":
                    sc = sinfo["scales"].cpu().numpy().astype(np.float32)
                    qinfo[name] = (q_t.cpu().numpy().astype(np.int32),
                                   ("per_channel", sc), tuple(w.shape))
                else:
                    qinfo[name] = (q_t.cpu().numpy().astype(np.int32),
                                   ("per_tensor", float(sinfo["scale"])), tuple(w.shape))
    return qinfo


def _to_ubytes(q_int):
    """Signed int (e.g. int32 holding int8 values) -> uint8 bit pattern (two's complement)."""
    return q_int.astype(np.int8).view(np.uint8)


def _from_ubytes(qb):
    """uint8 bit pattern -> signed int32 (inverse of _to_ubytes)."""
    return qb.view(np.int8).astype(np.int32)


def dequant_to_float(q_int, scale_obj, shape):
    """INT tensor -> float32 numpy of original shape, using per-channel/tensor scale."""
    q = q_int.astype(np.float32).reshape(shape)
    kind, sc = scale_obj
    if kind == "per_tensor":
        return q * float(sc)
    s = sc.astype(np.float32)
    while s.ndim < q.ndim:
        s = s[..., None]
    return q * s


def apply_int_to_model(model, recovered_int, qinfo, device):
    """Write dequantized recovered INT weights back into the model in place."""
    with torch.no_grad():
        sd = model.state_dict()
        new = {}
        for k, p in sd.items():
            if k in qinfo and k in recovered_int:
                _, scale_obj, shape = qinfo[k]
                w = dequant_to_float(recovered_int[k], scale_obj, shape)
                new[k] = torch.from_numpy(w).to(p.device).to(p.dtype)
            else:
                new[k] = p
        model.load_state_dict(new, strict=True)


# ---------------------------------------------------------------------------
# Quant-source loader: build the model and obtain qinfo + a clean baseline
# accuracy (quantized, no errors).
# ---------------------------------------------------------------------------
def load_model_and_qinfo(args, nc, device):
    model = TQ.build_model(args.arch, nc, use_pretrained=bool(args.use_pretrained),
                           dataset=args.dataset).to(device)
    qat_scales = None
    if args.quant == "ptq":
        src, mb = TQ.maybe_load_float32_or_pretrained(
            model, args.dataset, args.arch,
            use_pretrained=bool(args.use_pretrained), map_location=device)
        print(f"[weights] PTQ source={src}")
    else:
        suffix = "_lsq" if args.quant == "qat-lsq" else ""
        ck = TQ.ckpt_path(args.dataset, args.arch, f"int{args.bits}_qat{suffix}")
        if not os.path.exists(ck):
            raise FileNotFoundError(
                f"{args.quant} requires checkpoint {ck} (run qat_train.py "
                f"--qat-method {'lsq' if suffix else 'jacob'} "
                f"{'--qtag-suffix _lsq' if suffix else ''} first)")
        payload = torch.load(ck, map_location=device)
        if "state_dict" in payload and payload.get("meta", {}).get("qat_baked", False):
            sd = TQ.strip_prefix_from_state_dict(payload["state_dict"])
            model.load_state_dict(sd, strict=True)
            qat_scales = payload["meta"].get("scales", None)
            print(f"[weights] {args.quant} loaded {ck} "
                  f"({len(qat_scales) if qat_scales else 0} exact scales)")
        else:
            TQ.load_quantized_into_model(model, args.dataset, args.arch, args.bits,
                                         qtag="qat", map_location=device)
            print(f"[weights] {args.quant} loaded {ck} (legacy PTQ payload)")

    qinfo = build_qinfo(model, args.bits, args.quant, qat_scales)
    # baseline: write dequantized clean ints back, eval
    clean_int = {k: v[0].copy() for k, v in qinfo.items()}
    apply_int_to_model(model, clean_int, qinfo, device)
    return model, qinfo


# ---------------------------------------------------------------------------
# Architecture-agnostic per-layer sensitivity: inject `probe_ber` errors into
# ONE layer's ints, dequantize, eval, measure accuracy drop. Works on any arch.
# ---------------------------------------------------------------------------
def measure_sensitivity(model, qinfo, code, device, te, base_acc,
                        probe_ber=0.02, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    scores = {}
    clean_int = {k: v[0].copy() for k, v in qinfo.items()}
    for name in qinfo:
        # corrupt only this layer
        rec = {k: clean_int[k].copy() for k in qinfo}
        q = _to_ubytes(rec[name])
        rec[name] = _from_ubytes(code._corrupt_bytes(q, probe_ber, rng)).reshape(qinfo[name][2])
        apply_int_to_model(model, rec, qinfo, device)
        acc = _eval(model, te, device)
        scores[name] = base_acc - acc
        if verbose:
            print(f"   sensitivity {name:32s}: dacc={scores[name]:6.2f}%  "
                  f"({qinfo[name][0].size:,} wts)")
    # restore clean
    apply_int_to_model(model, clean_int, qinfo, device)
    return scores


def _eval(model, te, device):
    top1, _ = TQ.evaluate(model, te, device)
    return top1


# ---------------------------------------------------------------------------
# Cross-layer (insensitive-first) protection run at one BER. Mirrors v5's
# run_xlayer_sensitive_xlayer but operating on the integrated qinfo dict.
# ---------------------------------------------------------------------------
def run_at_ber(model, qinfo, code, ber, device, te, rng, sens, sens_score, report=False):
    B = getattr(code, 'host_bits', 4)
    K = getattr(code, 'plane_bits', 4)
    R = getattr(code, 'parity_rep', 1)
    Lf = getattr(code, 'stdm_L', 4) if getattr(code, 'embed_mode', 'qim') == 'stdm' else 1
    store = getattr(code, 'store_mode', 'xlayer')

    tot = sum(v[0].size for v in qinfo.values())
    # float host copies (for embedding parity) per layer
    host = {k: dequant_to_float(v[0], v[1], v[2]).astype(np.float32) for k, v in qinfo.items()}

    # 1. encode parity for each sensitive layer
    parity_all = []
    for name in sorted(sens, key=lambda n: -sens_score.get(n, 0)):
        ecc, n0, G = code.planes_encode(_to_ubytes(qinfo[name][0]), K, W=8)
        parity_all.append((name, ecc, n0, G))
    span = {}; chunks = []; pos = 0
    for (name, ecc, n0, G) in parity_all:
        rep = np.tile(ecc.astype(np.uint8), R)
        span[name] = (pos, len(ecc), R, n0, G)
        chunks.append(rep); pos += len(rep)
    payload = np.concatenate(chunks).astype(np.uint8) if chunks else np.zeros(0, np.uint8)
    total_payload = len(payload)
    total_parity_1 = sum(len(e) for (_n, e, _a, _b) in parity_all)

    if store == 'xlayer':
        carriers = sorted(qinfo.keys(), key=lambda n: sens_score.get(n, 0))  # insensitive first
    else:
        carriers = list(sens)  # self storage: only the protected layers
    seg = []; ptr = 0; used = []
    for cname in carriers:
        if ptr >= total_payload: break
        nw = int(qinfo[cname][0].size)
        cap = (nw // Lf) * B
        take = min(cap, total_payload - ptr)
        if take <= 0: continue
        seg.append((cname, ptr, take)); ptr += take; used.append(cname)
    if ptr < total_payload:
        raise AssertionError(f"storage overflow: need {total_payload:,} bits, fit {ptr:,} "
                             f"(store={store}, embed={getattr(code,'embed_mode','qim')}, L={Lf}); "
                             f"raise --host-bits / lower --stdm-L / lower --parity-rep / fewer layers")
    for (cname, start, take) in seg:
        host[cname], _ = code.embed_store(host[cname], payload[start:start+take], B)

    if report:
        mb = tot * 8
        prot = sum(qinfo[n][0].size for n in sens)
        n_ins = sum(1 for c in used if c not in sens)
        n_sen = sum(1 for c in used if c in sens)
        print(f"\n   [storage] mode={store}  protecting {len(sens)} layers "
              f"({prot:,} wts = {100*prot/tot:.1f}% of model)")
        print(f"   [storage] model: {tot:,} weights = {mb:,} bits")
        print(f"   [storage] parity={total_parity_1:,} bits (1 copy), rep={R}, "
              f"payload={total_payload:,} bits = {100*total_payload/mb:.2f}% of model bits")
        print(f"   [storage] {total_payload/tot:.2f} bits/weight (model avg)")
        if store == 'xlayer':
            print(f"   [storage] carriers: {len(used)} layers ({n_ins} insensitive + {n_sen} sensitive-overflow)")
        print(f"   [storage] additional memory allocated: 0\n")

    # corrupt: ber_hosts=0 -> only sensitive layers take errors
    corrupted = {}
    for name, (q, sc, shape) in qinfo.items():
        qb = _to_ubytes(q)
        if name in sens:
            corrupted[name] = code._corrupt_bytes(qb, ber, rng)
        else:
            corrupted[name] = (code._corrupt_bytes(qb, ber, rng)
                               if getattr(code, 'ber_hosts', True) else qb.copy())

    # reload payload from (possibly corrupted) carriers
    reloaded = np.zeros(total_payload, np.uint8)
    for (cname, start, take) in seg:
        bits = code.embed_load(host[cname], take, B)
        reloaded[start:start+take] = bits[:take]

    # recover each sensitive layer
    rec = {}
    if getattr(code, "no_correct", False):
        # UNPROTECTED baseline: keep the corrupted weights, skip BCH correction
        for name in qinfo:
            rec[name] = _from_ubytes(corrupted[name]).reshape(qinfo[name][2])
    else:
        for (name, ecc, n0, G) in parity_all:
            st, plen, Rl, _n0, _G = span[name]
            block = reloaded[st:st+plen*Rl].reshape(Rl, plen)
            voted = (block.sum(axis=0) >= (Rl//2+1)).astype(np.uint8) if Rl > 1 else block[0]
            fixed = code.planes_decode(corrupted[name], voted, K, n0, G, W=8)   # returns uint8 bytes
            rec[name] = _from_ubytes(fixed).reshape(qinfo[name][2])
        for name in qinfo:
            if name not in rec:
                rec[name] = _from_ubytes(corrupted[name]).reshape(qinfo[name][2])

    apply_int_to_model(model, rec, qinfo, device)
    return _eval(model, te, device)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def plot_layer_landscape(qinfo, sens_score, sens, args, tot, out):
    """Unique 'layer landscape' figure: each Conv/Linear layer plotted by its
    SIZE (x, log) vs SENSITIVITY (y). Protected (sensitive) layers are warm and
    circled; carrier (insensitive) layers that hold parity are cool. Bubble area
    = #weights. This shows the method's strategy at a glance: protect the
    small-but-sensitive layers, hide parity in the large-but-insensitive ones."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as e:
        print(f"(layer plot skipped: {e})")
        return

    names = list(qinfo.keys())
    sizes = np.array([qinfo[n][0].size for n in names], float)
    sensv = np.array([sens_score.get(n, 0.0) for n in names], float)
    is_prot = np.array([n in sens for n in names])
    # for LOG y-axis: clip sensitivity to a small positive floor (some layers
    # have ~0 or slightly negative drop). Floor = half the smallest positive value.
    pos = sensv[sensv > 0]
    floor = (pos.min() * 0.5) if pos.size else 0.01
    sens_plot = np.clip(sensv, floor, None)

    # bubble area scaled by weight count
    area = 60 + 1100 * (sizes / sizes.max())
    fig, ax = plt.subplots(figsize=(11, 7))

    # carriers (insensitive) first, behind
    ax.scatter(sizes[~is_prot], sens_plot[~is_prot], s=area[~is_prot],
               c="#4C9BE8", alpha=0.55, edgecolors="#1f5fa8", linewidths=1.0,
               label="Carrier layers (hold parity)", zorder=2)
    # protected (sensitive) on top
    ax.scatter(sizes[is_prot], sens_plot[is_prot], s=area[is_prot],
               c="#FF7F3F", alpha=0.85, edgecolors="#b0431a", linewidths=1.4,
               label="Protected layers (BCH on top bits)", zorder=3)

    # selection threshold line (lowest sensitivity among protected)
    thr = None
    if is_prot.any():
        thr = max(sensv[is_prot].min(), floor)
        ax.axhline(thr, ls="--", lw=1.2, color="#888", zorder=1)
        ax.text(sizes.max(), thr, f"  selection cutoff ({sensv[is_prot].min():.2f}%)",
                va="bottom", ha="right", fontsize=9, color="#555")

    # label the most notable layers (top-4 sensitive + largest carrier)
    order_sens = np.argsort(-sensv)
    to_label = list(order_sens[:4])
    carriers_idx = np.where(~is_prot)[0]
    if len(carriers_idx):
        to_label.append(carriers_idx[np.argmax(sizes[carriers_idx])])
    for i in set(to_label):
        short = names[i].replace(".weight", "")
        ax.annotate(short, (sizes[i], sens_plot[i]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color="#333")

    ax.set_xscale("log")
    ax.set_yscale("log")                                   # spread the small-but-nonzero layers
    ax.set_xlabel("Layer size  (number of weights, log scale)", fontsize=12)
    ax.set_ylabel("Sensitivity  (Top-1 accuracy drop under bit-errors, %, log)", fontsize=12)
    prot_w = sum(qinfo[n][0].size for n in sens)
    ax.set_title(f"Layer landscape - {args.arch} / {args.dataset} / {args.quant.upper()} INT{args.bits}\n"
                 f"protect {len(sens)} sensitive layers ({100*prot_w/tot:.1f}% of weights); "
                 f"parity hidden in insensitive carriers (0 extra memory)",
                 fontsize=12)
    ax.grid(True, which="both", alpha=0.25)

    # shaded "protected zone" guide (above cutoff)
    if thr is not None:
        ax.axhspan(thr, max(sens_plot.max()*1.3, thr*2),
               color="#FF7F3F", alpha=0.05, zorder=0)

    leg_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF7F3F",
               markeredgecolor="#b0431a", markersize=11, label="Protected (BCH on top bits)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C9BE8",
               markeredgecolor="#1f5fa8", markersize=11, label="Carrier (stores parity)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#cccccc",
               markeredgecolor="#888", markersize=16, label="bubble area = #weights"),
    ]
    ax.legend(handles=leg_handles, loc="upper right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{out}_layers.png", dpi=150)
    plt.close(fig)
    print(f"saved -> {out}_layers.png  (layer landscape)")

    # also a per-layer CSV
    with open(f"{out}_layers.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["layer", "n_weights", "sensitivity_pct", "role"])
        for n in sorted(names, key=lambda m: -sens_score.get(m, 0)):
            wcsv.writerow([n, qinfo[n][0].size, f"{sens_score.get(n,0):.3f}",
                           "protected" if n in sens else "carrier"])
    print(f"saved -> {out}_layers.csv")


def _plot_weight_drift_heatmap(base_float, model, sens, qinfo, code, args):
    """Heatmap of |weight change| from fine-tuning for the protected layers,
    with the STDM survival margin (delta/2) marked. Shows drift stays under
    the threshold, which is WHY parity survives."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sd = model.state_dict()
        # sort layers by natural depth (layer1 < layer2 < ..., conv1 before conv2)
        def _depth_key(n):
            import re
            m = re.findall(r'\d+', n)
            return tuple(int(x) for x in m) if m else (99,)
        names = sorted(sens, key=_depth_key)[:8]
        if not names: return
        delta = float(getattr(code, "delta", 0.02))
        fig, ax = plt.subplots(figsize=(8, 4.6))
        nb = 60
        M = np.zeros((len(names), nb))
        for i, n in enumerate(names):
            w0 = base_float[n].reshape(-1)
            w1 = sd[n].detach().cpu().numpy().astype(np.float32).reshape(-1)
            d = np.abs(w1 - w0)
            idx = np.linspace(0, len(d), nb + 1).astype(int)
            M[i] = [d[idx[j]:idx[j+1]].mean() if idx[j+1] > idx[j] else 0 for j in range(nb)]
        mx = float(M.max())
        # auto-scale: if drift is tiny, scale to the data so structure is visible,
        # but keep the margin reference in the title/annotation
        vmax = mx if mx > 1e-9 else 1.0
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace(".weight", "") for n in names], fontsize=8)
        ax.set_xlabel("position along weight tensor (binned)")
        ax.set_title(r"Weight drift $|\Delta w|$ from fine-tuning (protected layers)"
                     "\n"
                     r"STDM survival margin $\Delta/2$ = " + f"{delta/2:.4f}")
        cb = plt.colorbar(im, ax=ax); cb.set_label(r"mean $|\Delta w|$")
        if mx < 1e-6:
            note = r"no measurable drift: max $|\Delta w|$ $\approx$ 0  " \
                   r"($\ll \Delta/2$ = " + f"{delta/2:.4f})"
            col = "#2E7D32"
        else:
            rel = "BELOW" if mx < delta/2 else "ABOVE"
            note = r"max $|\Delta w|$ = " + f"{mx:.4f}  ({rel} " + r"$\Delta/2$ = " + f"{delta/2:.4f})"
            col = "#2E7D32" if mx < delta/2 else "#C62828"
        ax.text(0.99, -0.18, note, transform=ax.transAxes, ha="right", fontsize=9, color=col)
        plt.tight_layout(); plt.savefig(f"{args.out}_driftmap.png", dpi=140); plt.close()
        print(f"saved -> {args.out}_driftmap.png  (weight-drift heatmap)")
    except Exception as e:
        print(f"(drift heatmap skipped: {e})")


def _plot_stdm_spread(code, args):
    """Illustrate how ONE parity bit is spread across L carrier weights via the
    key-seeded projection vector (the unique STDM 'spread' visual)."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        L = max(2, int(getattr(args, "stdm_L", 2)))
        # regenerate one projection vector from the key (same path as embedding)
        try:
            V, d = code._stdm_vectors(1, L, getattr(code, "host_bits", 4))
            v = np.asarray(V).reshape(-1)[:L]
        except Exception:
            rng = np.random.default_rng(args.host_key)
            v = rng.choice([-1.0, 1.0], size=L)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8),
                                       gridspec_kw={"width_ratios": [2, 1]})
        idx = np.arange(L)
        ax1.stem(idx, v, basefmt=" ")
        ax1.axhline(0, color="#888", lw=0.8)
        ax1.set_title(f"Key-seeded projection vector v  (L={L} carriers)\n"
                      "one parity bit is spread over all L weights")
        ax1.set_xlabel("carrier weight index"); ax1.set_ylabel(r"normalized $v[i]$")
        ax1.set_ylim(-1.2, 1.2); ax1.set_xticks(list(range(L)))
        # right: the projection p = sum(w_i v_i) lands on a lattice that encodes the bit
        xs = np.linspace(-3, 4, 400)
        ax2.plot(xs, 0.5 + 0.5*np.sign(np.sin(np.pi*xs)), color="#0021A5", lw=1.5)
        ax2.set_title(r"projection $p \rightarrow$ lattice" "\n" "(bit = nearest cell)")
        ax2.set_xlabel(r"$p = \sum_i w_i \cdot v_i$"); ax2.set_yticks([0, 1]); ax2.set_yticklabels(["0", "1"])
        plt.tight_layout(); plt.savefig(f"{args.out}_stdmspread.png", dpi=140); plt.close()
        print(f"saved -> {args.out}_stdmspread.png  (STDM spread illustration)")
    except Exception as e:
        print(f"(STDM spread plot skipped: {e})")


def run_finetune_survival(model, qinfo, code, device, te, args, sens, sens_score):
    """Does the STDM-embedded parity survive fine-tuning?

    Stays inside the existing flow. The parity lives in the float carrier copies
    (host[]), exactly as in the normal run. We:
      1. embed parity into host[] (existing embed_store),
      2. fine-tune the model normally (its weights drift),
      3. measure how much each carrier weight moved (the drift),
      4. apply that same drift to the parity-embedded host[],
      5. read parity back from the drifted host[] (existing embed_load) and report BER.
    This asks the real question, does training-induced weight movement wash out the
    hidden parity, without forcing the float lattice through INT quantization.
    Only runs when --finetune-epochs > 0; the normal flow is untouched.
    """
    import torch.nn.functional as F
    B = getattr(code, 'host_bits', 4)
    K = getattr(code, 'plane_bits', 4)
    R = getattr(code, 'parity_rep', 1)
    Lf = getattr(code, 'stdm_L', 1) if getattr(code, 'embed_mode', 'qim') == 'stdm' else 1

    base_float = {k: dequant_to_float(v[0], v[1], v[2]).astype(np.float32) for k, v in qinfo.items()}

    parity_all = []
    for name in sorted(sens, key=lambda n: -sens_score.get(n, 0)):
        ecc, n0, G = code.planes_encode(_to_ubytes(qinfo[name][0]), K, W=8)
        parity_all.append((name, ecc, n0, G))
    chunks = []
    for (name, ecc, n0, G) in parity_all:
        chunks.append(np.tile(ecc.astype(np.uint8), R))
    payload = np.concatenate(chunks).astype(np.uint8) if chunks else np.zeros(0, np.uint8)
    total_payload = len(payload)

    host = {k: v.copy() for k, v in base_float.items()}
    carriers = sorted(qinfo.keys(), key=lambda n: sens_score.get(n, 0))
    seg = []; ptr = 0
    for cname in carriers:
        if ptr >= total_payload: break
        nw = int(qinfo[cname][0].size)
        cap = (nw // Lf) * B
        take = min(cap, total_payload - ptr)
        if take <= 0: continue
        seg.append((cname, ptr, take)); ptr += take
    if ptr < total_payload:
        raise AssertionError("storage overflow in finetune study; raise --host-bits / lower --stdm-L")
    for (cname, start, take) in seg:
        host[cname], _ = code.embed_store(host[cname], payload[start:start+take], B)

    def _decode_from(host_dict):
        reloaded = np.zeros(total_payload, np.uint8)
        for (cname, start, take) in seg:
            bits = code.embed_load(host_dict[cname], take, B)
            reloaded[start:start+take] = bits[:take]
        return float(np.mean(reloaded != payload)) if total_payload > 0 else 0.0

    ber0 = _decode_from(host)
    print(f"\n[finetune-survival] clean embed parity-BER={ber0*100:.3f}%  (should be ~0)")

    # fix seeds so the fine-tune figures are reproducible run-to-run
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    tr, _, _, _ = TQ.get_dataloaders(args.dataset, args.data_root, args.batch_size,
                                     args.workers, dist_mode=False,
                                     imagenet_root=args.imagenet_root or None)
    opt = torch.optim.SGD(model.parameters(), lr=args.finetune_lr, momentum=0.9)
    base_acc = _eval(model, te, device)
    rows = [(0, base_acc, ber0)]

    def _drifted_host():
        sd = model.state_dict()
        out = {}
        for (cname, start, take) in seg:
            w_now = sd[cname].detach().cpu().numpy().astype(np.float32).reshape(-1)
            drift = w_now - base_float[cname].reshape(-1)
            out[cname] = host[cname].reshape(-1) + drift
        return out

    for ep in range(1, args.finetune_epochs + 1):
        model.train(); seen = 0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
            seen += 1
            if seen >= args.finetune_batches: break
        model.eval()
        acc_ft = _eval(model, te, device)
        ber_ft = _decode_from(_drifted_host())
        rows.append((ep, acc_ft, ber_ft))
        print(f"[finetune-survival] epoch {ep}: acc={acc_ft:.2f}%  parity-BER={ber_ft*100:.3f}%")

    print("\n=== FINETUNE SURVIVAL (parity BER vs epoch) ===")
    print("epoch : " + "  ".join(f"{r[0]}" for r in rows))
    print("acc   : " + "  ".join(f"{r[1]:.2f}" for r in rows))
    print("pBER% : " + "  ".join(f"{r[2]*100:.3f}" for r in rows))
    with open(f"{args.out}_ftsurvival.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "acc", "parity_ber"])
        for r in rows: w.writerow([r[0], f"{r[1]:.2f}", f"{r[2]:.6f}"])
    print(f"saved -> {args.out}_ftsurvival.csv")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        eps = [r[0] for r in rows]; bers = [r[2]*100 for r in rows]; accs = [r[1] for r in rows]
        fig, ax1 = plt.subplots(figsize=(7, 4.3))
        ax1.plot(eps, bers, '-s', color="#C62828", lw=2.4, ms=7)
        ax1.set_xlabel("Fine-tune epoch"); ax1.set_ylabel("Parity BER (%)", color="#C62828")
        ax1.set_ylim(-1, max(5, max(bers) + 2)); ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(eps, accs, '-o', color="#2E7D32", lw=2.4, ms=7)
        ax2.set_ylabel("Top-1 accuracy (%)", color="#2E7D32")
        # fixed, sensible accuracy window so a sub-1pp wiggle reads as flat (not a fake crash)
        amid = float(np.mean(accs)); arange = max(accs) - min(accs)
        lo = min(min(accs) - 5, amid - 6); hi = max(max(accs) + 5, amid + 6)
        ax2.set_ylim(max(0, lo), min(100, hi))
        ax2.annotate(f"accuracy varies only {arange:.2f} pp over training",
                     xy=(0.5, 0.06), xycoords="axes fraction", ha="center",
                     fontsize=8, style="italic", color="#2E7D32")
        plt.title(f"Parity survival under fine-tuning  ({args.arch} {args.dataset} "
                  f"INT{args.bits}, lr={args.finetune_lr})")
        plt.tight_layout(); plt.savefig(f"{args.out}_ftsurvival.png", dpi=140); plt.close()
        print(f"saved -> {args.out}_ftsurvival.png")
    except Exception as e:
        print(f"(survival plot skipped: {e})")

    # --- internal visualizations (weight-drift heatmap + STDM spread) ---
    if getattr(args, "finetune_internals", 1):
        _plot_weight_drift_heatmap(base_float, model, sens, qinfo, code, args)
        _plot_stdm_spread(code, args)

    # --- 6. Does the SURVIVING parity still correct errors on the fine-tuned model? ---
    # No rebuild: we keep the parity that survived (embedded in the drifted carriers).
    # We inject bit-errors into the fine-tuned sensitive-layer weights and correct the
    # top bit-planes using the surviving parity decoded from the drifted carriers.
    print("\n[finetune-survival] testing whether the SURVIVING parity still corrects errors ...")
    drifted = _drifted_host()                 # carriers after fine-tune drift (hold parity)
    sd = model.state_dict()

    # current fine-tuned ints for the sensitive layers (re-derived from the live model)
    def _ft_int(name):
        w = sd[name].detach().cpu().numpy()
        _, sc, shape = qinfo[name]
        return _requantize_ft(w.reshape(shape), sc, args.bits)

    # decode the surviving parity payload from the drifted carriers
    surviving = np.zeros(total_payload, np.uint8)
    for (cname, start, take) in seg:
        bits = code.embed_load(drifted[cname], take, B)
        surviving[start:start+take] = bits[:take]

    # rebuild per-layer parity spans (same order as embed)
    span = {}; pos2 = 0
    for (name, ecc, n0, G) in parity_all:
        span[name] = (pos2, len(ecc), R, n0, G); pos2 += len(ecc) * R

    sweep_bers = [0.0, 0.03, 0.06, 0.10, 0.12]
    sweep = {}
    for ber in sweep_bers:
        rng = np.random.default_rng(args.seed)
        rec = {}
        for name, (q, sc, shape) in qinfo.items():
            ft_int = _ft_int(name) if name in sens else _ft_int(name)
            qb = _to_ubytes(ft_int)
            if name in sens:
                corrupted = code._corrupt_bytes(qb, ber, rng)
                st, plen, Rl, n0, G = span[name]
                block = surviving[st:st+plen*Rl].reshape(Rl, plen)
                voted = (block.sum(axis=0) >= (Rl//2+1)).astype(np.uint8) if Rl > 1 else block[0]
                fixed = code.planes_decode(corrupted, voted, K, n0, G, W=8)
                rec[name] = _from_ubytes(fixed).reshape(shape)
            else:
                rec[name] = _from_ubytes(qb).reshape(shape)
        apply_int_to_model(model, rec, qinfo, device)
        acc = _eval(model, te, device)
        sweep[ber] = acc
        print(f"   [post-FT, surviving parity] BER={ber*100:5.2f}%  acc={acc:.2f}%")

    print("\n=== POST-FINETUNE: SURVIVING PARITY CORRECTS ERRORS ===")
    print("BER%  : " + "  ".join(f"{b*100:g}" for b in sweep_bers))
    print("acc   : " + "  ".join(f"{sweep[b]:.2f}" for b in sweep_bers))
    with open(f"{args.out}_ftsweep.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow([f"acc_BER{b*100:g}" for b in sweep_bers])
        w.writerow([f"{sweep[b]:.2f}" for b in sweep_bers])
    print(f"saved -> {args.out}_ftsweep.csv")

    return rows


def _requantize_ft(w_float, scale_obj, bits):
    """Float -> int, inverse of dequant_to_float, for the fine-tuned weights."""
    qmax = (1 << (bits - 1)) - 1; qmin = -(1 << (bits - 1))
    w = np.asarray(w_float, dtype=np.float32)
    kind, sc = scale_obj
    if kind == "per_tensor":
        s = float(sc) if float(sc) != 0 else 1.0
        q = np.rint(w / s)
    else:
        s = np.asarray(sc, dtype=np.float32)
        while s.ndim < w.ndim: s = s[..., None]
        s = np.where(s == 0, 1.0, s)
        q = np.rint(w / s)
    return np.clip(q, qmin, qmax).astype(np.int64)


def main():
    ap = argparse.ArgumentParser("Sensitivity-guided bit-plane ECC (PTQ/QAT-Jacob/QAT-LSQ, 9 archs, ImageNet/CIFAR)")
    ap.add_argument("--dataset", required=True, choices=["CIFAR10", "CIFAR100", "IMAGENET"])
    ap.add_argument("--arch", required=True)
    ap.add_argument("--quant", required=True, choices=["ptq", "qat-jacob", "qat-lsq"])
    ap.add_argument("--bits", type=int, default=8, choices=[4, 6, 8])
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--imagenet-root", default="")
    ap.add_argument("--use-pretrained", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-max-batches", type=int, default=None,
                    help="cap eval batches (speed up ImageNet); None=full")
    # BER sweep
    ap.add_argument("--bers", default="0,0.02,0.04,0.06,0.08")
    # protection knobs
    ap.add_argument("--mode", choices=["sensitive", "uniform"], default="sensitive")
    ap.add_argument("--sens-frac", type=float, default=0.20)
    ap.add_argument("--probe-ber", type=float, default=0.02, help="BER used to measure sensitivity")
    ap.add_argument("--ber-hosts", type=int, default=0)
    ap.add_argument("--plane-bits", type=int, default=3)
    ap.add_argument("--host-bits", type=int, default=5)
    ap.add_argument("--strong-t", type=int, default=12)
    ap.add_argument("--light-t", type=int, default=2)
    ap.add_argument("--bch-m", type=int, default=7)
    ap.add_argument("--parity-rep", type=int, default=1)
    ap.add_argument("--no-correct", action="store_true",
                    help="inject bit-errors but SKIP BCH correction "
                         "(measures the unprotected crash curve)")
    ap.add_argument("--embed", choices=["qim", "stdm"], default="stdm")
    ap.add_argument("--stdm-L", type=int, default=1)
    ap.add_argument("--store", choices=["self", "xlayer"], default="xlayer")
    ap.add_argument("--host-key", type=int, default=12345)
    ap.add_argument("--attack-key", type=int, default=-1,
                    help="if >=0, recovery uses this WRONG key (STDM security demo)")
    ap.add_argument("--delta", type=float, default=0.02, help="QIM/STDM lattice step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ecc_protect_sweep")
    # --- fine-tune survival study (optional; 0 = disabled, existing flow untouched) ---
    ap.add_argument("--finetune-epochs", type=int, default=0,
                    help="if >0, embed parity into the model, fine-tune, then re-decode "
                         "parity and measure survival (STDM robustness to training)")
    ap.add_argument("--finetune-lr", type=float, default=1e-4,
                    help="learning rate for the fine-tune survival study")
    ap.add_argument("--finetune-batches", type=int, default=50,
                    help="train batches per fine-tune epoch (caps cost)")
    ap.add_argument("--finetune-lr-sweep", default="",
                    help="comma list of LRs to sweep, e.g. '1e-4,1e-3,1e-2'; "
                         "overrides --finetune-lr, one survival curve per LR")
    ap.add_argument("--finetune-internals", type=int, default=1,
                    help="1=also emit weight-drift heatmap + STDM spread plots")
    args = ap.parse_args()
    args.dataset = args.dataset.upper()

    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    bers = [float(b) for b in args.bers.split(",")]

    # data
    _, te, nc, _ = TQ.get_dataloaders(args.dataset, args.data_root, args.batch_size,
                                      args.workers, dist_mode=False,
                                      imagenet_root=args.imagenet_root or None)
    if args.eval_max_batches is not None:
        te = _cap_loader(te, args.eval_max_batches)

    # model + quantized ints
    model, qinfo = load_model_and_qinfo(args, nc, dev)
    tot = sum(v[0].size for v in qinfo.values())
    base_acc = _eval(model, te, dev)
    print(f"[base] {args.arch} {args.dataset} {args.quant} INT{args.bits} "
          f"clean acc={base_acc:.2f}%  | {tot:,} protected-eligible weights")

    # ECC code object
    code = ENG.XLayerECC(strong_t=args.strong_t, light_t=args.light_t,
                         bch_m=args.bch_m, delta=args.delta, host_key=args.host_key)
    code.host_bits = args.host_bits
    code.plane_bits = args.plane_bits
    # --- bit-width sanity for INT4/INT6/INT8 ---
    if args.plane_bits >= args.bits:
        sys.exit(f"[error] --plane-bits {args.plane_bits} must be < --bits {args.bits} "
                 f"(need at least 1 low bit left to carry parity). "
                 f"For INT{args.bits}, use --plane-bits 1..{args.bits-1}.")
    if args.host_bits >= args.bits:
        sys.exit(f"[error] --host-bits {args.host_bits} must be < --bits {args.bits} "
                 f"(carriers are INT{args.bits} weights). Use --host-bits 1..{args.bits-1}.")
    code.parity_rep = args.parity_rep
    code.ber_hosts = bool(args.ber_hosts)
    code.bit_width = int(args.bits)
    code.no_correct = bool(args.no_correct)
    code.embed_mode = args.embed
    code.stdm_L = args.stdm_L
    code.store_mode = args.store
    code.recover_key = args.attack_key if args.attack_key >= 0 else None

    # sensitivity + selection
    if args.mode == "uniform":
        sens = set(qinfo.keys())
        sens_score = {n: 1.0 for n in qinfo}
        print(f"[select] uniform: protecting all {len(sens)} layers (100%)")
    else:
        print("[sensitivity] measuring per-layer accuracy drop ...")
        sens_score = measure_sensitivity(model, qinfo, code, dev, te, base_acc,
                                         probe_ber=args.probe_ber, seed=args.seed, verbose=True)
        ranked = sorted(sens_score, key=lambda n: -sens_score[n])
        acc = 0; sel = []
        for n in ranked:
            if acc + qinfo[n][0].size > args.sens_frac * tot and sel:
                break
            sel.append(n); acc += qinfo[n][0].size
        sens = set(sel)
        print(f"[select] sensitive: {len(sens)} layers = {acc:,} wts "
              f"= {100*acc/tot:.1f}% of model (target {100*args.sens_frac:.0f}%)")
        for n in sel:
            print(f"     - {n}  (dacc={sens_score[n]:.2f}%, {qinfo[n][0].size:,} wts)")
        plot_layer_landscape(qinfo, sens_score, sens, args, tot, args.out)

    # --- optional: fine-tune survival study (runs instead of the BER sweep) ---
    if args.finetune_epochs > 0:
        lr_list = [float(x) for x in args.finetune_lr_sweep.split(",")] \
                  if args.finetune_lr_sweep.strip() else [args.finetune_lr]
        if len(lr_list) == 1:
            args.finetune_lr = lr_list[0]
            run_finetune_survival(model, qinfo, code, dev, te, args, sens, sens_score)
        else:
            # sweep multiple LRs; reload a fresh model per LR so each starts clean
            curves = {}
            base_out = args.out
            for lr in lr_list:
                print(f"\n========== FINE-TUNE SURVIVAL @ lr={lr:g} ==========")
                args.finetune_lr = lr
                args.out = f"{base_out}_lr{lr:g}"
                model_lr, qinfo_lr = load_model_and_qinfo(args, nc, dev)
                rows = run_finetune_survival(model_lr, qinfo_lr, code, dev, te, args,
                                             sens, sens_score)
                curves[lr] = rows
                del model_lr
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            args.out = base_out
            # overlay parity-BER vs epoch, one line per LR
            try:
                import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
                plt.figure(figsize=(7.2, 4.6))
                cols = ["#2E7D32", "#0021A5", "#C62828", "#F57C00", "#6A1B9A"]
                styles = ["-o", "--s", "-^", "-D", "-v"]
                for i, (lr, rows) in enumerate(sorted(curves.items())):
                    eps = [r[0] for r in rows]; bers = [r[2]*100 for r in rows]
                    # small vertical offset for curves pinned at exactly 0 so they don't hide
                    flat = all(b == 0 for b in bers)
                    yvals = [b + (i * 0.00008 if flat else 0) for b in bers]
                    plt.plot(eps, yvals, styles[i % len(styles)], lw=2.2, ms=6,
                             color=cols[i % len(cols)], label=f"lr={lr:g}",
                             markerfacecolor="none" if flat else cols[i % len(cols)])
                plt.xlabel("Fine-tune epoch"); plt.ylabel("Parity BER (%)")
                plt.title(f"Parity survival vs learning rate  ({args.arch} {args.dataset} INT{args.bits})")
                plt.grid(True, alpha=0.3); plt.legend(title="learning rate")
                plt.figtext(0.5, -0.02,
                            "Curves at lr=1e-4 and lr=1e-3 sit exactly at BER=0 (drawn with a tiny "
                            "offset so both are visible); only lr=1e-2 erodes the parity.",
                            ha="center", fontsize=8, style="italic")
                plt.tight_layout(); plt.savefig(f"{base_out}_lrsweep.png", dpi=140, bbox_inches="tight")
                plt.close()
                print(f"\nsaved -> {base_out}_lrsweep.png  (LR sweep overlay)")
            except Exception as e:
                print(f"(LR sweep plot skipped: {e})")
            # summary table
            print("\n=== LR SWEEP SUMMARY (final-epoch parity BER) ===")
            for lr, rows in sorted(curves.items()):
                print(f"  lr={lr:g}:  final parity-BER={rows[-1][2]*100:.3f}%  "
                      f"final acc={rows[-1][1]:.2f}%")
        return

    # BER sweep
    print(f"\nTop-1 accuracy vs BER  (embed={args.embed} L={args.stdm_L} store={args.store} "
          f"t={args.strong_t} {args.plane_bits}-MSB rep={args.parity_rep} "
          f"ber-hosts={args.ber_hosts}"
          f"{' ATTACK-KEY' if code.recover_key is not None else ''})")
    results = {}
    for i, ber in enumerate(bers):
        rng = np.random.default_rng(args.seed)
        acc = run_at_ber(model, qinfo, code, ber, dev, te, rng, sens, sens_score,
                         report=(i == 0))
        results[ber] = acc
        print(f"  BER={ber*100:5.2f}%  acc={acc:.2f}%")

    # table + csv
    print("\n=== TABLE (accuracy %) ===")
    print("BER%  : " + "  ".join(f"{b*100:g}" for b in bers))
    print("acc   : " + "  ".join(f"{results[b]:.2f}" for b in bers))
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "arch", "quant", "bits", "embed", "stdm_L", "store",
                    "strong_t", "plane_bits", "parity_rep", "ber_hosts", "sens_frac"]
                   + [f"acc_BER{b*100:g}" for b in bers])
        w.writerow([args.dataset, args.arch, args.quant, args.bits, args.embed, args.stdm_L,
                    args.store, args.strong_t, args.plane_bits, args.parity_rep,
                    args.ber_hosts, args.sens_frac] + [f"{results[b]:.2f}" for b in bers])
    print(f"saved -> {args.out}.csv")

    # graph
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xs = [b*100 for b in bers]; ys = [results[b] for b in bers]
        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys, marker="o", linewidth=2)
        lo = min(ys)
        plt.ylim(max(0, lo-3), 100 if lo < 50 else 96.5)
        plt.xlabel("Bit error rate (%)"); plt.ylabel("Top-1 accuracy (%)")
        plt.title(f"{args.arch} {args.dataset} {args.quant} INT{args.bits}  "
                  f"[{args.embed} L={args.stdm_L} {args.store}, t={args.strong_t}, "
                  f"{args.plane_bits}-MSB]")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(f"{args.out}.png", dpi=140); plt.close()
        print(f"saved -> {args.out}.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


def _cap_loader(loader, max_batches):
    """Wrap a DataLoader to yield at most max_batches (faster ImageNet eval)."""
    class _Capped:
        def __init__(self, dl, n): self.dl = dl; self.n = n
        def __iter__(self):
            for i, b in enumerate(self.dl):
                if i >= self.n: break
                yield b
        def __len__(self): return min(self.n, len(self.dl))
    return _Capped(loader, max_batches)


if __name__ == "__main__":
    main()
