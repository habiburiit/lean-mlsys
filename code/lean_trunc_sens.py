#!/usr/bin/env python
"""
lean_trunc_sens.py  -  SAME as lean_trunc_sweep.py (all old flags kept) plus two new quantizer modes:
--mode requant_opt  : per-channel clipping point chosen to minimise plain MSE (LAPQ/ACIQ style)
--mode requant_sens : per-channel clipping point chosen to minimise SENSITIVITY-WEIGHTED error  sum d_i (w_i - Q(w_i))^2
                      (d_i = our bit-flip damage score; ours, not borrowed)
Original modes chop / requant unchanged.
--mode chop    : drop the low bits of the per-layer INT8 code (zeroes small weights; too crude for ImageNet)
--mode requant : MID/LOW weights get a per-output-channel scale and b-bit codes (default); HIGH stays INT8.
SEPARATE file;
imports helpers from lean_sensreg.py only (nothing there is changed).

For each checkpoint (baseline first = sets the bin thresholds) and each bit width b in --bits:
  HIGH weights keep 8 bits; MID and LOW weights keep only their top b bits (round-to-nearest).
Reports per (model, b): accuracy, bits/weight, storage/area/energy vs INT8, and connection counts:
  n_high (kept intact), n_trunc (truncated), n_zero (became exactly 0 = pruned connection).
Figures: accuracy vs bits, memory vs bits, pruned connections vs bits, bin occupancy per model.

Usage:
  python lean_trunc_sweep.py --dataset cifar10 --arch resnet18 \
     --ckpts <baseline.pth> <lam10.pth> <lam20.pth> ... --labels baseline lam10 lam20 ... \
     --bits 8,6,4,3,2,1 --out sensreg_results/cifar10_resnet18/trunc_sweep
"""
import argparse, json, os
import numpy as np, torch
import lean_sensreg as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpts", nargs="+", required=True); ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--bits", default="8,6,4,3,2,1", help="bits kept for MID and LOW weights (HIGH always 8)")
    ap.add_argument("--probe-batches", type=int, default=20)
    ap.add_argument("--mode", choices=["chop", "requant", "requant_opt", "requant_sens"], default="requant_sens",
                    help="chop = drop low bits of the per-layer INT8 code (old); requant = per-channel scale for MID/LOW at b bits (new)")
    ap.add_argument("--clip-grid", type=int, default=24, help="candidate clipping points per channel for requant_opt / requant_sens")
    ap.add_argument("--high-bits", type=int, default=8,
                    help="bits for HIGH-bin weights (default 8 = protected). Set equal to --bits to quantize everything, no binning, as a reference")
    ap.add_argument("--out", default="trunc_sweep")
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels = a.labels or [os.path.basename(c).replace(".pth", "") for c in a.ckpts]
    bits_list = [int(b) for b in a.bits.split(",")]
    _, vl = S.cifar_loaders(a.data, 128, a.workers)
    res = {}; thr = None
    for c, lab in zip(a.ckpts, labels):
        s, clean = S.score_checkpoint(c, a, dev)
        if thr is None: thr = (float(np.quantile(s, 0.99)), float(np.quantile(s, 0.91)))
        hi, mid = thr
        model = S.build_model(a.arch).to(dev); S.load_state_flex(model, c)
        params = S.weight_params(model)
        w0 = {n: p.detach().clone() for n, p in params}                 # original fp32 weights, restored for every bit setting
        store = S.quantize_model_(model, 8)
        base_q = [(n, p, q.clone(), scale) for n, p, q, scale in store]
        n_total = int(s.size); n_high = int((s >= hi).sum()); n_mid = int(((s >= mid) & (s < hi)).sum()); n_low = n_total - n_high - n_mid
        row = {"clean_fp32": clean, "n_total": n_total, "n_high": n_high, "n_mid": n_mid, "n_low": n_low, "per_bits": {}}
        for b in bits_list:
            off = 0; n_zero = 0; n_zero_before = 0
            with torch.no_grad():
                for n, p, q, scale in base_q:
                    seg = torch.from_numpy(s[off:off + q.numel()]).to(dev).view_as(q); off += q.numel()
                    qq = q.clone(); mask = seg < hi                     # MID + LOW get truncated
                    n_zero_before += int((q == 0).sum())
                    if b < 8 and a.mode == "chop":
                        # chop: one scale per layer (INT8 code), drop the low 8-b bits, round-to-nearest
                        step = 2 ** (8 - b)
                        qq[mask] = (torch.round(qq[mask] / step) * step).clamp(-128, 127)
                        w_new = qq * scale
                    elif b < 8:
                        # requant / requant_opt / requant_sens: MID/LOW weights get a per-output-channel scale at b bits
                        qmax = 2 ** (b - 1) - 1
                        w = w0[n]                                        # always start from the ORIGINAL weights
                        red = tuple(range(1, w.dim()))
                        wmax = (w.abs() * mask).amax(dim=red, keepdim=True).clamp(min=1e-12)
                        if a.mode == "requant":
                            ch_scale = wmax / qmax
                        else:
                            # search clipping point c in (0.2..1.0]*wmax per channel; weight errors by d_i if requant_sens
                            dw_ = seg.float() if a.mode == "requant_sens" else torch.ones_like(w)
                            best = None; best_scale = wmax / qmax
                            for f in torch.linspace(0.2, 1.0, a.clip_grid):
                                sc = (wmax * f / qmax).clamp(min=1e-12)
                                wq_ = torch.round(w / sc).clamp(-qmax - 1, qmax) * sc
                                err = ((w - wq_) ** 2 * dw_ * mask).sum(dim=red, keepdim=True)
                                if best is None: best, best_scale = err, sc
                                else:
                                    better = err < best; best = torch.where(better, err, best); best_scale = torch.where(better, sc, best_scale)
                            ch_scale = best_scale
                        wq = torch.round(w / ch_scale).clamp(-qmax - 1, qmax) * ch_scale
                        if a.high_bits < 8:                                 # reference: HIGH also low-bit, no protection
                            qh = 2 ** (a.high_bits - 1) - 1
                            hs = ((w.abs() * (~mask)).amax(dim=red, keepdim=True) / qh).clamp(min=1e-12)
                            wh = torch.round(w / hs).clamp(-qh - 1, qh) * hs
                            w_new = torch.where(mask, wq, wh)
                            qq = torch.where(mask, torch.round(w / ch_scale), torch.round(w / hs))
                        else:
                            w_new = torch.where(mask, wq, qq * scale)      # HIGH stays INT8 with the layer scale
                            qq = torch.where(mask, torch.round(w / ch_scale), qq)
                    else:
                        w_new = qq * scale
                    n_zero += int((qq == 0).sum()); p.copy_(w_new)
            acc = S.evaluate(model, vl, dev)
            avg_bits = (a.high_bits * n_high + b * (n_mid + n_low)) / n_total
            if a.mode != "chop" and b < 8:
                n_ch = sum(p.shape[0] for _, p in params)
                avg_bits += 8 * n_ch / n_total            # one 8-bit scale per output channel
            row["per_bits"][str(b)] = {"acc": acc, "avg_bits": avg_bits, "rel_cost": avg_bits / 8,
                                       "n_trunc": n_mid + n_low, "n_zero": n_zero, "n_zero_before": n_zero_before,
                                       "frac_zero": n_zero / n_total}
            print(f"{lab:10s} bits={b}  acc={acc:6.2f}  bits/w={avg_bits:.2f}  cost={100*avg_bits/8:5.1f}%  "
                  f"truncated={n_mid+n_low:,}  zeroed={n_zero:,} ({100*n_zero/n_total:.1f}%)", flush=True)
        res[lab] = row; del model; torch.cuda.empty_cache()

    os.makedirs(a.out, exist_ok=True); json.dump(res, open(os.path.join(a.out, "sweep.json"), "w"), indent=2)
    # summary table
    print(f"\n{'bits kept':10s}" + "".join(f"{l:>22s}" for l in labels))
    print(f"{'(MID/LOW)':10s}" + "".join(f"{'acc   bits/w  cost':>22s}" for _ in labels))
    for b in bits_list:
        print(f"{b:<10d}" + "".join(f"{res[l]['per_bits'][str(b)]['acc']:8.2f}{res[l]['per_bits'][str(b)]['avg_bits']:7.2f}{100*res[l]['per_bits'][str(b)]['rel_cost']:6.0f}%" for l in labels))

    # Table 1 style: per model at each bit width (HIGH share, bits/w, saving, acc before/after/lost)
    md = []
    for b in bits_list:
        hdr = f"\nTable: MID and LOW weights truncated to {b} bit{'s' if b>1 else ''}, HIGH kept at 8 bits"
        print(hdr); md.append(hdr + "\n")
        cols = "| model | HIGH bin (8 bits) | bits/weight | saving vs INT8 | acc before | acc after | acc lost |"
        print(f"{'model':10s}{'HIGH%':>8s}{'bits/w':>8s}{'saving':>9s}{'before':>9s}{'after':>8s}{'lost':>7s}")
        md.append(cols + "\n|---|---|---|---|---|---|---|\n")
        for l in labels:
            r = res[l]; pb = r["per_bits"][str(b)]; hp = 100 * r["n_high"] / r["n_total"]; sav = 100 * (1 - pb["rel_cost"])
            print(f"{l:10s}{hp:8.2f}{pb['avg_bits']:8.2f}{sav:8.1f}%{r['clean_fp32']:9.2f}{pb['acc']:8.2f}{r['clean_fp32']-pb['acc']:7.2f}")
            md.append(f"| {l} | {hp:.2f}% | {pb['avg_bits']:.2f} | {sav:.1f}% | {r['clean_fp32']:.1f} | {pb['acc']:.1f} | {r['clean_fp32']-pb['acc']:.1f} |\n")
    md.append("\nTable: accuracy after truncation, all settings\n\n| bits kept (MID/LOW) | saving vs INT8 | " + " | ".join(labels) + " |\n|---|---|" + "---|" * len(labels) + "\n")
    for b in bits_list:
        sav = 100 * (1 - res[labels[0]]["per_bits"][str(b)]["rel_cost"])
        md.append(f"| {b} | {sav:.0f}% | " + " | ".join(f"{res[l]['per_bits'][str(b)]['acc']:.1f}" for l in labels) + " |\n")
    open(os.path.join(a.out, "tables.md"), "w").write("".join(md)); print(f"[saved] {a.out}/tables.md  (paste into email)")

    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    cols = plt.cm.viridis(np.linspace(0.05, 0.95, len(labels))); cols[0] = (0.55, 0.55, 0.55, 1)
    x = np.arange(len(bits_list)); w = 0.8 / len(labels)
    def grouped(ax, key, ylabel, title, fmt="{:.1f}", scale=1.0):
        for j, l in enumerate(labels):
            v = [scale * res[l]["per_bits"][str(b)][key] for b in bits_list]
            bars = ax.bar(x + (j - (len(labels)-1)/2) * w, v, w, color=cols[j], label=l, edgecolor="white", linewidth=0.4)
            if len(labels) <= 5:
                for bb, vv in zip(bars, v): ax.text(bb.get_x() + bb.get_width()/2, vv, fmt.format(vv), ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels([f"{b} bit" + ("s" if b > 1 else "") for b in bits_list]); ax.set_xlabel("bits kept for MID and LOW weights (HIGH stays 8)")
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.25)
    # 1 accuracy
    fig, ax = plt.subplots(figsize=(9, 3.8)); grouped(ax, "acc", "accuracy (%)", "Accuracy after sensitivity-guided truncation"); ax.legend(fontsize=7, ncol=len(labels)); plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig1_accuracy.png"), dpi=200); plt.close(fig)
    # 2 memory / area / energy
    fig, ax = plt.subplots(figsize=(9, 3.8)); grouped(ax, "rel_cost", "% of INT8", "Weight storage, SRAM area and memory-access energy (relative to all-INT8)", fmt="{:.0f}", scale=100); ax.legend(fontsize=7, ncol=len(labels)); plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig2_memory_area_energy.png"), dpi=200); plt.close(fig)
    # 3 pruned connections
    fig, ax = plt.subplots(figsize=(9, 3.8)); grouped(ax, "frac_zero", "% of all weights", "Connections that become exactly zero (pruned) after truncation", fmt="{:.0f}", scale=100); ax.legend(fontsize=7, ncol=len(labels)); plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig3_pruned_connections.png"), dpi=200); plt.close(fig)
    # 4 bin occupancy per model (counts)
    fig, ax = plt.subplots(figsize=(7, 3.8)); xx = np.arange(len(labels)); ww = 0.26
    for k, (key, name, col) in enumerate([("n_high", "HIGH (8 bits, intact)", "#dc2626"), ("n_mid", "MID (truncated)", "#f59e0b"), ("n_low", "LOW (truncated)", "#10b981")]):
        v = [res[l][key] for l in labels]; bars = ax.bar(xx + (k-1)*ww, v, ww, color=col, label=name)
        for bb, vv in zip(bars, v): ax.text(bb.get_x()+bb.get_width()/2, vv, f"{vv/1e3:.0f}k" if vv < 1e6 else f"{vv/1e6:.2f}M", ha="center", va="bottom", fontsize=6)
    ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels(labels); ax.set_ylabel("number of weights (log)"); ax.set_title("Connections per sensitivity bin: fewer HIGH means fewer weights that must stay at 8 bits", fontsize=9); ax.legend(fontsize=7); ax.grid(axis="y", alpha=.25)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig4_bin_connections.png"), dpi=200); plt.close(fig)
    # 5 accuracy vs cost curve
    fig, ax = plt.subplots(figsize=(6, 4))
    for j, l in enumerate(labels):
        pts = sorted((100*res[l]["per_bits"][str(b)]["rel_cost"], res[l]["per_bits"][str(b)]["acc"], b) for b in bits_list)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=cols[j], label=l)
        for cx, cy, b in pts: ax.annotate(f"{b}b", (cx, cy), fontsize=6, textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("memory / area / energy (% of INT8)"); ax.set_ylabel("accuracy (%)"); ax.set_title("Accuracy vs cost: up and to the left is better", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig5_accuracy_vs_cost.png"), dpi=200); plt.close(fig)
    print(f"[saved] figures in {a.out}/")


if __name__ == "__main__":
    main()
