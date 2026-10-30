#!/usr/bin/env python
"""
lean_trunc.py  -  per-bin mixed precision (truncation) study, SEPARATE from lean_sensreg.py.
Imports helpers from lean_sensreg.py (read-only, nothing there is modified).

HIGH-bin weights keep 8 bits. MID and LOW weights are truncated to fewer bits: keep the top
`bits` bits of the INT8 code, round-to-nearest on the dropped low bits.
Bin thresholds come from the FIRST checkpoint (baseline) so all models share the same bins.
Reports accuracy, average bits/weight, and storage / area / memory-access energy relative to
all-INT8 (linear model: relative cost = avg_bits / 8).

Usage (All_final, ANON):
  python lean_trunc.py --dataset cifar10 --arch resnet18 \
      --ckpts artifacts/models/cifar10/resnet18/model_float32.pth \
              artifacts/models/cifar10/resnet18/model_float32_sensreg_lam10.pth \
              artifacts/models/cifar10/resnet18/model_float32_sensreg_lam30.pth \
      --labels baseline lam10 lam30 --bits-cfg 8/8,8/6,8/4,6/4,4/4,6/2,4/2 --out sensreg_results/cifar10_resnet18/trunc
"""
import argparse, json, os
import numpy as np
import torch
import lean_sensreg as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpts", nargs="+", required=True, help="first = baseline (sets the bin thresholds)")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--bits-cfg", default="8/8,8/6,8/4,6/4,4/4,6/2,4/2", help="MID/LOW bit widths, comma list")
    ap.add_argument("--probe-batches", type=int, default=20)
    ap.add_argument("--out", default="trunc_out")
    a = ap.parse_args()
    S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root          # same globals the main script sets from its CLI

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels = a.labels if a.labels else [os.path.basename(c).replace(".pth", "") for c in a.ckpts]
    _, vl = S.cifar_loaders(a.data, 128, a.workers)
    cfgs = [(int(m), int(l)) for m, l in (t.split("/") for t in a.bits_cfg.split(","))]
    res = {}; thr = None
    for c, lab in zip(a.ckpts, labels):
        s, clean = S.score_checkpoint(c, a, dev)
        if thr is None:
            thr = (float(np.quantile(s, 0.99)), float(np.quantile(s, 0.91)))
        hi, mid = thr
        model = S.build_model(a.arch).to(dev); S.load_state_flex(model, c)
        params = S.weight_params(model); store = S.quantize_model_(model, 8)
        base_q = [(n, p, q.clone(), scale) for n, p, q, scale in store]
        frac = {"HIGH": float((s >= hi).mean()), "MID": float(((s >= mid) & (s < hi)).mean()), "LOW": float((s < mid).mean())}
        row = {"clean_fp32": clean, "int8": S.evaluate(model, vl, dev), "bins": frac}
        for bm, bl in cfgs:
            off = 0
            with torch.no_grad():
                for n, p, q, scale in base_q:
                    seg = torch.from_numpy(s[off:off + q.numel()]).to(dev).view_as(q); off += q.numel()
                    qq = q.clone()
                    for bits, mask in ((bm, (seg >= mid) & (seg < hi)), (bl, seg < mid)):
                        drop = 8 - bits
                        if drop > 0 and mask.any():
                            step = 2 ** drop
                            qq[mask] = (torch.round(qq[mask] / step) * step).clamp(-128, 127)   # keep top `bits`, round
                    p.copy_(qq * scale)
            acc = S.evaluate(model, vl, dev)
            avg_bits = 8 * frac["HIGH"] + bm * frac["MID"] + bl * frac["LOW"]
            row[f"mid{bm}_low{bl}"] = {"acc": acc, "avg_bits": avg_bits, "rel_storage": avg_bits / 8,
                                       "rel_area": avg_bits / 8, "rel_mem_energy": avg_bits / 8}
        res[lab] = row; del model; torch.cuda.empty_cache()
        print(f"{lab:12s} fp32 {clean:6.2f}  int8 {row['int8']:6.2f}  bins H/M/L = "
              f"{100*frac['HIGH']:.2f}/{100*frac['MID']:.2f}/{100*frac['LOW']:.2f}%", flush=True)

    print(f"\n{'config':14s}" + "".join(f"{l+' acc':>12s}{l+' bits':>11s}" for l in labels))
    for bm, bl in cfgs:
        k = f"mid{bm}_low{bl}"
        print(f"{'MID'+str(bm)+'/LOW'+str(bl):14s}" + "".join(f"{res[l][k]['acc']:12.2f}{res[l][k]['avg_bits']:11.2f}" for l in labels))
    print("\nrelative to all-INT8: storage = area = memory-access energy = avg_bits / 8 (linear model; HIGH always 8 bits)")
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(os.path.join(a.out, "trunc.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for l in labels:
            pts = sorted((res[l][f"mid{bm}_low{bl}"]["avg_bits"], res[l][f"mid{bm}_low{bl}"]["acc"]) for bm, bl in cfgs)
            ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", label=l)
            for bm, bl in cfgs:
                r = res[l][f"mid{bm}_low{bl}"]
                ax.annotate(f"{bm}/{bl}", (r["avg_bits"], r["acc"]), fontsize=6, textcoords="offset points", xytext=(3, 3))
        ax.set_xlabel("average bits per weight (storage, area, memory energy scale with this)")
        ax.set_ylabel("accuracy (%)"); ax.set_title("Per-bin mixed precision: HIGH 8 bits, MID/LOW truncated")
        ax.grid(alpha=.3); ax.legend(fontsize=7); plt.tight_layout()
        plt.savefig(os.path.join(a.out, "trunc_tradeoff.png"), dpi=200); print(f"[saved] {a.out}/trunc_tradeoff.png")
    except Exception as e:
        print("[plot skipped]", e)


if __name__ == "__main__":
    main()
