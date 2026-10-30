#!/usr/bin/env python
"""
plot_compare_hessian.py  -  compare baseline / Hessian-aware (Pranto et al. 2025) / ours (lambda) under the same
real PBS and the same ECC budget, from the csv files written by pbs_four.py (sensreg_results/pbs_four/*.csv).
Expected labels in the csv 'model' column: baseline, hessian, lam<k>.

Usage:
  python plot_compare_hessian.py --dataset cifar10 --arch resnet18 --lam lam10 --clean 95.1 --out fig_compare_c10r18.png
"""
import argparse, csv, glob, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 13, "axes.titleweight": "bold", "axes.labelweight": "bold"})


def load(dataset, arch, label, protect):
    f = f"sensreg_results/pbs_four/{dataset}_{arch}_{label}_{protect}.csv"
    if not os.path.exists(f): return None
    rows = list(csv.DictReader(open(f)))
    out = {}
    for r in rows:
        fl = [k for k in r if k.startswith("flips_")]
        out[r["protection"]] = {int(k.split("_")[1]): float(r[k]) for k in fl}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--lam", default="lam10"); ap.add_argument("--clean", type=float, default=95.1)
    ap.add_argument("--cov", default="selcode 1.00%"); ap.add_argument("--out", default="fig_compare.png")
    a = ap.parse_args()
    series = {}
    for lab, name, col in [("baseline", "normal model", "#e07b39"), ("pc", "Piece-wise Clustering (CVPR 2020)", "#b07aa1"), ("wcat", "WCAT (TMLR 2023)", "#d4a017"), ("hessian", "Hessian-aware (2025)", "#2a9d8f"), ("far", "Forget-and-Rewire (USENIX Sec 2024)", "#6c757d"), (a.lam, f"ours ({a.lam})", "#1e3a8a")]:
        d = load(a.dataset, a.arch, lab, "selcode")
        if d and a.cov in d: series[name] = (col, d[a.cov])
        n = load(a.dataset, a.arch, lab, "none")
        if n and "none" in n: series[name + ", no ECC"] = (col, n["none"])
    fig, ax = plt.subplots(figsize=(10, 6)); ax.axhline(a.clean, color="#222", ls="--", lw=1.6, label=f"no attack ({a.clean})")
    for name, (col, d) in series.items():
        xs = sorted(d); ys = [d[x] for x in xs]
        ls = ":" if "no ECC" in name else "-"; ax.plot(xs, ys, ls + "o", color=col, lw=2.4, ms=8, label=name + ("" if "no ECC" in name else f" + the selective code ECC ({a.cov.split()[1]})"))
    ax.set_xscale("log"); allx = sorted({x for _, (_, d) in series.items() for x in d}); ax.set_xticks(allx); ax.set_xticklabels([str(x) for x in allx])
    ax.set_xlabel("number of PBS bit flips"); ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 105); ax.grid(alpha=.3)
    ax.set_title(f"{a.arch} {a.dataset}: real PBS on INT8, baseline vs published defenses vs ours"); ax.legend(fontsize=10, frameon=False, loc="lower left")
    plt.tight_layout(); plt.savefig(a.out, dpi=200, bbox_inches="tight"); print("saved", a.out)


if __name__ == "__main__":
    main()
