#!/usr/bin/env python
"""
plot_sens_curves.py  -  number of weights vs sensitivity, several models on one axis, baseline vs lambda.
SEPARATE file; imports from lean_sensreg.py (nothing there is changed).

For every (dataset, arch, checkpoint) it computes the damage score d_i = |dL/dw_i| * INT8 sign-flip step (same
definition as the bins everywhere), on --probe-batches batches, and histograms it. Two things make the curves
comparable across models:
  x axis  : sensitivity as a percentage of the BASELINE model's maximum for that dataset/arch (0..100), log-spaced bins
  y axis  : number of weights per bin (log scale), or fraction of weights with --norm
Vertical lines mark the baseline's bin thresholds: top 1% (HIGH) and top 10% (MID) for that model.
Saves a json with the histograms so the figure can be redrawn without recomputing.

Usage (pairs are "dataset:arch:ckpt:label"; put the baseline of each dataset/arch FIRST):
  python plot_sens_curves.py --out sensreg_results/sens_curves_c10 \
     --pairs cifar10:resnet18:artifacts/models/cifar10/resnet18/model_float32.pth:"ResNet-18 baseline" \
             cifar10:resnet18:artifacts/models/cifar10/resnet18/model_float32_sensreg_lam40.pth:"ResNet-18 lam40" \
             cifar10:resnet50:artifacts/models/cifar10/resnet50/model_float32.pth:"ResNet-50 baseline" \
             cifar10:resnet50:artifacts/models/cifar10/resnet50/model_float32_sensreg_lam40.pth:"ResNet-50 lam40"
"""
import argparse, json, os
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lean_sensreg as S


def damage(dataset, arch, ckpt, probe, data, workers, imagenet_root, dev):
    S._DS = dataset; S.IMAGENET_ROOT = imagenet_root
    _, vl = S.cifar_loaders(data, 128, workers)
    model = S.build_model(arch).to(dev); S.load_state_flex(model, ckpt); params = S.weight_params(model)
    plist = [p for _, p in params]; acc = None; n = 0; model.eval()
    for x, y in vl:
        if n >= probe: break
        x, y = x.to(dev), y.to(dev); loss = F.cross_entropy(model(x), y)
        g = torch.autograd.grad(loss, plist)
        d = torch.cat([(gg.abs() * S.msb_step(p)).flatten().cpu() for gg, p in zip(g, plist)])
        acc = d if acc is None else acc + d; n += 1; model.zero_grad(set_to_none=True)
    del model; torch.cuda.empty_cache()
    return (acc / n).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="dataset:arch:ckpt:label, baseline first per dataset/arch")
    ap.add_argument("--nbins", type=int, default=40); ap.add_argument("--probe-batches", type=int, default=20)
    ap.add_argument("--norm", action="store_true", help="y = fraction of weights instead of count")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imagenet-root", default="./imagenet-val"); ap.add_argument("--out", default="sens_curves")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    edges = np.logspace(-3, 2, a.nbins + 1)                # 0.001% .. 100% of baseline max, log bins
    res = {}; ref = {}
    for pr in a.pairs:
        ds, arch, ck, lab = pr.split(":", 3)
        d = damage(ds, arch, ck, a.probe_batches, a.data, a.workers, a.imagenet_root, dev)
        key = f"{ds}:{arch}"
        if key not in ref:
            ref[key] = {"max": float(d.max()), "th_high": float(np.quantile(d, 0.99)), "th_mid": float(np.quantile(d, 0.90))}
        r = ref[key]; x = 100.0 * d / r["max"]
        h, _ = np.histogram(np.clip(x, edges[0], edges[-1]), bins=edges)
        res[lab] = {"dataset": ds, "arch": arch, "ckpt": ck, "n": int(d.size), "hist": h.tolist(),
                    "th_high_pct": 100.0 * r["th_high"] / r["max"], "th_mid_pct": 100.0 * r["th_mid"] / r["max"],
                    "frac_above_high": float((d >= r["th_high"]).mean()), "frac_above_mid": float((d >= r["th_mid"]).mean())}
        print(f"{lab:28s} n={d.size:>10,}  above HIGH thr {100*res[lab]['frac_above_high']:.3f}%  above MID thr {100*res[lab]['frac_above_mid']:.2f}%", flush=True)
    json.dump({"edges": edges.tolist(), "models": res}, open(os.path.join(a.out, "sens_curves.json"), "w"), indent=1)

    # ---- figure: one axis, colour per dataset/arch, solid = baseline, dashed = lambda ----
    keys = list(dict.fromkeys(f"{v['dataset']}:{v['arch']}" for v in res.values()))
    cols = plt.cm.tab10(np.arange(len(keys)))
    cen = np.sqrt(edges[:-1] * edges[1:])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    seen = {}
    for lab, v in res.items():
        k = f"{v['dataset']}:{v['arch']}"; c = cols[keys.index(k)]
        ls = "-" if k not in seen else "--"; seen[k] = True
        y = np.array(v["hist"], dtype=float); y = y / v["n"] if a.norm else y
        y = np.where(y > 0, y, np.nan)
        ax.plot(cen, y, ls, color=c, lw=2.2, label=lab)
    # bin regions from the first model (baseline) of the first key
    first = next(iter(res.values()))
    ax.axvline(first["th_mid_pct"], color="k", lw=1, ls=":"); ax.axvline(first["th_high_pct"], color="k", lw=1, ls=":")
    ymax = ax.get_ylim()[1]
    for xx, t in ((np.sqrt(edges[0] * first["th_mid_pct"]), "least sensitive\n(LOW, 90%)"),
                  (np.sqrt(first["th_mid_pct"] * first["th_high_pct"]), "moderate\n(MID, 9%)"),
                  (np.sqrt(first["th_high_pct"] * 100), "high\n(HIGH, 1%)")):
        ax.text(xx, ymax, t, ha="center", va="top", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sensitivity (% of baseline max,  damage = |dL/dw| x INT8 sign-flip step)", fontsize=11)
    ax.set_ylabel("fraction of weights" if a.norm else "number of weights", fontsize=11)
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9, ncol=2, frameon=False, loc="lower left")
    ax.set_title("Weight sensitivity distribution: baseline (solid) vs lambda (dashed)", fontsize=12, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "sens_curves.png"), dpi=200)
    print(f"saved {a.out}/sens_curves.png and sens_curves.json")


if __name__ == "__main__":
    main()
