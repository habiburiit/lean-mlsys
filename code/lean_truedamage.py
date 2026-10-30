#!/usr/bin/env python
"""
lean_truedamage.py  -  is the proxy d_i = |g_i| * Delta_l honest, or does LEAN just hide weights from the gradient?
SEPARATE file; imports lean_sensreg unchanged.

For each checkpoint (baseline first, then LEAN):
  1. quantize to INT8 per tensor, measure clean loss on a fixed probe set
  2. rank weights by the proxy (own gradient), take the top-K and a random sample
  3. REALLY flip the sign bit of each sampled weight, one at a time, and measure the true loss increase
  4. report: true damage of the proxy-top weights, true damage of the worst weights found, proxy-vs-true rank
     correlation, and the per-layer Delta_l ratio LEAN/baseline
Also evaluates, in the LEAN model, the weights that were the baseline's top-K (same indices), so "hidden"
weights are checked directly.

usage:
  python lean_truedamage.py --dataset cifar10 --arch resnet18 \
     --ckpts artifacts/models/cifar10/resnet18/model_float32.pth artifacts/models/cifar10/resnet18/model_float32_sensreg_lam40.pth \
     --labels baseline lam40 --topk 1000 --nrand 1000 --out sensreg_results/cifar10_resnet18/truedamage
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn.functional as F
import lean_sensreg as S
try:
    import lean_vgg
except Exception:
    pass


def probe_set(a, dev):
    _, vl = S.cifar_loaders(a.data, 128, a.workers)
    xs, ys = [], []
    for i, (x, y) in enumerate(vl):
        if i >= a.probe_batches: break
        xs.append(x.to(dev)); ys.append(y.to(dev))
    return xs, ys


@torch.no_grad()
def loss_on(model, xs, ys):
    model.eval(); tot = 0.0
    for x, y in zip(xs, ys):
        tot += F.cross_entropy(model(x), y, reduction="sum").item()
    return tot / sum(y.numel() for y in ys)


def proxy_scores(model, params, xs, ys):
    acc = None
    for x, y in zip(xs, ys):
        model.eval()
        loss = F.cross_entropy(model(x), y)
        grads = torch.autograd.grad(loss, [p for _, p in params])
        s = torch.cat([(g.abs() * S.msb_step(p)).flatten() for g, (_, p) in zip(grads, params)])
        acc = s if acc is None else acc + s
    return (acc / len(xs)).cpu().numpy()


def true_damage(model, store, offs, idx, xs, ys, base_loss):
    """really flip the INT8 sign bit of weight idx (global index), measure loss, restore."""
    out = np.zeros(len(idx))
    with torch.no_grad():
        for t, i in enumerate(idx):
            j = int(np.searchsorted(offs, i, side="right") - 1); k = int(i - offs[j])
            _, p, q, scale = store[j]
            qf = q.view(-1); old = int(qf[k]); new = old - 128 if old >= 0 else old + 128
            p.view(-1)[k] = new * scale
            out[t] = loss_on(model, xs, ys) - base_loss
            p.view(-1)[k] = old * scale
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--ckpts", nargs="+", required=True, help="baseline first")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--topk", type=int, default=1000); ap.add_argument("--nrand", type=int, default=1000)
    ap.add_argument("--probe-batches", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="truedamage")
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda"; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    xs, ys = probe_set(a, dev)
    os.makedirs(a.out, exist_ok=True); res = {}; base_top_idx = None; base_delta = None

    for c, lab in zip(a.ckpts, a.labels):
        t0 = time.time()
        model = S.build_model(a.arch).to(dev); S.load_state_flex(model, c); model.eval()
        params = S.weight_params(model)
        prox = proxy_scores(model, params, xs, ys)               # float model proxy, as in the paper
        store = S.quantize_model_(model, 8)
        sizes = [q.numel() for _, _, q, _ in store]; offs = np.cumsum([0] + sizes)[:-1]; N = int(sum(sizes))
        delta = np.array([128.0 * float(s) for _, _, _, s in store])
        base_loss = loss_on(model, xs, ys)
        order = np.argsort(-prox); top = order[:a.topk]
        rand = rng.choice(N, a.nrand, replace=False)
        if base_top_idx is None: base_top_idx = top.copy(); base_delta = delta.copy()

        td_top = true_damage(model, store, offs, top, xs, ys, base_loss)
        td_rand = true_damage(model, store, offs, rand, xs, ys, base_loss)
        td_basetop = td_top if lab == a.labels[0] else true_damage(model, store, offs, base_top_idx, xs, ys, base_loss)
        # rank agreement between proxy and true damage over the sampled set
        samp = np.concatenate([top, rand]); tv = np.concatenate([td_top, td_rand]); pv = prox[samp]
        from scipy.stats import spearmanr
        rho = float(spearmanr(pv, tv).correlation)
        share_top1 = float(np.sort(prox)[-int(0.01 * N):].sum() / prox.sum())
        r = {"ckpt": c, "clean_loss_int8": base_loss, "n_weights": N,
             "proxy_top1pct_damage_share": share_top1,
             "true_dmg_proxy_top%d_mean" % a.topk: float(td_top.mean()), "true_dmg_proxy_top%d_median" % a.topk: float(np.median(td_top)),
             "true_dmg_worst100_mean": float(np.sort(tv)[-100:].mean()), "true_dmg_worst1": float(tv.max()),
             "true_dmg_random_mean": float(td_rand.mean()), "true_dmg_random_frac_gt_0.01": float((td_rand > 0.01).mean()),
             "true_dmg_of_BASELINE_top_idx_mean": float(td_basetop.mean()),
             "spearman_proxy_vs_true": rho,
             "delta_ratio_vs_baseline_mean": float((delta / base_delta).mean()), "delta_ratio_min": float((delta / base_delta).min()),
             "delta_ratio_max": float((delta / base_delta).max())}
        res[lab] = r
        print(f"\n[{lab}]  {time.time()-t0:.0f}s   INT8 clean loss {base_loss:.4f}   N={N/1e6:.1f}M")
        print(f"  proxy: share of total damage in top 1%           {share_top1*100:6.1f}%")
        print(f"  TRUE loss increase, proxy-top-{a.topk} weights       mean {td_top.mean():.4f}  median {np.median(td_top):.4f}")
        print(f"  TRUE loss increase, worst 100 found / worst 1       {np.sort(tv)[-100:].mean():.4f} / {tv.max():.4f}")
        print(f"  TRUE loss increase, random weights                   mean {td_rand.mean():.5f}  frac>0.01 {100*(td_rand>0.01).mean():.1f}%")
        print(f"  TRUE loss increase of the BASELINE's top-{a.topk} idx   mean {td_basetop.mean():.4f}")
        print(f"  Spearman(proxy, true) over sampled weights           {rho:.3f}")
        print(f"  Delta_l ratio vs baseline  mean {r['delta_ratio_vs_baseline_mean']:.3f}  min {r['delta_ratio_min']:.3f}  max {r['delta_ratio_max']:.3f}", flush=True)
        np.save(os.path.join(a.out, f"{lab}_true_top.npy"), td_top); np.save(os.path.join(a.out, f"{lab}_true_rand.npy"), td_rand)
        del model, store; torch.cuda.empty_cache()

    json.dump(res, open(os.path.join(a.out, "truedamage.json"), "w"), indent=2)
    print("\nsummary (true loss increase per single sign flip, probe set):")
    print(f"{'model':12s}{'top1% share':>12s}{'proxy-top mean':>16s}{'worst100':>10s}{'base-top idx':>14s}{'random':>10s}{'rho':>7s}{'Delta ratio':>12s}")
    for lab, r in res.items():
        print(f"{lab:12s}{100*r['proxy_top1pct_damage_share']:11.1f}%{r['true_dmg_proxy_top%d_mean'%a.topk]:16.4f}{r['true_dmg_worst100_mean']:10.4f}"
              f"{r['true_dmg_of_BASELINE_top_idx_mean']:14.4f}{r['true_dmg_random_mean']:10.5f}{r['spearman_proxy_vs_true']:7.3f}{r['delta_ratio_vs_baseline_mean']:12.3f}")


if __name__ == "__main__":
    main()
