#!/usr/bin/env python
"""
lean_eval_mp.py  -  evaluate MIXED-PRECISION models in the format they were trained for.
SEPARATE file; imports from lean_sensreg.py (nothing there is changed).

Rebuilds the deployed weights: HIGH (top --tail-frac by damage) at 8-bit per-tensor, MID (next --mid-frac)
at --mid-bits per-channel, LOW at --low-bits per-channel, exactly like lean_train_qat_mp.py.
Then reports, per checkpoint:
  clean accuracy in that format
  sensitivity bins (damage d_i = |g_i| * dw_i with dw_i = the sign-bit flip step OF THAT WEIGHT'S FORMAT),
      thresholds fixed from the FIRST checkpoint
  accuracy after targeted sign-bit flips applied in that format (most damaging first)

Usage:
  python lean_eval_mp.py --dataset imagenet --arch resnet50 --imagenet-root <path> --mid-bits 6 --low-bits 3 \
      --ckpts pretrained <mp_6_3_lam0.pth> <mp_6_3_lam40.pth> --labels baseline mp63_lam0 mp63_lam40 \
      --nflips 1,2,3,5,10,20 --out sensreg_results/imagenet_resnet50/eval_mp63
"""
import argparse, json, os
import numpy as np, torch, torch.nn.functional as F
import lean_sensreg as S


def damage_scores(model, params, vl, dev, probe_batches):
    """first-order gradient magnitude per weight (format-independent part)"""
    plist = [p for _, p in params]; acc = None; n = 0; model.eval()
    for x, y in vl:
        if n >= probe_batches: break
        x, y = x.to(dev), y.to(dev)
        loss = F.cross_entropy(model(x), y); grads = torch.autograd.grad(loss, plist)
        g = torch.cat([gg.abs().flatten().cpu() for gg in grads]); acc = g if acc is None else acc + g; n += 1
        model.zero_grad(set_to_none=True)
    return (acc / n)


def build_mp(params, gflat, a, saved=None):
    """quantize in place using a per-weight tier (8 = per-tensor INT8; 6/4/3 = per-channel). Returns store tuples.
    Tier source: saved["tier"] (mp3 models), else saved high/mid masks (mp2 models), else recomputed like training."""
    plist = [p for _, p in params]; sizes = [p.numel() for p in plist]; offs = np.cumsum([0] + sizes)
    if saved is None:
        d = torch.cat([(gflat[offs[i]:offs[i+1]].view_as(p.cpu()) * S.msb_step(p).cpu()).flatten() for i, p in enumerate(plist)])
        kh = max(1, int(a.tail_frac * d.numel())); km = max(1, int((a.tail_frac + a.mid_frac) * d.numel()))
        th = torch.topk(d, kh, sorted=False).values.min(); tm = torch.topk(d, km, sorted=False).values.min()
    store = []; QM = {8: 127, 6: 31, 5: 15, 4: 7, 3: 3, 2: 1}
    with torch.no_grad():
        for i, (n, p) in enumerate(params):
            if saved is not None and "tier" in saved:
                tier = saved["tier"][n].to(p.device).to(torch.int64)
            else:
                if saved is not None:
                    high = saved["high"][n].to(p.device); mid = saved["mid"][n].to(p.device)
                else:
                    seg = d[offs[i]:offs[i+1]].view_as(p.cpu()).to(p.device); high = seg >= th; mid = (seg >= tm) & (seg < th)
                tier = torch.full_like(p, a.low_bits, dtype=torch.int64); tier[mid] = a.mid_bits; tier[high] = 8
            red = tuple(range(1, p.dim())); w = p.detach().clone()
            q = torch.zeros_like(w); sc = torch.zeros_like(w); fm = torch.zeros_like(w)
            for b in torch.unique(tier).tolist():
                m = tier == b; qm = QM[b]
                s_ = (w.abs().max() / 127).clamp(min=1e-12).expand_as(w) if b == 8 else ((w.abs() * m).amax(dim=red, keepdim=True) / qm).clamp(min=1e-12).expand_as(w)
                q = torch.where(m, torch.round(w / s_).clamp(-qm - 1, qm), q); sc = torch.where(m, s_, sc)
                fm = torch.where(m, torch.full_like(w, float(2 ** (b - 1))), fm)
            p.copy_(q * sc)
            store.append((n, p, q, sc, fm, tier == 8, (tier < 8) & (tier > a.low_bits)))
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpts", nargs="+", required=True, help="first = baseline, sets bin thresholds")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--mid-bits", type=int, default=6); ap.add_argument("--low-bits", type=int, default=3)
    ap.add_argument("--tail-frac", type=float, default=0.01); ap.add_argument("--mid-frac", type=float, default=0.08)
    ap.add_argument("--nflips", default="1,2,3,5,10,20"); ap.add_argument("--probe-batches", type=int, default=20)
    ap.add_argument("--out", default="eval_mp")
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels = a.labels or [os.path.basename(c).replace(".pth", "") for c in a.ckpts]
    _, vl = S.cifar_loaders(a.data, 128, a.workers); nflips = [int(x) for x in a.nflips.split(",")]
    res = {}; thr = None
    print(f"format: HIGH 8-bit per-tensor, MID {a.mid_bits}-bit per-channel, LOW {a.low_bits}-bit per-channel")
    for c, lab in zip(a.ckpts, labels):
        model = S.build_model(a.arch).to(dev); S.load_state_flex(model, c); params = S.weight_params(model)
        saved = torch.load(c + ".bins.pt", map_location="cpu") if os.path.exists(c + ".bins.pt") else None
        print(f"[{lab}] bin assignment: {'saved from training' if saved is not None else 'recomputed (no .bins.pt found)'}")
        g = None if saved is not None else damage_scores(model, params, vl, dev, a.probe_batches)
        store = build_mp(params, g, a, saved)
        clean = S.evaluate(model, vl, dev)
        # damage in deployed format: |g| * flip step of that weight's format, g recomputed on the quantized model
        g2 = damage_scores(model, params, vl, dev, a.probe_batches)
        off = 0; d = []
        for n, p, q, sc, fm, high, mid in store:
            seg = g2[off:off + p.numel()].view_as(p.cpu()); off += p.numel()
            d.append((seg * (fm * sc).cpu()).flatten())
        d = torch.cat(d).numpy()
        if thr is None: thr = (float(np.quantile(d, 0.99)), float(np.quantile(d, 0.91)))
        hi, mi = thr; bins = {"HIGH": 100 * float((d >= hi).mean()), "MID": 100 * float(((d >= mi) & (d < hi)).mean()), "LOW": 100 * float((d < mi).mean())}
        avg_bits = sum((torch.log2(fm).sum() + fm.numel()).item() for _, _, _, _, fm, _, _ in store) / sum(p.numel() for _, p in params)   # bits = log2(flip_mag)+1
        # targeted sign-bit flips in the deployed format
        order = torch.from_numpy(np.argsort(-d)); base = [(p, p.detach().clone()) for _, p, *_ in store]; flips = {}
        for nf in nflips:
            with torch.no_grad():
                for p, w0 in base: p.copy_(w0)
                idx = order[:nf].sort().values; off = 0
                for n, p, q, sc, fm, high, mid in store:
                    sel = idx[(idx >= off) & (idx < off + p.numel())] - off; off += p.numel()
                    if sel.numel():
                        qf = q.flatten().clone(); fmf = fm.flatten(); scf = sc.flatten()
                        qf[sel] = torch.where(qf[sel] >= 0, qf[sel] - fmf[sel], qf[sel] + fmf[sel])   # sign-bit flip in that format
                        p.copy_((qf * scf).view_as(p))
            flips[str(nf)] = S.evaluate(model, vl, dev)
        with torch.no_grad():
            for p, w0 in base: p.copy_(w0)
        res[lab] = {"clean": clean, "avg_bits": avg_bits, "bins": bins, "flips": flips}
        print(f"{lab:12s} clean={clean:6.2f}  avg_bits={avg_bits:.3f}  HIGH/MID/LOW={bins['HIGH']:.3f}/{bins['MID']:.2f}/{bins['LOW']:.2f}%  flips: " + "  ".join(f"n={k}:{v:.2f}" for k, v in flips.items()), flush=True)
        del model; torch.cuda.empty_cache()
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(os.path.join(a.out, "eval_mp.json"), "w"), indent=2)
    print(f"\n{'model':12s}{'clean':>8s}{'bits/w':>8s}{'HIGH%':>8s}{'MID%':>8s}" + "".join(f"{'n='+str(n):>9s}" for n in nflips))
    for lab in labels:
        r = res[lab]; print(f"{lab:12s}{r['clean']:8.2f}{r['avg_bits']:8.3f}{r['bins']['HIGH']:8.3f}{r['bins']['MID']:8.2f}" + "".join(f"{r['flips'][str(n)]:9.2f}" for n in nflips))


if __name__ == "__main__":
    main()
