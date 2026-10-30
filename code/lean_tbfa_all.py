#!/usr/bin/env python
"""
lean_tbfa_all.py  -  T-BFA type I ("all-to-one", Rakin et al. 2021) on per-tensor INT8 models, evaluated on the paper's
HELD-OUT split (lean_sensreg's loaders).  Unlike the one-class-to-target form, this one needs no source class, so it
works on the 10-images-per-class ImageNet held-out split: --n-attack held-out images (all classes) form the attack set,
every other held-out image measures the attack success rate (ASR = share classified as the target).
Protection as in the paper's PBS tables: the top --cov of weights by deployed damage |g| x INT8 sign-flip step on probe
batches are unflippable.  Attack: at each step every sign bit of every unprotected weight is ranked by the first-order
decrease of CE(f(x), target), the real loss of the --topk best candidates is evaluated, the best is committed.
SEPARATE file; imports lean_sensreg unchanged.

usage:
  python -u lean_tbfa_all.py --dataset imagenet --arch resnet50 --imagenet-root ./imagenet-val \
      --ckpt artifacts/models/imagenet/resnet50/model_float32_sensreg_lam40.pth --cov 0.01 --targets 1,100,500 --label r50_lam40
"""
import argparse, time, torch, torch.nn.functional as F
import lean_sensreg as S
try:
    import lean_vgg
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="imagenet"); ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--imagenet-root", default="./imagenet-val"); ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--label", default="")
    ap.add_argument("--cov", type=float, default=0.01, help="fraction of weights under the code (0 = none)"); ap.add_argument("--probe-batches", type=int, default=8)
    ap.add_argument("--n-attack", type=int, default=256); ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--targets", default="1,100,500"); ap.add_argument("--n-flips", type=int, default=30); ap.add_argument("--report", default="10,20,30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", choices=["ce", "logit", "both"], default="ce",
                    help="how the coded weights are chosen: ce = damage on the normal loss (paper tables); logit = max over all classes of "
                         "|d logit_c/dw| x flip step (targeted-aware); both = half the budget each")
    ap.add_argument("--logit-batches", type=int, default=2, help="probe batches for the logit ranking (1000 backward passes each)")
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    _, vl = S.cifar_loaders(a.data, 128, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.ckpt); params = S.weight_params(model); plist = [p for _, p in params]; names = [n for n, _ in params]; model.eval()
    N = sum(p.numel() for p in plist)
    # per-tensor INT8
    q, sc = [], []
    with torch.no_grad():
        for p in plist:
            s = p.abs().max().item() / 127.0 + 1e-12; qi = torch.clamp(torch.round(p / s), -128, 127).to(torch.int16); q.append(qi); sc.append(s); p.copy_(qi.float() * s)
    q_clean = [qi.clone() for qi in q]
    def write(i):
        with torch.no_grad(): plist[i].copy_(q[i].float() * sc[i])
    clean = S.evaluate(model, vl, dev)
    # held-out images: attack set (first n_attack), evaluation set (the rest), kept on CPU
    xs, ys = [], []
    for x, y in vl: xs.append(x); ys.append(y)
    X = torch.cat(xs); Y = torch.cat(ys); g_ = torch.Generator().manual_seed(a.seed); perm = torch.randperm(X.size(0), generator=g_)
    xa = X[perm[:a.n_attack]].to(dev); xe = X[perm[a.n_attack:]]; ye = Y[perm[a.n_attack:]]
    # protected set from the untargeted damage on probe batches (as the paper)
    prot = [torch.zeros(p.numel(), dtype=torch.bool, device=dev) for p in plist]
    if a.cov > 0:
        flat = torch.zeros(N, dtype=torch.bool, device=dev); k_all = max(1, int(a.cov * N))
        k_ce = k_all if a.rank == "ce" else (0 if a.rank == "logit" else k_all // 2)
        if k_ce > 0:
            dsum = None; nb = 0
            for x, y in vl:
                if nb >= a.probe_batches: break
                x, y = x.to(dev), y.to(dev); model.zero_grad(set_to_none=True)
                g = torch.autograd.grad(F.cross_entropy(model(x), y), plist)
                d = torch.cat([(gi.abs() * 128.0 * s).flatten() for gi, s in zip(g, sc)]); dsum = d if dsum is None else dsum + d; nb += 1
            flat[torch.topk(dsum, k_ce).indices] = True; del dsum
        if k_all - k_ce > 0:
            # targeted-aware ranking: max over classes of |d z_c / dw| x flip step, i.e. how much one sign flip can move any single logit
            t0 = time.time(); dmax = torch.zeros(N, device=dev); nb = 0
            for x, _ in vl:
                if nb >= a.logit_batches: break
                x = x.to(dev); out = model(x); C = out.size(1)
                for c in range(C):
                    g = torch.autograd.grad(out[:, c].sum(), plist, retain_graph=(c < C - 1))
                    torch.maximum(dmax, torch.cat([(gi.abs() * 128.0 * s).flatten() for gi, s in zip(g, sc)]) / x.size(0), out=dmax)
                nb += 1; print(f"[{a.label}] logit ranking batch {nb}/{a.logit_batches}  {time.time()-t0:.0f}s", flush=True)
            dmax[flat] = -1.0; flat[torch.topk(dmax, k_all - k_ce).indices] = True; del dmax
        off = 0
        for i, p in enumerate(plist): prot[i] = flat[off:off + p.numel()]; off += p.numel()
        per = {}
        for n_, pr in zip(names, prot): per[n_.split(".")[0]] = per.get(n_.split(".")[0], 0) + int(pr.sum())
        print(f"[{a.label}] coded weights by stage ({a.rank}): " + ", ".join(f"{k} {v}" for k, v in per.items()), flush=True)

    @torch.no_grad()
    def asr(t):
        hit = 0; n = 0
        for i in range(0, xe.size(0), 256):
            xb = xe[i:i + 256].to(dev); yb = ye[i:i + 256]; m = (yb != t).to(dev)          # images not already of class t
            hit += ((model(xb).argmax(1) == t) & m).sum().item(); n += int(m.sum())
        return 100.0 * hit / max(n, 1)

    print(f"[{a.label or a.ckpt}] INT8 clean acc {clean:.2f}  held-out {X.size(0)} images: {a.n_attack} attack, {xe.size(0)} evaluation  code on {100*a.cov:.1f}% of {N} weights", flush=True)
    report = [int(x) for x in a.report.split(",")]; rows = []
    for t in [int(x) for x in a.targets.split(",")]:
        for i in range(len(q)): q[i] = q_clean[i].clone(); write(i)
        tl = torch.full((xa.size(0),), t, device=dev, dtype=torch.long); t0 = time.time(); hist = {}; asr0 = asr(t)
        for n in range(1, a.n_flips + 1):
            model.zero_grad(set_to_none=True); loss = F.cross_entropy(model(xa), tl); g = torch.autograd.grad(loss, plist); cands = []
            for i, p in enumerate(plist):
                qi = q[i].reshape(-1); gi = g[i].reshape(-1)
                delta = torch.where(qi >= 0, -128.0, 128.0) * sc[i]                     # sign flip in two's complement
                score = (-gi * delta).masked_fill(prot[i], float("-inf"))                # first-order DEcrease of the targeted loss
                k = min(a.topk, score.numel()); v, ix = torch.topk(score, k); cands += [(float(v[j]), i, int(ix[j])) for j in range(k)]
            cands.sort(key=lambda z: -z[0]); cands = cands[:a.topk]; best = None
            with torch.no_grad():
                for _, i, ix in cands:
                    old = int(q[i].reshape(-1)[ix]); q[i].reshape(-1)[ix] = old - 128 if old >= 0 else old + 128; write(i)
                    L = F.cross_entropy(model(xa), tl).item()
                    if best is None or L < best[0]: best = (L, i, ix)
                    q[i].reshape(-1)[ix] = old; write(i)
            L, i, ix = best; old = int(q[i].reshape(-1)[ix]); q[i].reshape(-1)[ix] = old - 128 if old >= 0 else old + 128; write(i)
            print(f"[{a.label}] target {t}  flip {n:3d} in {names[i]} (q {old} -> {int(q[i].reshape(-1)[ix])})", flush=True)
            if n in report:
                hist[n] = asr(t); print(f"[{a.label}] target {t}  flips {n:3d}  ASR {hist[n]:6.2f}  targeted loss {L:.3f}  {time.time()-t0:.0f}s", flush=True)
        acc = S.evaluate(model, vl, dev); rows.append((t, asr0, hist, acc))
        print(f"[{a.label}] target {t}: ASR {asr0:.2f} -> {hist[report[-1]]:.2f} at {report[-1]} flips; accuracy after {acc:.2f}", flush=True)
    for i in range(len(q)): q[i] = q_clean[i].clone(); write(i)
    mean = {r: sum(h[r] for _, _, h, _ in rows) / len(rows) for r in report}
    print(f"RESULT {a.label} cov={a.cov} clean={clean:.2f} targets={a.targets} " + " ".join(f"ASR@{r}:{mean[r]:.2f}" for r in report)
          + f" acc_after:{sum(x[3] for x in rows)/len(rows):.2f}", flush=True)


if __name__ == "__main__":
    main()
