#!/usr/bin/env python
"""
lean_tbfa.py  -  targeted bit-flip attack (T-BFA, Rakin et al. 2021, 1-to-1 variant) on INT8 models,
with and without a 1% sensitivity-aware code.  SEPARATE file; imports lean_sensreg (unchanged).

Attack: pick a source class s and a target class t.  Attack loss
    L = CE(f(x_s), t) + CE(f(x_o), y_o)
(source images pushed to t, other images kept on their labels).  Progressive search, as PBS: every step rank
candidate sign-bit flips by first-order gain, evaluate the top-k candidates by the real attack loss, commit the
best one.  Stops at --max-flips or when the attack success rate (ASR, share of held-out source-class images
classified as t) reaches --asr-stop.

Protection (--protect selcode --cov 0.01): the top 1% of weights by sign-flip damage |g|*dw on the attack data of
THIS model are covered by the BCH code; a sign flip on a covered weight is corrected, so the adaptive attacker
only searches the rest.  (One sign plane, 12 weights per codeword, t=12: sign flips can never overload a word.)

Output per (s,t) pair: flips used, ASR at 10/20/30 flips, accuracy on the non-source test images at the end.
Usage:
  python lean_tbfa.py --dataset cifar10 --arch resnet18 --ckpt artifacts/models/cifar10/resnet18/model_float32.pth --label base --protect none
  python lean_tbfa.py --dataset cifar10 --arch resnet18 --ckpt artifacts/models/cifar10/resnet18/model_float32_sensreg_lam10.pth --label lam10 --protect selcode --cov 0.01
"""
import argparse, csv, os, time
import numpy as np, torch, torch.nn.functional as F
import lean_sensreg as S
try:
    import lean_vgg          # registers vgg* in S.build_model; harmless for ResNets
except Exception as _e:
    print(f"[warn] lean_vgg not importable ({_e}); vgg archs unavailable", flush=True)


def quantize_int8(params):
    """per-tensor symmetric INT8; returns flat codes and per-weight scales; writes the quantized values back"""
    qs, ss = [], []
    with torch.no_grad():
        for _, p in params:
            s = p.abs().max() / 127.0
            q = torch.round(p / s).clamp(-128, 127)
            p.copy_(q * s); qs.append(q.flatten()); ss.append(torch.full((q.numel(),), float(s), device=p.device))
    return torch.cat(qs), torch.cat(ss)


def collect(vl, dev, src, n_src, n_other, n_eval_src):
    """attack set: n_src source-class images + n_other others; evaluation set: n_eval_src more source images"""
    xs, xo, yo, xe = [], [], [], []
    for x, y in vl:
        m = y == src
        xs.append(x[m]); xo.append(x[~m]); yo.append(y[~m])
        if sum(a.size(0) for a in xs) >= n_src + n_eval_src and sum(a.size(0) for a in xo) >= n_other: break
    xs = torch.cat(xs); xo = torch.cat(xo)[:n_other]; yo = torch.cat(yo)[:n_other]
    if xs.size(0) < n_src + n_eval_src:                       # small classes (CIFAR-100: 100 test images)
        n_src = xs.size(0) // 2; n_eval_src = xs.size(0) - n_src
        print(f"   [data] source class has {xs.size(0)} test images: {n_src} for the attack, {n_eval_src} for ASR", flush=True)
    xa, xe = xs[:n_src], xs[n_src:n_src + n_eval_src]
    return xa.to(dev), xo.to(dev), yo.to(dev), xe.to(dev)


@torch.no_grad()
def asr(model, xe, tgt, bs=256):
    hit = 0
    for i in range(0, xe.size(0), bs): hit += (model(xe[i:i+bs]).argmax(1) == tgt).sum().item()
    return 100.0 * hit / xe.size(0)


@torch.no_grad()
def acc_except(model, vl, dev, src):
    c = n = 0
    for x, y in vl:
        x, y = x.to(dev), y.to(dev); m = y != src
        c += (model(x[m]).argmax(1) == y[m]).sum().item(); n += int(m.sum())
    return 100.0 * c / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--label", default=None)
    ap.add_argument("--protect", choices=["none", "selcode"], default="none")
    ap.add_argument("--cov", type=float, default=0.01, help="coverage fraction of all weights for --protect selcode")
    ap.add_argument("--rank", choices=["mixed", "classmax"], default="mixed",
                    help="defender's ranking: mixed = damage of the true-label loss on a mixed batch; classmax = per weight, the maximum damage over all target classes (gradient of pushing the batch to class c), which covers class-steering weights")
    ap.add_argument("--pairs", default="0:1,3:5,7:2", help="source:target class pairs")
    ap.add_argument("--n-src", type=int, default=128); ap.add_argument("--n-other", type=int, default=256)
    ap.add_argument("--n-eval-src", type=int, default=512)
    ap.add_argument("--max-flips", type=int, default=30); ap.add_argument("--report", default="10,20,30")
    ap.add_argument("--asr-stop", type=float, default=95.0)
    ap.add_argument("--topk", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-val", action="store_true", help="imagenet: use all 50k val images (50 per class) instead of the 10k held-out split")
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab = a.label or os.path.basename(a.ckpt).replace(".pth", "")
    report = [int(x) for x in a.report.split(",")]

    _, vl = S.cifar_loaders(a.data, 128, a.workers)
    if a.dataset == "imagenet" and a.full_val:
        import torchvision, torchvision.transforms as T
        te = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        vl = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(a.imagenet_root, te), 128, shuffle=False, num_workers=a.workers, pin_memory=True)
        print("[data] imagenet: all 50k val images (50 per class)", flush=True)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.ckpt); model.eval()
    params = S.weight_params(model); plist = [p for _, p in params]
    q_all, sc_all = quantize_int8(params); n_total = q_all.numel()
    sizes = [p.numel() for p in plist]; offs = np.cumsum([0] + sizes)
    clean = S.evaluate(model, vl, dev)
    print(f"[{lab}] INT8 clean={clean:.2f}%  protect={a.protect}  cov={a.cov if a.protect!='none' else 0}  rank={a.rank}", flush=True)

    def write(q_flat):
        with torch.no_grad():
            for i, p in enumerate(plist):
                p.copy_((q_flat[offs[i]:offs[i+1]] * sc_all[offs[i]:offs[i+1]]).view_as(p))

    def grad_flat(loss_fn):
        model.zero_grad(set_to_none=True); loss_fn().backward()
        g = torch.cat([p.grad.flatten() for p in plist]); model.zero_grad(set_to_none=True); return g

    def flip(q_flat, idx):
        qf = q_flat.clone(); v = qf[idx]; qf[idx] = torch.where(v >= 0, v - 128, v + 128); return qf

    def flip_val(q_flat, idx):
        v = float(q_flat[idx]); return v - 128.0 if v >= 0 else v + 128.0

    def set_one(idx, qval):                                     # write one weight only (fast candidate test)
        j = int(np.searchsorted(offs, idx, side="right") - 1)
        with torch.no_grad(): plist[j].view(-1)[idx - offs[j]] = qval * float(sc_all[idx])

    # protection: top --cov by damage on a mixed batch (defender's own ranking of this model)
    prot = torch.zeros(n_total, dtype=torch.bool, device=dev)
    if a.protect == "selcode":
        xb, yb = next(iter(vl)); xb, yb = xb.to(dev), yb.to(dev)
        g0 = grad_flat(lambda: F.cross_entropy(model(xb), yb))
        dmg = g0.abs() * 128.0 * sc_all
        if a.rank == "classmax":
            n_cls = {"cifar10": 10, "cifar100": 100, "imagenet": 1000}[a.dataset]
            for c in range(n_cls):
                tc = torch.full_like(yb, c)
                gc = grad_flat(lambda: F.cross_entropy(model(xb), tc))
                dmg = torch.maximum(dmg, gc.abs() * 128.0 * sc_all)
            print(f"[{lab}] classmax ranking over {n_cls} classes", flush=True)
        k = max(1, int(a.cov * n_total))
        prot[torch.topk(dmg, k).indices] = True
        for j, (n, p) in enumerate(params):                       # how much of each layer the code covers (top few)
            fr = 100.0 * float(prot[offs[j]:offs[j+1]].sum()) / sizes[j]
            if fr > 20: print(f"[{lab}] covered {fr:5.1f}% of {n} ({sizes[j]} weights)", flush=True)
    cov = 100.0 * float(prot.sum()) / n_total
    names = [n for n, _ in params]

    rows = []
    for pair in a.pairs.split(","):
        src, tgt = [int(v) for v in pair.split(":")]
        xa, xo, yo, xe = collect(vl, dev, src, a.n_src, a.n_other, a.n_eval_src)
        tl = torch.full((xa.size(0),), tgt, device=dev, dtype=torch.long)
        attack_loss = lambda: F.cross_entropy(model(xa), tl) + F.cross_entropy(model(xo), yo)
        q_cur = q_all.clone(); write(q_cur); flipped = torch.zeros(n_total, dtype=torch.bool, device=dev)
        a0 = asr(model, xe, tgt); hist = {}; t0 = time.time(); used = 0; cur = a0
        for step in range(1, a.max_flips + 1):
            g = grad_flat(attack_loss)
            delta = torch.where(q_cur >= 0, -128.0, 128.0) * sc_all        # weight change of a sign flip
            gain = -(g * delta)                                            # first-order DEcrease of the attack loss
            gain[prot] = -float("inf"); gain[flipped] = -float("inf")
            cand = torch.topk(gain, a.topk).indices
            best_i, best_l = None, float("inf")
            with torch.no_grad():
                for i in cand.tolist():
                    set_one(i, flip_val(q_cur, i)); l = attack_loss().item(); set_one(i, float(q_cur[i]))
                    if l < best_l: best_l, best_i = l, i
            q_cur[best_i] = flip_val(q_cur, best_i); flipped[best_i] = True; set_one(best_i, float(q_cur[best_i])); used = step
            cur = asr(model, xe, tgt)
            if step in report: hist[step] = cur
            lj = int(np.searchsorted(offs, best_i, side="right") - 1)
            print(f"   [{lab}] {a.protect:5s} {src}->{tgt} flips={step:3d} loss={best_l:.3f} ASR={cur:5.1f}% {names[lj]} ({time.time()-t0:.0f}s)", flush=True)
            if cur >= a.asr_stop: break
        for r in report:
            if r not in hist: hist[r] = cur if r >= used else hist.get(r, cur)
        acc_rest = acc_except(model, vl, dev, src)
        rows.append((src, tgt, a0, used, cur, hist, acc_rest))
        print(f"[{lab}] {src}->{tgt}: ASR {a0:.1f}% -> {cur:.1f}% with {used} flips; accuracy on other classes {acc_rest:.2f}%", flush=True)
        write(q_all)

    print(f"\n{'model':10s}{'protect':8s}{'cov':>7s}{'pair':>7s}{'flips':>7s}{'ASR_end':>9s}" + "".join(f"{'ASR@'+str(r):>8s}" for r in report) + f"{'acc_rest':>10s}")
    for src, tgt, a0, used, cur, hist, acc_rest in rows:
        print(f"{lab:10s}{a.protect:8s}{cov:6.2f}%{f'{src}->{tgt}':>7s}{used:7d}{cur:9.1f}" + "".join(f"{hist[r]:8.1f}" for r in report) + f"{acc_rest:10.2f}")
    mean_asr = np.mean([r[4] for r in rows]); mean_fl = np.mean([r[3] for r in rows])
    print(f"{lab:10s}{a.protect:8s}  mean: ASR {mean_asr:.1f}%  flips {mean_fl:.1f}")
    os.makedirs("sensreg_results/tbfa", exist_ok=True)
    sfx = "" if a.rank == "mixed" else "_" + a.rank
    with open(f"sensreg_results/tbfa/{a.dataset}_{a.arch}_{lab}_{a.protect}{sfx}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "protect", "cov_pct", "src", "tgt", "asr0", "flips", "asr_end"] + [f"asr_{r}" for r in report] + ["acc_rest", "clean"])
        for src, tgt, a0, used, cur, hist, acc_rest in rows:
            w.writerow([lab, a.protect, f"{cov:.2f}", src, tgt, f"{a0:.1f}", used, f"{cur:.1f}"] + [f"{hist[r]:.1f}" for r in report] + [f"{acc_rest:.2f}", f"{clean:.2f}"])


if __name__ == "__main__":
    main()
