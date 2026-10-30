#!/usr/bin/env python
"""
lean_ecc_four.py  -  the four protection conditions under ONE attack, for a baseline and a lambda model.
SEPARATE file; imports from lean_sensreg.py (nothing there is changed).

Conditions (ECC modeled as perfect: a protected weight cannot be flipped):
  none      : no ECC
  uniform   : --cov fraction of weights protected, chosen at RANDOM (conventional ECC, no sensitivity knowledge)
  selcode     : --cov fraction protected, chosen by SENSITIVITY (top damage score), i.e. the selective code's choice
Attack: progressive targeted sign-bit flips on the INT8 model (PBS style): in rounds, recompute damage on the
unprotected weights, flip the most damaging --round bits, until --nflips. Reports accuracy after each checkpoint.

Usage:
  python lean_ecc_four.py --dataset cifar10 --arch resnet18 --cov 0.05 --nflips 10,25,50 \
      --ckpts artifacts/models/cifar10/resnet18/model_float32.pth artifacts/models/cifar10/resnet18/model_float32_sensreg_lam10.pth \
      --labels baseline lam10 --out sensreg_results/cifar10_resnet18/ecc_four
"""
import argparse, json, os
import numpy as np, torch, torch.nn.functional as F
import lean_sensreg as S


def damage(model, params, batches, dev):
    plist = [p for _, p in params]; acc = None
    for x, y in batches:
        loss = F.cross_entropy(model(x), y); g = torch.autograd.grad(loss, plist)
        d = torch.cat([(gg.abs() * S.msb_step(p)).flatten() for gg, p in zip(g, plist)])
        acc = d if acc is None else acc + d; model.zero_grad(set_to_none=True)
    return acc / len(batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpts", nargs="+", required=True); ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--cov", type=float, default=0.05, help="fraction of weights protected (uniform and selcode)")
    ap.add_argument("--nflips", default="10,25,50"); ap.add_argument("--round", type=int, default=5)
    ap.add_argument("--probe-batches", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ecc_four")
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed); np.random.seed(a.seed)
    _, vl = S.cifar_loaders(a.data, 128, a.workers); nflips = [int(x) for x in a.nflips.split(",")]
    batches = []
    for i, (x, y) in enumerate(vl):
        if i >= a.probe_batches: break
        batches.append((x.to(dev), y.to(dev)))
    labels = a.labels or [os.path.basename(c) for c in a.ckpts]; res = {}
    for c, lab in zip(a.ckpts, labels):
        res[lab] = {}
        for cond in ("none", "uniform", "selcode"):
            model = S.build_model(a.arch).to(dev); S.load_state_flex(model, c); params = S.weight_params(model)
            store = S.quantize_model_(model, 8)                                  # INT8 per-tensor, in place
            n_total = sum(p.numel() for _, p in params); clean = S.evaluate(model, vl, dev)
            d0 = damage(model, params, batches, dev)
            k = int(a.cov * n_total)
            if cond == "none": prot = torch.zeros(n_total, dtype=torch.bool, device=dev)
            elif cond == "uniform": prot = torch.zeros(n_total, dtype=torch.bool, device=dev); prot[torch.randperm(n_total, device=dev)[:k]] = True
            else: prot = torch.zeros(n_total, dtype=torch.bool, device=dev); prot[torch.topk(d0, k).indices] = True
            # progressive attack
            flipped = torch.zeros(n_total, dtype=torch.bool, device=dev); done = 0; accs = {}
            sizes = [p.numel() for _, p in params]; offs = np.cumsum([0] + sizes)
            while done < max(nflips):
                d = damage(model, params, batches, dev); d[prot | flipped] = -1
                sel = torch.topk(d, min(a.round, max(nflips) - done)).indices; flipped[sel] = True
                with torch.no_grad():
                    for i, (n, p, q, scale) in enumerate(store):
                        s = sel[(sel >= offs[i]) & (sel < offs[i + 1])] - offs[i]
                        if s.numel():
                            qf = q.flatten(); qf[s] = torch.where(qf[s] >= 0, qf[s] - 128, qf[s] + 128); p.copy_(qf.view_as(p) * scale)
                done += sel.numel()
                if done in nflips: accs[str(done)] = S.evaluate(model, vl, dev)
            res[lab][cond] = {"clean": clean, "protected_frac": float(prot.float().mean()), "acc": accs}
            print(f"{lab:10s} {cond:8s} clean={clean:6.2f} protected={100*prot.float().mean():5.1f}%  " + "  ".join(f"n={k}:{v:.2f}" for k, v in accs.items()), flush=True)
            del model; torch.cuda.empty_cache()
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(os.path.join(a.out, "ecc_four.json"), "w"), indent=2)
    print(f"\n{'model':10s}{'condition':10s}{'protected':>10s}" + "".join(f"{'n='+str(n):>9s}" for n in nflips))
    for lab in labels:
        for cond in ("none", "uniform", "selcode"):
            r = res[lab][cond]; print(f"{lab:10s}{cond:10s}{100*r['protected_frac']:9.1f}%" + "".join(f"{r['acc'][str(n)]:9.2f}" for n in nflips))


if __name__ == "__main__":
    main()
