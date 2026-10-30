#!/usr/bin/env python
"""
lean_train_hessian.py  -  baseline: Hessian-aware training (Pranto, Kim, Chen, Hong, arXiv 2504.01933, 2025).
SEPARATE file; imports from lean_sensreg.py (nothing there is changed). Same data, epochs, lr and init as our
lambda training, so the comparison is fair.

    L = CE + alpha * Tr(H)      Tr(H) estimated with Hutchinson (p Rademacher probes, v^T H v by double backprop)
Their stabilisation tricks are kept: regularise only when the current trace exceeds the running median of
previous traces (--gate 1), and optionally compute the trace on the last --last-layers layers only (their ImageNet
setting; 0 = all layers). Their alpha: 1e-2 for CIFAR-10 ResNet-18, 1e-3 for ImageNet.

Usage:
  python lean_train_hessian.py --dataset cifar10 --arch resnet18 --init artifacts/models/cifar10/resnet18/model_float32.pth \
      --alpha 1e-2 --epochs 10 --out artifacts/models/cifar10/resnet18/model_float32_hessian.pth
  python lean_train_hessian.py --dataset imagenet --arch resnet50 --imagenet-root <path> --init pretrained \
      --alpha 1e-3 --last-layers 1 --lr 0.0005 --bs 32 --epochs 10 --out artifacts/models/imagenet/resnet50/model_float32_hessian.pth
"""
import argparse, time, statistics, torch, torch.nn.functional as F
import lean_sensreg as S


def hutchinson_trace(loss, plist, probes):
    grads = torch.autograd.grad(loss, plist, create_graph=True)
    tr = 0.0
    for _ in range(probes):
        vs = [torch.randint_like(p, 2) * 2 - 1 for p in plist]          # Rademacher
        gv = sum((g * v).sum() for g, v in zip(grads, vs))
        hv = torch.autograd.grad(gv, plist, retain_graph=True)
        tr = tr + sum((h * v).sum() for h, v in zip(hv, vs))
    return tr / probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--alpha", type=float, default=1e-2, help="Hessian-trace regulariser weight (their alpha)")
    ap.add_argument("--probes", type=int, default=1, help="Hutchinson probes per step (they found 1 best)")
    ap.add_argument("--every", type=int, default=2, help="apply the trace term every N steps (cost control; scaled by N)")
    ap.add_argument("--gate", type=int, default=1, help="1 = regularise only when trace > running median (their trick)")
    ap.add_argument("--last-layers", type=int, default=0, help="compute the trace on the last N weight layers only (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.init)
    print(f"[init] clean acc = {S.evaluate(model, vl, dev):.2f}   Hessian-aware baseline, alpha={a.alpha}, last_layers={a.last_layers}")
    params = S.weight_params(model); plist = [p for _, p in params]
    if a.last_layers > 0: plist = plist[-a.last_layers:]
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl)); hist = []; it = 0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); ce_s = tr_s = 0.0; n_reg = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            ce = F.cross_entropy(model(x), y); loss = ce
            if it % a.every == 0:
                tr = hutchinson_trace(ce, plist, a.probes); trv = tr.item(); tr_s += trv
                use = (not a.gate) or (len(hist) < 10) or (trv > statistics.median(hist[-200:]))
                hist.append(trv)
                if use: loss = loss + a.alpha * a.every * tr; n_reg += 1
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); it += 1; ce_s += ce.item()
        acc = S.evaluate(model, vl, dev); nb = len(tl)
        print(f"ep {ep+1}/{a.epochs}  ce={ce_s/nb:.3f}  trace={tr_s/max(1,nb//a.every):.3e}  regularised_steps={n_reg}  acc={acc:.2f}  {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), a.out); print(f"[saved] {a.out}  final acc {acc:.2f}")


if __name__ == "__main__":
    main()
