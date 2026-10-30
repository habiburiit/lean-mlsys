#!/usr/bin/env python
"""
lean_train_inset3.py  -  two more last-epochs variants, from the saved epoch-190 checkpoint (plain_e190.pth).
Reuses run_epochs from lean_train_inset2.py unchanged.

variants (--variant):
  heldlr_cos    : last 10 epochs, LEAN lam 40, lr 0.002 cosine to 0 (the fine-tune's schedule, inside training)
  heldlr20      : last 20 epochs, LEAN lam 40, lr held at 0.002
  heldlr20_ctrl : last 20 epochs, plain CE, lr held at 0.002 (control for heldlr20)

usage:
  CUDA_VISIBLE_DEVICES=2 python -u lean_train_inset3.py --variant heldlr_cos --out artifacts/models/cifar10/resnet18/inset
"""
import argparse, os, torch
import lean_sensreg as S
from lean_train_inset2 import run_epochs
try:
    import lean_vgg
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imagenet-root", default="./imagenet-val"); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--variant", required=True, choices=["heldlr_cos", "heldlr20", "heldlr20_ctrl"])
    ap.add_argument("--tail-frac", type=float, default=0.01); ap.add_argument("--tail-every", type=int, default=4)
    ap.add_argument("--clip-q", type=float, default=0.99); ap.add_argument("--clip-w", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="artifacts/models/cifar10/resnet18/inset")
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda"; torch.manual_seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    pre = os.path.join(a.out, "plain_e190.pth"); st = torch.load(pre, map_location=dev)
    model = S.build_model(a.arch).to(dev); model.load_state_dict(st["model"]); params = S.weight_params(model)
    print(f"[start] {pre}  acc {S.evaluate(model, vl, dev):.2f}", flush=True)

    v = a.variant
    opt = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)
    if v == "heldlr_cos":
        n_ep = 10; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_ep * len(tl)); lam = 40.0
    else:
        n_ep = 20; sched = None; lam = 40.0 if v == "heldlr20" else 0.0

    acc = run_epochs(model, tl, vl, opt, sched, n_ep, dev, params, lam, a, v)
    out = os.path.join(a.out, f"{v}.pth"); torch.save(model.state_dict(), out)
    print(f"DONE {v}  acc {acc:.2f}  saved {out}", flush=True)


if __name__ == "__main__":
    main()
