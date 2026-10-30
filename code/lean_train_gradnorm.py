#!/usr/bin/env python
"""
lean_train_gradnorm.py  -  the plain gradient-norm baseline for the reviewer: same fine-tune as lean_sensreg.py train
(10 epochs, lr 0.002 cosine, from the clean checkpoint) but the penalty is the squared weight-gradient norm
sum_i g_i^2 over all weight tensors (double backprop), applied every 4th step scaled by 4 like the LEAN tail term.
No tail selection, no storage-format weighting, no range term.  SEPARATE file; imports lean_sensreg unchanged.

usage:
  CUDA_VISIBLE_DEVICES=2 python -u lean_train_gradnorm.py --dataset cifar10 --arch resnet18 \
      --init artifacts/models/cifar10/resnet18/model_float32.pth --lam 10 --out artifacts/models/cifar10/resnet18/gradnorm_lam10.pth
"""
import argparse, time, torch, torch.nn.functional as F
import lean_sensreg as S
try:
    import lean_vgg
except Exception:
    pass


def gradnorm_penalty(loss, params):
    grads = torch.autograd.grad(loss, [p for _, p in params], create_graph=True)
    return sum(g.pow(2).sum() for g in grads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imagenet-root", default="./imagenet-val"); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--lam", type=float, default=10.0); ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda"; torch.manual_seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.init)
    print(f"[init] clean acc = {S.evaluate(model, vl, dev):.2f}", flush=True)
    params = S.weight_params(model)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl))
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); ce_sum = pen_sum = 0.0
        for it, (x, y) in enumerate(tl):
            x, y = x.to(dev), y.to(dev)
            ce = F.cross_entropy(model(x), y); pen = torch.zeros((), device=dev)
            if it % a.every == 0:
                pen = gradnorm_penalty(ce, params) * a.every
            loss = ce + a.lam * pen
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            ce_sum += ce.item(); pen_sum += pen.item()
        acc = S.evaluate(model, vl, dev)
        print(f"[gradnorm lam{a.lam:g}] ep {ep+1}/{a.epochs}  ce={ce_sum/len(tl):.3f}  pen={pen_sum/len(tl):.3e}  acc={acc:.2f}  {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), a.out)
    print(f"DONE gradnorm lam{a.lam:g}  acc {acc:.2f}  saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
