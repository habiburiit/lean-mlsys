#!/usr/bin/env python
"""
lean_train_inset.py  -  LEAN switched on during the LAST epochs of normal training, instead of a separate fine-tune.
SEPARATE file; imports lean_sensreg unchanged.

One run from scratch: epochs 1..E-K plain cross-entropy (cosine LR from --lr), checkpoint at E-K, then two branches
that continue the SAME schedule for the last K epochs:
   control : plain CE                          -> <out>/control_e{E}.pth
   inset   : CE + lam*(P_tail + 100*P_clip)    -> <out>/inset_lam{lam}_e{E}.pth
Compare afterwards with:
   python lean_sensreg.py hist --dataset cifar10 --arch resnet18 --ckpt <out>/control_e200.pth --ckpt2 <out>/inset_lam40_e200.pth --out <out>/hist

usage:
  CUDA_VISIBLE_DEVICES=2 python -u lean_train_inset.py --dataset cifar10 --arch resnet18 --epochs 200 --last 10 --lam 40 --lr 0.1 --bs 128 \
      --out artifacts/models/cifar10/resnet18/inset
"""
import argparse, copy, os, time, torch, torch.nn.functional as F
import lean_sensreg as S
try:
    import lean_vgg
except Exception:
    pass


def run_epochs(model, tl, vl, opt, sched, n_ep, dev, params, lam, tail_frac, tail_every, clip_q, clip_w, tag):
    for ep in range(n_ep):
        model.train(); t0 = time.time(); ce_sum = pen_sum = 0.0
        for it, (x, y) in enumerate(tl):
            x, y = x.to(dev), y.to(dev)
            ce = F.cross_entropy(model(x), y); pen = torch.zeros((), device=dev)
            if lam > 0:
                if it % tail_every == 0:
                    pen = pen + S.tail_penalty(ce, params, tail_frac) * tail_every
                pen = pen + clip_w * S.clip_penalty(params, clip_q)
            loss = ce + lam * pen
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            ce_sum += ce.item(); pen_sum += pen.item()
        acc = S.evaluate(model, vl, dev)
        print(f"[{tag}] ep {ep+1}/{n_ep}  ce={ce_sum/len(tl):.3f}  pen={pen_sum/len(tl):.3e}  acc={acc:.2f}  lr={sched.get_last_lr()[0]:.4f}  {time.time()-t0:.0f}s", flush=True)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lam", type=float, default=40.0); ap.add_argument("--tail-frac", type=float, default=0.01)
    ap.add_argument("--tail-every", type=int, default=4); ap.add_argument("--clip-q", type=float, default=0.99)
    ap.add_argument("--clip-w", type=float, default=100.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="inset")
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda"; torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    E, K = a.epochs, a.last

    model = S.build_model(a.arch).to(dev); params = S.weight_params(model)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E * len(tl))

    # phase 1: plain training up to epoch E-K
    pre = os.path.join(a.out, f"plain_e{E-K}.pth")
    if os.path.exists(pre):
        st = torch.load(pre, map_location=dev); model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        print(f"[resume] loaded {pre}", flush=True)
    else:
        run_epochs(model, tl, vl, opt, sched, E - K, dev, params, 0.0, a.tail_frac, a.tail_every, a.clip_q, a.clip_w, "plain")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict()}, pre)
        print(f"[saved] {pre}", flush=True)
    snap = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), copy.deepcopy(sched.state_dict()))

    # branch a: control, last K epochs plain
    acc_c = run_epochs(model, tl, vl, opt, sched, K, dev, params, 0.0, a.tail_frac, a.tail_every, a.clip_q, a.clip_w, "control")
    torch.save(model.state_dict(), os.path.join(a.out, f"control_e{E}.pth"))

    # branch b: LEAN term on for the last K epochs, same schedule
    model.load_state_dict(snap[0]); opt.load_state_dict(snap[1]); sched.load_state_dict(snap[2])
    acc_i = run_epochs(model, tl, vl, opt, sched, K, dev, params, a.lam, a.tail_frac, a.tail_every, a.clip_q, a.clip_w, f"inset lam{a.lam:g}")
    torch.save(model.state_dict(), os.path.join(a.out, f"inset_lam{a.lam:g}_e{E}.pth"))
    print(f"\nDONE  control acc {acc_c:.2f}   inset acc {acc_i:.2f}   ckpts in {a.out}", flush=True)


if __name__ == "__main__":
    main()
