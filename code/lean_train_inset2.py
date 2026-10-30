#!/usr/bin/env python
"""
lean_train_inset2.py  -  ablations of LEAN-in-the-last-epochs, all starting from the saved epoch-190 checkpoint
of lean_train_inset.py (plain_e190.pth, with optimizer and scheduler state).  SEPARATE file; imports lean_sensreg unchanged.

variants (--variant):
  heldlr      : last 10 epochs, LEAN lam 40, learning rate HELD at 0.002 (the fine-tune's rate) instead of decaying to 0
  heldlr_ctrl : last 10 epochs, plain CE, learning rate held at 0.002 (control for heldlr)
  long20      : last 20 epochs, LEAN lam 40, cosine from the epoch-190 rate (0.0006) to 0 over 20 epochs
  long20_ctrl : last 20 epochs, plain CE, same schedule (control for long20)
  lam100      : last 10 epochs, LEAN lam 100, the original schedule (rate 0.0005 to 0)

usage:
  CUDA_VISIBLE_DEVICES=2 python -u lean_train_inset2.py --variant heldlr --out artifacts/models/cifar10/resnet18/inset
"""
import argparse, os, time, torch, torch.nn.functional as F
import lean_sensreg as S
try:
    import lean_vgg
except Exception:
    pass


def run_epochs(model, tl, vl, opt, sched, n_ep, dev, params, lam, a, tag):
    for ep in range(n_ep):
        model.train(); t0 = time.time(); ce_sum = pen_sum = 0.0
        for it, (x, y) in enumerate(tl):
            x, y = x.to(dev), y.to(dev)
            ce = F.cross_entropy(model(x), y); pen = torch.zeros((), device=dev)
            if lam > 0:
                if it % a.tail_every == 0:
                    pen = pen + S.tail_penalty(ce, params, a.tail_frac) * a.tail_every
                pen = pen + a.clip_w * S.clip_penalty(params, a.clip_q)
            loss = ce + lam * pen
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if sched is not None: sched.step()
            ce_sum += ce.item(); pen_sum += pen.item()
        acc = S.evaluate(model, vl, dev); lr = opt.param_groups[0]["lr"]
        print(f"[{tag}] ep {ep+1}/{n_ep}  ce={ce_sum/len(tl):.3f}  pen={pen_sum/len(tl):.3e}  acc={acc:.2f}  lr={lr:.4f}  {time.time()-t0:.0f}s", flush=True)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imagenet-root", default="./imagenet-val"); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--variant", required=True, choices=["heldlr", "heldlr_ctrl", "long20", "long20_ctrl", "lam100"])
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
    if v in ("heldlr", "heldlr_ctrl"):
        opt = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4); sched = None; n_ep = 10
        lam = 40.0 if v == "heldlr" else 0.0
    elif v in ("long20", "long20_ctrl"):
        opt = torch.optim.SGD(model.parameters(), lr=0.0006, momentum=0.9, weight_decay=5e-4); n_ep = 20
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_ep * len(tl)); lam = 40.0 if v == "long20" else 0.0
    else:  # lam100: original schedule, resumed
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4); opt.load_state_dict(st["opt"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200 * len(tl)); sched.load_state_dict(st["sched"]); n_ep = 10; lam = 100.0

    acc = run_epochs(model, tl, vl, opt, sched, n_ep, dev, params, lam, a, v)
    out = os.path.join(a.out, f"{v}.pth"); torch.save(model.state_dict(), out)
    print(f"DONE {v}  acc {acc:.2f}  saved {out}", flush=True)


if __name__ == "__main__":
    main()
