#!/usr/bin/env python
"""
lean_train_qat_nvfp4c.py  -  LEAN in NVFP4, third version: the BLOCK SCALE is the lever.
SEPARATE file; imports lean_sensreg and lean_nvfp4 unchanged.  lean_train_qat_nvfp4.py (v1) is untouched.

Why v1 failed (CIFAR-10 ResNet-18, Oct 19): in NVFP4 the damage of a weight is d_i = |g_i| x 2|w_i| (sign flip of its own
value; no shared tensor step).  v1 took the step as a constant, as in INT8, so the penalty could only push on |g_i|,
which raises the training loss and with it every gradient; the measured tail grew with lambda.
v2 (lean_train_qat_nvfp4b.py) used the step 2|w_i| live; it did not move the tail either (held-out HIGH 0.92%).
v3, this file: the step is the weight's OWN largest single-bit change, which for the small codes is an EXPONENT flip
of 2.5 to 4.5 times the BLOCK SCALE, not the sign flip of 2|w_i|.  The block scale is max|w| over the 16-weight block / 6,
so the shared quantity that LEAN shrinks in INT8 (the tensor maximum) exists in NVFP4 at block level.
P_tail = mean of the top --tail-frac of |g_i| x table[code_i] x s_block(w), with s_block differentiable (straight-through
over the E4M3 rounding), so the loss pushes down the maximum of exactly the blocks whose weights are in the tail, and
the gradient of those weights.  Optional --block-w term on the block-scale flip damage |sum_block g_j w_j| as in v2.

usage (sweep in run_lean_nvfp4c_sweep.sh):
  python -u lean_train_qat_nvfp4c.py --dataset cifar10 --arch resnet18 --init artifacts/models/cifar10/resnet18/model_float32.pth \
      --lam 10 --block-w 1 --out artifacts/models/cifar10/resnet18/nvfp4/v2_lam10.pth
"""
import argparse, time, torch, torch.nn as nn, torch.nn.functional as F
import lean_sensreg as S
try:
    import lean_vgg
except Exception:
    pass
import lean_nvfp4 as Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--tail-frac", type=float, default=0.01)
    ap.add_argument("--tail-every", type=int, default=4)
    ap.add_argument("--block-w", type=float, default=0.0, help="weight of the block-scale damage term inside the lambda bracket (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.init)
    print(f"[init] clean acc (fp32) = {S.evaluate(model, vl, dev):.2f}", flush=True)
    params = S.weight_params(model); plist = [p for _, p in params]
    mods = {n.replace(".weight", ""): m for n, m in model.named_modules()}

    def make_fwd(mod):
        def fwd(x):
            w = Q.fq_nvfp4(mod.weight)
            if isinstance(mod, nn.Conv2d): return F.conv2d(x, w, mod.bias, mod.stride, mod.padding, mod.dilation, mod.groups)
            return F.linear(x, w, mod.bias)
        mod.forward = fwd
    for n, _ in params: make_fwd(mods[n.replace(".weight", "")])
    print(f"[nvfp4 v3] acc before fine-tune = {S.evaluate(model, vl, dev):.2f}   lam {a.lam}  block-w {a.block_w}", flush=True)

    def live_step(w):
        """flip step of each weight = table[code] x block scale, with the block scale differentiable (STE over E4M3)"""
        code, eff, s_t, s_b = Q.codes_nvfp4(w)
        blocks, pad = Q._blocks(w)
        raw = blocks.abs().amax(dim=1) / 6.0                                   # differentiable block scale
        eff_live = (raw + (s_b * s_t - raw).detach()).view(-1, 1).expand(-1, Q.BLOCK).reshape(-1)
        if pad: eff_live = eff_live[:-pad]
        return Q._flip_table(w.device)[code.long()] * eff_live.view_as(w)

    def penalty(loss):
        grads = torch.autograd.grad(loss, plist, create_graph=True)
        d = torch.cat([(g.abs() * live_step(p)).flatten() for g, p in zip(grads, plist)])       # live in g AND block scale
        k = max(1, int(a.tail_frac * d.numel()))
        pen = torch.topk(d, k, sorted=False).values.mean()
        if a.block_w > 0:
            db = []
            for g, p in zip(grads, plist):
                gw = (g * p).reshape(-1); pad = (-gw.numel()) % Q.BLOCK
                if pad: gw = torch.cat([gw, gw.new_zeros(pad)])
                db.append(gw.view(-1, Q.BLOCK).sum(1).abs())
            db = torch.cat(db); kb = max(1, int(a.tail_frac * db.numel()))
            pen = pen + a.block_w * torch.topk(db, kb, sorted=False).values.mean()
        return pen

    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl)); it = 0; acc = 0.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); ce_s = pen_s = 0.0; seen = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            ce = F.cross_entropy(model(x), y); pen = torch.zeros((), device=dev)
            if a.lam > 0 and it % a.tail_every == 0: pen = a.tail_every * penalty(ce)
            loss = ce + a.lam * pen
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); it += 1
            ce_s += ce.item(); pen_s += pen.item(); seen += 1
            if it % 200 == 0:
                print(f"  it {it}  ce={ce_s/seen:.3f}  pen={pen_s/seen:.3e}  {time.time()-t0:.0f}s", flush=True)
        acc = S.evaluate(model, vl, dev); nb = len(tl)
        print(f"[nvfp4 v3 lam{a.lam:g}] ep {ep+1}/{a.epochs}  ce={ce_s/nb:.3f}  pen={pen_s/nb:.3e}  acc={acc:.2f}  {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), a.out)
    print(f"DONE nvfp4 v3 lam{a.lam:g} block-w {a.block_w}  acc {acc:.2f}  saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
