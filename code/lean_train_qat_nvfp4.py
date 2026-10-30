#!/usr/bin/env python
"""
lean_train_qat_nvfp4.py  -  LEAN in the NVFP4 format: quantization-aware fine-tune (NVFP4 weights, E2M1 + FP8 block
scales of 16, see lean_nvfp4.py) from the baseline checkpoint, with the LEAN objective
      L = CE + lambda ( P_tail + 100 P_clip ),
where P_tail is the mean of the top --tail-frac damages d_i = |g_i| Delta_i with Delta_i the weight's OWN NVFP4
single-bit flip step (lean_nvfp4.flip_step_nvfp4), every --tail-every step scaled by --tail-every, and P_clip is the
range term of lean_sensreg with weight --clip-w.  --lam 0 is the NVFP4 control with the identical schedule.
NOTE (CIFAR-10 ResNet-18, Oct 19): in NVFP4 the damage d_i is ~50x smaller than in INT8 (flip step 2|w| instead of the
tensor maximum), so lambda must be ~50x larger for the same push, and the range term does nothing useful in this format
(the block scales, not the tensor maximum, set the grid).  With --lam 40 --clip-w 100 the range term dominated and the
tail GREW (HIGH 1% -> 3.1%).  Default is now --clip-w 0 (tail term only); sweep --lam 400 / 2000 / 10000.
Same schedule as every other LEAN fine-tune (10 epochs, cosine, SGD 0.9).  SEPARATE file; imports lean_sensreg and
lean_nvfp4 unchanged.  Saves the float weights after QAT in <out> (deploy by re-quantizing with lean_nvfp4.codes_nvfp4).

usage:
  CIFAR-10 ResNet-18:  python -u lean_train_qat_nvfp4.py --dataset cifar10 --arch resnet18 --init artifacts/models/cifar10/resnet18/model_float32.pth --lam 40 --out <ckpt>
  ImageNet ResNet-50:  python -u lean_train_qat_nvfp4.py --dataset imagenet --arch resnet50 --imagenet-root <path> --init pretrained --lr 0.0005 --bs 32 --lam 40 --out <ckpt>
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
    ap.add_argument("--tail-every", type=int, default=4); ap.add_argument("--clip-q", type=float, default=0.99)
    ap.add_argument("--clip-w", type=float, default=0.0, help="weight of the range term inside the lambda bracket (100 = INT8 setting; 0 = tail term only, the NVFP4 default)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.init)
    print(f"[init] clean acc (fp32) = {S.evaluate(model, vl, dev):.2f}", flush=True)
    params = S.weight_params(model); plist = [p for _, p in params]
    mods = {n.replace(".weight", ""): m for n, m in model.named_modules()}
    n_total = sum(p.numel() for p in plist)
    bpw = sum(Q.bits_per_weight(p) * p.numel() for p in plist) / n_total

    def make_fwd(mod):
        def fwd(x):
            w = Q.fq_nvfp4(mod.weight)
            if isinstance(mod, nn.Conv2d): return F.conv2d(x, w, mod.bias, mod.stride, mod.padding, mod.dilation, mod.groups)
            return F.linear(x, w, mod.bias)
        mod.forward = fwd
    for n, _ in params: make_fwd(mods[n.replace(".weight", "")])
    print(f"[nvfp4] {n_total} weights in blocks of {Q.BLOCK}: {bpw:.3f} bits/weight incl. block and tensor scales;  "
          f"NVFP4 acc before fine-tune = {S.evaluate(model, vl, dev):.2f}", flush=True)

    def tail_penalty(loss):
        grads = torch.autograd.grad(loss, plist, create_graph=True)
        scores = torch.cat([(g.abs() * Q.flip_step_nvfp4(p)).flatten() for g, p in zip(grads, plist)])
        k = max(1, int(a.tail_frac * scores.numel()))
        return torch.topk(scores, k, sorted=False).values.mean()

    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl)); it = 0; acc = 0.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); ce_s = pen_s = 0.0; seen = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            ce = F.cross_entropy(model(x), y); pen = torch.zeros((), device=dev)
            if a.lam > 0:
                if it % a.tail_every == 0: pen = pen + a.tail_every * tail_penalty(ce)
                if a.clip_w > 0: pen = pen + a.clip_w * S.clip_penalty(params, a.clip_q)
            loss = ce + a.lam * pen
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); it += 1
            ce_s += ce.item(); pen_s += pen.item(); seen += 1
            if it % 200 == 0:
                print(f"  it {it}  ce={ce_s/seen:.3f}  pen={pen_s/seen:.3e}  {time.time()-t0:.0f}s", flush=True)
        acc = S.evaluate(model, vl, dev); nb = len(tl)
        print(f"[nvfp4 lam{a.lam:g}] ep {ep+1}/{a.epochs}  ce={ce_s/nb:.3f}  pen={pen_s/nb:.3e}  acc={acc:.2f}  {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), a.out)
    print(f"DONE nvfp4 lam{a.lam:g}  acc {acc:.2f}  {bpw:.3f} bits/weight  saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
