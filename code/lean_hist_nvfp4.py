#!/usr/bin/env python
"""
lean_hist_nvfp4.py  -  Weight-Sensitivity Distribution of NVFP4 models: damage d_i = |g_i| Delta_i with Delta_i the
weight's own NVFP4 single-bit flip step (lean_nvfp4.flip_step_nvfp4), averaged over --batches training batches, both
models evaluated in the NVFP4 format.  Bins HIGH / MID / LOW are fixed from the FIRST checkpoint (--ckpt, the control)
at the 99% and 91% quantiles, like lean_hist.py; the second checkpoint (--ckpt2, LEAN) is counted against the same
thresholds.  Also reports the share of total damage in the top 1%, and the block-scale damage (one E4M3 scale flip moves
all 16 weights of its block: d_block = |sum_j g_j w_j| x (largest single-bit relative change of the E4M3 scale, 2x)).
SEPARATE file.  Saves <out>.npz with both damage vectors and prints the table.

usage:
  python -u lean_hist_nvfp4.py --dataset cifar10 --arch resnet18 --ckpt <nvfp4_lam0.pth> --ckpt2 <nvfp4_lam40.pth> --out sensreg_results/cifar10_resnet18/hist_nvfp4
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import lean_sensreg as S
try:
    import lean_vgg
except Exception:
    pass
import lean_nvfp4 as Q


def damage(model, tl, dev, batches):
    params = S.weight_params(model); plist = [p for _, p in params]
    mods = {n.replace(".weight", ""): m for n, m in model.named_modules()}
    for n, _ in params:
        mod = mods[n.replace(".weight", "")]
        def fwd(x, mod=mod):
            w = Q.fq_nvfp4(mod.weight)
            if isinstance(mod, nn.Conv2d): return F.conv2d(x, w, mod.bias, mod.stride, mod.padding, mod.dilation, mod.groups)
            return F.linear(x, w, mod.bias)
        mod.forward = fwd
    model.eval(); acc = None; accb = None; nb = 0
    for x, y in tl:
        if nb >= batches: break
        x, y = x.to(dev), y.to(dev); loss = F.cross_entropy(model(x), y)
        grads = torch.autograd.grad(loss, plist)
        d = torch.cat([(g.abs() * Q.flip_step_nvfp4(p)).flatten() for g, p in zip(grads, plist)]).cpu()
        # block-scale damage: first-order change of the loss when the whole block is scaled 2x (largest E4M3 single-bit change)
        db = []
        for g, p in zip(grads, plist):
            gw = (g * Q.fq_nvfp4(p).detach()).reshape(-1); pad = (-gw.numel()) % Q.BLOCK
            if pad: gw = torch.cat([gw, gw.new_zeros(pad)])
            db.append(gw.view(-1, Q.BLOCK).sum(1).abs())
        db = torch.cat(db).cpu()
        acc = d if acc is None else acc + d; accb = db if accb is None else accb + db; nb += 1
        model.zero_grad(set_to_none=True)
    return (acc / nb).numpy(), (accb / nb).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--ckpt2", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=20); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--split", choices=["train", "test"], default="train", help="batches the damage is measured on (test = held-out, as plot_sens_curves.py)")
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers); ml = tl if a.split == "train" else vl
    res = {}
    for tag, ck in (("control", a.ckpt), ("lean", a.ckpt2)):
        m = S.build_model(a.arch).to(dev); S.load_state_flex(m, ck)
        d, db = damage(m, ml, dev, a.batches); acc = S.evaluate(m, vl, dev)
        res[tag] = (d, db, acc); print(f"[{tag}] {ck}: NVFP4 acc {acc:.2f}", flush=True)
    d0, db0, _ = res["control"]; th_h = np.quantile(d0, 0.99); th_m = np.quantile(d0, 0.91); tb_h = np.quantile(db0, 0.99)
    print(f"\nthresholds from the control: HIGH >= {th_h:.3e}, MID >= {th_m:.3e}   (block scales: HIGH >= {tb_h:.3e})")
    print(f"{'model':10s}{'acc':>7s}{'HIGH %':>9s}{'MID %':>9s}{'LOW %':>9s}{'top1% share':>13s}{'max d':>11s}{'blocks HIGH %':>15s}")
    for tag in ("control", "lean"):
        d, db, acc = res[tag]; n = d.size; k = max(1, int(0.01 * n)); top = np.sort(d)[-k:].sum() / d.sum()
        print(f"{tag:10s}{acc:7.2f}{100*(d>=th_h).mean():9.3f}{100*((d>=th_m)&(d<th_h)).mean():9.2f}{100*(d<th_m).mean():9.2f}{100*top:13.1f}{d.max():11.3e}{100*(db>=tb_h).mean():15.3f}")
    np.savez_compressed(a.out + ".npz", d_control=d0, d_lean=res["lean"][0], db_control=db0, db_lean=res["lean"][1],
                        acc_control=res["control"][2], acc_lean=res["lean"][2], th_high=th_h, th_mid=th_m)
    print(f"[saved] {a.out}.npz")


if __name__ == "__main__":
    main()
