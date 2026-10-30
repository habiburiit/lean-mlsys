#!/usr/bin/env python
"""
lean_train_far.py  -  Forget-and-Rewire (FaR) baseline (Nazari et al., USENIX Security 2024), adapted from transformer
linear layers to CNN conv/linear layers. SEPARATE file; imports from lean_sensreg.py.

Idea of FaR: weights that matter most to the attacker are "forgotten" (zeroed) and their contribution is "rewired" into
a less important weight of the same neuron, so the model's most vulnerable bits disappear from memory; a short
fine-tune recovers accuracy. Here:
  1. damage score d_i = |grad_i| * INT8 sign-flip step (our score, same as the attacker's ranking)
  2. per output channel: take the top --far-frac fraction of weights by d_i, add each one's value to a randomly chosen
     weight in the bottom half of that channel's d_i ranking, set the original to zero
  3. fine-tune --epochs epochs with the forgotten positions kept at zero (mask), standard CE loss

Usage:
  python lean_train_far.py --dataset cifar10 --arch resnet18 --init artifacts/models/cifar10/resnet18/model_float32.pth \
      --far-frac 0.01 --epochs 10 --out artifacts/models/cifar10/resnet18/model_float32_far.pth
"""
import argparse, time, torch, torch.nn.functional as F
import lean_sensreg as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--far-frac", type=float, default=0.01, help="fraction of weights per channel to forget-and-rewire")
    ap.add_argument("--probe-batches", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    tl, vl = S.cifar_loaders(a.data, a.bs, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.init)
    print(f"[init] clean acc = {S.evaluate(model, vl, dev):.2f}   FaR baseline, far_frac={a.far_frac}")
    params = S.weight_params(model); plist = [p for _, p in params]
    # 1. damage score on a few batches
    model.eval(); acc = None; n = 0
    for x, y in tl:
        if n >= a.probe_batches: break
        x, y = x.to(dev), y.to(dev); loss = F.cross_entropy(model(x), y); g = torch.autograd.grad(loss, plist)
        d = [(gg.abs() * S.msb_step(p)).detach() for gg, p in zip(g, plist)]
        acc = d if acc is None else [x_ + y_ for x_, y_ in zip(acc, d)]; n += 1; model.zero_grad(set_to_none=True)
    # 2. forget and rewire per output channel
    masks = {}; n_moved = 0
    with torch.no_grad():
        for (name, p), dmap in zip(params, acc):
            w = p.view(p.shape[0], -1); dm = dmap.view(p.shape[0], -1); keep = torch.ones_like(w, dtype=torch.bool)
            k = max(1, int(a.far_frac * w.shape[1]))
            for c in range(w.shape[0]):
                order = torch.argsort(dm[c], descending=True); top = order[:k]; low = order[w.shape[1] // 2:]
                tgt = low[torch.randint(len(low), (k,), device=dev)]
                w[c, tgt] += w[c, top]; w[c, top] = 0.0; keep[c, top] = False; n_moved += k
            masks[name] = keep.view_as(p)
    print(f"[far] forgot and rewired {n_moved:,} weights ({100*n_moved/sum(p.numel() for p in plist):.2f}%); acc after rewiring (before fine-tune) = {S.evaluate(model, vl, dev):.2f}")
    # 3. fine-tune with forgotten positions held at zero
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl))
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); ce_s = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev); ce = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True); ce.backward(); opt.step(); sched.step(); ce_s += ce.item()
            with torch.no_grad():
                for name, p in params: p.mul_(masks[name])
        acc_ = S.evaluate(model, vl, dev)
        print(f"ep {ep+1}/{a.epochs}  ce={ce_s/len(tl):.3f}  acc={acc_:.2f}  {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), a.out); print(f"[saved] {a.out}  final acc {acc_:.2f}")


if __name__ == "__main__":
    main()
