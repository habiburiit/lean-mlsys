#!/usr/bin/env python
"""
lean_train_band.py  -  LEAN with a CODE-AWARE tail: the tail penalty acts on the damage band the code leaves exposed.
With a code over the top --code-frac of the weights (1% in the paper), those weights cannot be flipped, so shaping them is
wasted; what decides survival at 300 to 500 flips is the band just below the code.  The penalty here is the mean damage of the
weights ranked between --code-frac and --tail-frac (default: ranks 1% to 5%), differentiable through the gradient as in
lean_sensreg.tail_penalty.  Runs lean_sensreg.cmd_train UNCHANGED otherwise.  SEPARATE file.

usage: same options as `lean_sensreg.py train` plus --code-frac (default 0.01); use --tail-frac 0.05.
"""
import os, sys, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_sensreg as S

CODE_FRAC = float(os.environ.get("CODE_FRAC", "0.01"))

def band_penalty(loss, params, tail_frac):
    grads = torch.autograd.grad(loss, [p for _, p in params], create_graph=True)
    scores = torch.cat([(g.abs() * S.msb_step(p)).flatten() for g, (_, p) in zip(grads, params)])
    k_hi = max(1, int(tail_frac * scores.numel())); k_code = int(CODE_FRAC * scores.numel())
    top = torch.topk(scores, k_hi, sorted=True).values          # largest first
    return top[k_code:].mean()                                   # the band the code does not cover
S.tail_penalty = band_penalty

def main():
    ap = argparse.ArgumentParser()
    for k, v in [("--arch", "resnet18"), ("--dataset", "cifar10"), ("--imagenet-root", "./imagenet-val"), ("--data", "./data"), ("--init", None), ("--out", None), ("--reg", "both")]:
        ap.add_argument(k, default=v)
    for k, v in [("--workers", 4), ("--seed", 0), ("--epochs", 10), ("--bs", 128), ("--tail-every", 4)]: ap.add_argument(k, type=int, default=v)
    for k, v in [("--lr", 0.002), ("--lam", 40.0), ("--tail-frac", 0.05), ("--clip-q", 0.99), ("--clip-w", 100.0), ("--code-frac", 0.01)]: ap.add_argument(k, type=float, default=v)
    a = ap.parse_args(); global CODE_FRAC; CODE_FRAC = a.code_frac; S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    print(f"[band] tail penalty on damage ranks {100*a.code_frac:.1f}% to {100*a.tail_frac:.1f}% (code-aware)", flush=True)
    S.cmd_train(a)

if __name__ == "__main__":
    main()
