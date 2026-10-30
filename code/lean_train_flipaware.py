#!/usr/bin/env python
"""
lean_train_flipaware.py  -  LEAN fine-tune WITH FLIP-AWARE TRAINING: every step, before the forward pass, FLIPS_PER_STEP
weights get a real INT8 sign flip (q -> q-128 for q>=0, q+128 for q<0, i.e. delta = -sign(w) * 128 * max|w|/127 per tensor),
half of them chosen at random and half from the weights with the largest current gradient (the attacker's kind of flip);
the loss and the LEAN penalty are computed on the flipped weights and the gradient is applied to the clean weights
(random-weight-perturbation training with the storage format's flip step, in the spirit of bit-error training).
Runs lean_sensreg.cmd_train UNCHANGED (hooks on the conv/linear modules and on the optimizer step).  SEPARATE file.

usage: same options as `lean_sensreg.py train`; FLIPS_PER_STEP (default 200), FLIP_P (prob. a step is perturbed, default 1.0),
       FLIP_TARGETED (fraction of flips chosen by gradient, default 0.5) via environment.
"""
import os, sys, argparse, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_sensreg as S

N_FLIPS = int(os.environ.get("FLIPS_PER_STEP", "200")); P_STEP = float(os.environ.get("FLIP_P", "1.0")); TARGETED = float(os.environ.get("FLIP_TARGETED", "0.5"))
_S = {"mods": [], "deltas": {}, "steps": 0, "sizes": None}


def _perturb(mod, inp):
    if not mod.training or torch.rand(()) > P_STEP: return
    w = mod.weight; n = w.numel(); share = n / _S["total"]; k = max(1, int(round(N_FLIPS * share)))
    kt = int(round(k * TARGETED)); kr = k - kt
    idx = [torch.randint(0, n, (kr,), device=w.device)] if kr > 0 else []
    if kt > 0 and w.grad is not None:                                   # last step's gradient: the attacker's ranking
        idx.append(torch.topk(w.grad.detach().abs().flatten(), min(kt * 20, n)).indices[torch.randperm(min(kt * 20, n), device=w.device)[:kt]])
    elif kt > 0:
        idx.append(torch.randint(0, n, (kt,), device=w.device))
    idx = torch.unique(torch.cat(idx))
    with torch.no_grad():
        flat = w.data.view(-1); step = 128.0 * flat.abs().max() / 127.0
        delta = -torch.sign(flat[idx]) * step; delta[flat[idx] == 0] = -step
        flat.index_add_(0, idx, delta); _S["deltas"][mod] = (idx, delta)


def _restore():
    with torch.no_grad():
        for mod, (idx, delta) in _S["deltas"].items(): mod.weight.data.view(-1).index_add_(0, idx, -delta)
    _S["deltas"].clear()


_orig_build = S.build_model
def build_model_flipaware(*a, **k):
    m = _orig_build(*a, **k); mods = [mod for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
    _S["total"] = sum(mod.weight.numel() for mod in mods)
    for mod in mods: mod.register_forward_pre_hook(_perturb)
    print(f"[flipaware] {len(mods)} layers, {N_FLIPS} sign flips per step ({int(100*TARGETED)}% gradient-chosen), p={P_STEP}", flush=True)
    return m
S.build_model = build_model_flipaware

_orig_step = torch.optim.SGD.step
def _step(self, *a, **k):
    _restore(); _S["steps"] += 1                                         # gradient from the flipped forward, applied to the clean weights
    return _orig_step(self, *a, **k)
torch.optim.SGD.step = _step

_orig_eval = S.evaluate
def _evaluate(model, loader, dev):
    _restore(); return _orig_eval(model, loader, dev)                    # evaluation always on clean weights (hooks are off in eval mode)
S.evaluate = _evaluate


def main():
    ap = argparse.ArgumentParser()
    for k, v in [("--arch", "resnet18"), ("--dataset", "cifar10"), ("--imagenet-root", "./imagenet-val"), ("--data", "./data"), ("--init", None), ("--out", None), ("--reg", "both")]:
        ap.add_argument(k, default=v)
    for k, v in [("--workers", 4), ("--seed", 0), ("--epochs", 10), ("--bs", 128), ("--tail-every", 4)]: ap.add_argument(k, type=int, default=v)
    for k, v in [("--lr", 0.002), ("--lam", 40.0), ("--tail-frac", 0.01), ("--clip-q", 0.99), ("--clip-w", 100.0)]: ap.add_argument(k, type=float, default=v)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    S.cmd_train(a)


if __name__ == "__main__":
    main()
