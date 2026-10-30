#!/usr/bin/env python
"""
lean_train_logit.py  -  E6: LEAN with a LOGIT-AWARE tail term.  Runs lean_sensreg.cmd_train UNCHANGED, but the tail penalty
is computed on the gradient of   CE(x, y) + W * mean_k CE(x, t_k)   where t_1..t_K are K target classes drawn at random each
step, i.e. the damage ranking also sees the "push everything to class t" path that an all-to-one targeted attack uses.
The targeted loss enters only through the penalty (its gradient is NOT added to the training loss), so the model is not
trained toward any class.  Logits are captured by a forward hook on the classifier.  SEPARATE file.

usage (same options as `lean_sensreg.py train`; LOGIT_W and LOGIT_K via environment):
  LOGIT_W=1.0 LOGIT_K=4 python -u lean_train_logit.py --dataset imagenet --arch resnet50 --imagenet-root ./imagenet-val \
      --init artifacts/models/imagenet/resnet50/model_float32.pth --reg both --lam 40 --lr 0.0005 --bs 32 --epochs 10 \
      --out artifacts/models/imagenet/resnet50/model_float32_sensreg_lam40_logit.pth
"""
import os, sys, argparse, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_sensreg as S
try:
    import lean_vgg
except Exception:
    pass

W = float(os.environ.get("LOGIT_W", "1.0")); K = int(os.environ.get("LOGIT_K", "4"))
_S = {"logits": None, "calls": 0}

_orig_build = S.build_model
def build_model_hooked(*a, **k):
    m = _orig_build(*a, **k)
    last = [mod for mod in m.modules() if isinstance(mod, nn.Linear)][-1]
    last.register_forward_hook(lambda mod, inp, out: _S.__setitem__("logits", out))
    print(f"[logit] hooked classifier {last.__class__.__name__}({last.in_features}->{last.out_features}); W={W}, K={K} sampled targets per step", flush=True)
    return m
S.build_model = build_model_hooked

_orig_tail = S.tail_penalty
def tail_penalty_logit(loss, params, tail_frac):
    z = _S["logits"]
    if z is None: return _orig_tail(loss, params, tail_frac)
    C = z.size(1); ts = torch.randint(0, C, (K,), device=z.device)
    lt = sum(F.cross_entropy(z, torch.full((z.size(0),), int(t), device=z.device, dtype=torch.long)) for t in ts) / K
    _S["calls"] += 1
    if _S["calls"] <= 2: print(f"[logit] CE {float(loss):.3f}  targeted CE {float(lt):.3f}  (penalty on their sum, W={W})", flush=True)
    return _orig_tail(loss + W * lt, params, tail_frac)
S.tail_penalty = tail_penalty_logit


def main():
    ap = argparse.ArgumentParser()
    for k, v in [("--arch", "resnet50"), ("--dataset", "imagenet"), ("--imagenet-root", "./imagenet-val"), ("--data", "./data"), ("--init", None), ("--out", None), ("--reg", "both")]:
        ap.add_argument(k, default=v)
    for k, v in [("--workers", 4), ("--seed", 0), ("--epochs", 10), ("--bs", 32), ("--tail-every", 4)]: ap.add_argument(k, type=int, default=v)
    for k, v in [("--lr", 0.0005), ("--lam", 40.0), ("--tail-frac", 0.01), ("--clip-q", 0.99), ("--clip-w", 100.0)]: ap.add_argument(k, type=float, default=v)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    S.cmd_train(a)


if __name__ == "__main__":
    main()
