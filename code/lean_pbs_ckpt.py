#!/usr/bin/env python
"""lean_pbs_ckpt.py - run pbs_four.py unchanged on ANY checkpoint, no swap trick.
Pops --ckpt from the command line, redirects every torch.load of a .pth/.pt file to that checkpoint
(wrapping it the same way the original file is wrapped, e.g. {'model': state_dict}), then calls pbs_four.main().
usage:
  python -u lean_pbs_ckpt.py --ckpt artifacts/models/cifar10/resnet18/inset/heldlr.pth \
      --dataset CIFAR10 --arch resnet18 --bits 8 --protect selcode --covs 0.01 --flips 100,200,300,500 --label inset_heldlr
"""
import os, sys, torch

i = sys.argv.index("--ckpt"); CKPT = sys.argv[i + 1]; del sys.argv[i:i + 2]
_orig = torch.load

def _wrap_like(orig_obj, new_obj):
    if isinstance(orig_obj, dict) and isinstance(new_obj, dict):
        for key in ("model", "state_dict", "net", "model_state_dict"):
            if key in orig_obj and key not in new_obj:
                o = dict(orig_obj); o[key] = new_obj; return o
        if isinstance(new_obj, dict) and any(k in new_obj for k in ("model", "state_dict", "net", "model_state_dict")) \
           and not any(k in orig_obj for k in ("model", "state_dict", "net", "model_state_dict")):
            for key in ("model", "state_dict", "net", "model_state_dict"):
                if key in new_obj: return new_obj[key]
    return new_obj

def _load(f, *a, **k):
    if isinstance(f, (str, os.PathLike)) and str(f).endswith((".pth", ".pt")):
        print(f"[lean_pbs_ckpt] torch.load({f}) -> {CKPT}", flush=True)
        new = _orig(CKPT, *a, **k)
        try:
            orig = _orig(f, *a, **k) if os.path.exists(f) else None
        except Exception:
            orig = None
        return _wrap_like(orig, new) if orig is not None else new
    return _orig(f, *a, **k)

torch.load = _load
import pbs_four
pbs_four.main()
