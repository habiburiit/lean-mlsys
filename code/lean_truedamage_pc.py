#!/usr/bin/env python
"""lean_truedamage_pc.py  -  lean_truedamage.py UNCHANGED except that the INT8 quantizer and the flip step are per output channel
(the deployment default), so 'worst 100' is the real single-flip damage under per-channel INT8.  Same arguments.  SEPARATE file.
  python lean_truedamage_pc.py --dataset imagenet --arch resnet50 --ckpts <base> <lean> <tail> <clip> --labels baseline lam40 tail_only clip_only --out <dir>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import lean_pcstep; lean_pcstep.patch()
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "lean_truedamage.py")).read()
def rep(old, new):
    global src
    assert src.count(old) >= 1, old; src = src.replace(old, new)
rep("store = S.quantize_model_(model, 8)",
    "store = _quantize_pc(model)")
rep("def true_damage(model, store, offs, idx, xs, ys, base_loss):",
    "def _quantize_pc(model):\n"
    "    store = []\n"
    "    with torch.no_grad():\n"
    "        for n, p in S.weight_params(model):\n"
    "            if p.dim() > 1: sc = (p.abs().flatten(1).max(1).values / 127.0 + 1e-12).view(-1, *([1] * (p.dim() - 1))).expand_as(p).contiguous()\n"
    "            else: sc = torch.full_like(p, float(p.abs().max()) / 127.0 + 1e-12)\n"
    "            q = torch.round(p / sc).clamp(-128, 127); p.copy_(q * sc); store.append((n, p, q, sc))\n"
    "    print('[quant] per-output-channel INT8', flush=True); return store\n\n"
    "def true_damage(model, store, offs, idx, xs, ys, base_loss):")
rep("p.view(-1)[k] = new * scale", "p.view(-1)[k] = new * scale.view(-1)[k]")
rep("p.view(-1)[k] = old * scale", "p.view(-1)[k] = old * scale.view(-1)[k]")
rep("delta = np.array([128.0 * float(s) for _, _, _, s in store])", "delta = np.array([128.0 * float(s.mean()) for _, _, _, s in store])")
code = compile(src, "lean_truedamage.py(per-channel)", "exec")
g = {"__name__": "__main__", "__file__": os.path.join(HERE, "lean_truedamage.py")}
exec(code, g)
