"""lean_pcstep.py  -  per-output-channel INT8 flip step, as a drop-in for lean_sensreg.msb_step.  SEPARATE file.
msb_step(w) in lean_sensreg is the per-tensor step 128*max|W|/127 (a scalar).  pc_step(w) returns the same quantity per
output channel, broadcast to w's shape, so every `g.abs() * step(p)` in the existing code works unchanged.
import lean_pcstep; lean_pcstep.patch()   # before running hist / truedamage"""
import torch
import lean_sensreg as S

def pc_step(w):
    w = w.detach()
    if w.dim() > 1:
        m = w.abs().flatten(1).max(1).values
        return (128.0 * m / 127.0).view(-1, *([1] * (w.dim() - 1))).expand_as(w)
    return 128.0 * w.abs().max() / 127.0

def patch():
    S.msb_step = pc_step
    print("[pcstep] per-output-channel INT8 flip step active", flush=True)
