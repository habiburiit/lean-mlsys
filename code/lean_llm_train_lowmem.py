#!/usr/bin/env python
"""
lean_llm_train_lowmem.py  -  runs lean_sensreg_llm.py UNCHANGED (imported, then its own __main__ block executed) with a
memory-light tail_penalty patched in.  Same mathematics as the original:
    tail_penalty = sum of the top tail_frac of  |g_i| x msb_step(p)  over all regularized weights, / 1e4
The original concatenates the scores of all weights into one 1.3B-element tensor (plus |g| temporaries) on top of the
second-order graph, about 10 GB of peak memory on Qwen2-1.5B.  This version takes the top-k inside every layer first
(the global top-k is a subset of the union of per-layer top-k's, so the result is identical), concatenates only those
candidates, and frees each layer's score tensor as it goes.  Also makes the optimizer run without foreach temporaries.
usage: identical to lean_sensreg_llm.py, e.g.
  python -u lean_llm_train_lowmem.py train --model Qwen/Qwen2-1.5B --epochs 1 --lr 2e-5 --reg both --lam 3 --tail-frac 0.001 --out ckpt_llm/qwen2_1.5b/lam3_sum
"""
import os, sys, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_sensreg_llm as L

for name in ("AdamW", "Adam"):
    base = getattr(torch.optim, name)
    def make(base):
        class LowMem(base):
            def __init__(self, *a, **k):
                k["foreach"] = False; super().__init__(*a, **k)
        LowMem.__name__ = base.__name__; return LowMem
    setattr(torch.optim, name, make(base))
    if hasattr(L, name): setattr(L, name, getattr(torch.optim, name))


def tail_penalty_lowmem(loss, params, tail_frac):
    plist = [p for _, p in params]
    grads = torch.autograd.grad(loss, plist, create_graph=True)
    n_total = sum(p.numel() for p in plist); k = max(1, int(tail_frac * n_total))
    cands = []
    for g, p in zip(grads, plist):
        s = (g.abs() * L.msb_step(p)).flatten()
        kk = min(k, s.numel())
        cands.append(torch.topk(s, kk, sorted=False).values)      # keeps the graph through g for these entries only
        del s
    cands = torch.cat(cands)
    return torch.topk(cands, k, sorted=False).values.sum() / 1e4


L.tail_penalty = tail_penalty_lowmem
src = open(os.path.join(os.path.dirname(os.path.abspath(L.__file__)), "lean_sensreg_llm.py")).read()
main_block = src[src.index('if __name__ == "__main__":'):]
sys.argv = ["lean_sensreg_llm.py"] + sys.argv[1:]
g = L.__dict__; g["__name__"] = "__main__"
exec(compile(main_block, "lean_sensreg_llm.py(main)", "exec"), g)
