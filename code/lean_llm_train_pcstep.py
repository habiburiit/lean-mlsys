#!/usr/bin/env python
"""
lean_llm_train_pcstep.py  -  lean_llm_train_big.py with the sensitivity step matched to PER-CHANNEL INT8 deployment.
Why: lean_llm_train_big.py concentrates sensitivity with the per-tensor MSB step (one scale per matrix).  When the model is then
deployed and attacked under per-channel INT8 (one scale per output channel), the training's notion of "which weights are dangerous
to flip" no longer matches the attacker's, and the LEAN model can end up weaker than the control (seen on Qwen2.5-7B: control
unbroken, LEAN broken at 43k flips).  Here the step is per output channel: step[row] = 128 * max|w[row,:]| / 127, exactly the
magnitude of an MSB flip in that channel's INT8 code, and the clip penalty is applied per channel as well.
Nothing in lean_llm_train_big.py or lean_sensreg_llm.py is modified: their code is loaded and only msb_step / tail_penalty /
clip_penalty are replaced.  Same arguments and environment variables as lean_llm_train_big.py.

usage (Qwen2.5-7B, 4 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -u lean_llm_train_pcstep.py train --model Qwen/Qwen2.5-7B --epochs 1 --lr 1e-5 --bs 1 \
      --reg both --lam 10 --tail-frac 0.001 --tail-every 2 --max-train-tokens 1000000 --out ckpt_llm/qwen25_7b/lam10_pcstep
"""
import os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "lean_llm_train_big.py")).read()
defs = src[: src.index("# ----------------------------------------------------------------- run lean_sensreg_llm's main")]
g = {"__name__": "lean_llm_train_pcstep_defs", "__file__": os.path.join(HERE, "lean_llm_train_big.py")}
exec(compile(defs, "lean_llm_train_big.py(defs)", "exec"), g)          # loader, penalties, grad clip: all the stabilizers, unchanged
L, DEV0, CLIP_RATIO, _S, _Penalty = g["L"], g["DEV0"], g["CLIP_RATIO"], g["_S"], g["_Penalty"]


def msb_step_pc(w):
    """per-output-channel MSB step, shaped [rows, 1] for 2-D weights (broadcasts over the row); per-tensor for 1-D params"""
    a = w.detach().abs()
    if a.dim() == 2: return 128.0 * a.amax(dim=1, keepdim=True) / 127.0
    return 128.0 * a.max() / 127.0
L.msb_step = msb_step_pc


def clip_penalty_pc(params, q):
    pen = torch.zeros((), device=DEV0)
    for _, p in params:
        a = p.abs()
        if a.dim() == 2:
            thr = torch.quantile(a.detach().float(), q, dim=1, keepdim=True).to(a.dtype)        # per channel
        else:
            thr = torch.quantile(a.detach().float().flatten(), q).to(a.dtype)
        pen = pen + (F.relu(a - thr).pow(2).sum() / a.numel()).to(DEV0)
    return pen
L.clip_penalty = clip_penalty_pc


def tail_penalty_pc(loss, params, tail_frac):
    plist = [p for _, p in params]
    g0 = [gi.detach() for gi in torch.autograd.grad(loss, plist, retain_graph=True)]
    steps = [msb_step_pc(p) for p in plist]                                 # tensors [rows,1] (or scalars)
    n_total = sum(p.numel() for p in plist); k = max(1, int(tail_frac * n_total))
    cands = []
    for gi, s in zip(g0, steps):
        sc = (gi.abs() * s).flatten(); cands.append(torch.topk(sc, min(k, sc.numel()), sorted=False).values.float().to(DEV0))
    thr = torch.topk(torch.cat(cands), k, sorted=True).values[-1]
    value = torch.zeros((), device=DEV0); vidx, vval, wsum, wcnt = [], [], 0.0, 0
    for gi, s, p in zip(g0, steps, plist):
        sc = (gi.abs() * s).flatten().float(); ix = torch.nonzero(sc >= thr.to(sc.device)).flatten()
        s_ix = (s.expand_as(gi).flatten()[ix] if torch.is_tensor(s) and s.dim() > 0 else s)
        value = value + sc[ix].sum().to(DEV0); vidx.append(ix); vval.append(torch.sign(gi.flatten()[ix].float()) * s_ix / 1e4)
        if ix.numel(): wsum += float(p.detach().flatten()[ix].float().abs().sum()); wcnt += ix.numel()
    value = value / 1e4
    vmax = max([float(vv.abs().max()) for vv in vval if vv.numel()] + [1e-30])
    eps = 1e-3 * max(wsum / max(wcnt, 1), 1e-4) / vmax
    _S["calls"] += 1
    if _S["calls"] <= 2: print(f"[fd-pc] tail penalty {float(value):.4e}  k={k}  eps={eps:.3e}  (finite-difference HVP, per-channel step)", flush=True)
    return _Penalty.apply(value.detach(), plist[0], plist, g0, vidx, vval, eps)
L.tail_penalty = tail_penalty_pc

# ----------------------------------------------------------------- run lean_sensreg_llm's main with the patches in place
src2 = open(os.path.join(os.path.dirname(os.path.abspath(L.__file__)), "lean_sensreg_llm.py")).read()
main_block = src2[src2.index('if __name__ == "__main__":'):]
sys.argv = ["lean_sensreg_llm.py"] + sys.argv[1:]
print(f"[big-pc] per-channel sensitivity step and per-channel clip; dtype {g['DTYPE']}, per-GPU cap {g['MAX_GB']} GiB", flush=True)
ns = L.__dict__; ns["__name__"] = "__main__"
exec(compile(main_block, "lean_sensreg_llm.py(main)", "exec"), ns)
