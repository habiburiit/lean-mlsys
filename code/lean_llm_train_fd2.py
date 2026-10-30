#!/usr/bin/env python
"""
lean_llm_train_fd2.py  -  lean_llm_train_fd.py plus two stabilizers (the unclipped finite-difference term diverged twice on Qwen2-1.5B
at batch size 1: final PPL 684 / 1245): per-layer clipping of the second-order term to the norm of the first-order gradient,
and global gradient-norm clipping (1.0) before every optimizer step.  Runs lean_sensreg_llm.py UNCHANGED; the second-order
step is a finite-difference Hessian-vector product instead of a retained double-backward graph, for models whose double backward
does not fit the GPU (Qwen2-1.5B needs more than 44 GB with the retained graph; this needs about 32 GB).

Mathematics, identical to the original up to the finite-difference step:
  P    = sum_{i in top-k} |g_i| Δ_i / 1e4         g = dL/dw on the batch, Δ_i the INT8 sign-flip step (detached, as in the original)
  dP/dw = H v,  v_i = sign(g_i) Δ_i / 1e4 on the top-k and 0 elsewhere
  H v  ≈ (g(w + εv) - g(w)) / ε                  Pearlmutter's finite difference: one extra forward+backward on the same batch
The optimizer receives the same gradient the double backward would give, with O(ε) error; ε moves the top-k weights by
about 1e-3 of their magnitude.  v is sparse (top-k only), so the extra memory is one first-order gradient.

usage: same arguments as lean_sensreg_llm.py, e.g.
  python -u lean_llm_train_fd.py train --model Qwen/Qwen2-1.5B --epochs 1 --lr 2e-5 --bs 1 --reg both --lam 3 --tail-frac 0.001 \
         --tail-every 2 --max-train-tokens 1000000 --out ckpt_llm/qwen2_1.5b/lam3_sum
"""
import os, sys, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_sensreg_llm as L

_S = {"model": None, "last": None, "calls": 0}
CLIP_RATIO = float(os.environ.get("FD_CLIP_RATIO", "1.0"))   # |H v| <= CLIP_RATIO * |g| per layer
MAX_NORM = float(os.environ.get("FD_MAX_NORM", "1.0"))       # global grad-norm clip before each optimizer step


def _hook(model):
    """remember the model and the arguments of its last forward call (the batch the penalty belongs to)"""
    orig = model.forward
    def fwd(*a, **k):
        _S["last"] = (a, k); return orig(*a, **k)
    model.forward = fwd; _S["model"] = model; return model


_orig_load = L.load_model
def _load_model(*a, **k):
    out = _orig_load(*a, **k); _hook(out[0] if isinstance(out, tuple) else out); return out
L.load_model = _load_model


class _Penalty(torch.autograd.Function):
    """forward: the penalty value.  backward: adds grad_out * H v to every parameter's .grad (finite-difference HVP)."""
    @staticmethod
    def forward(ctx, value, anchor, plist, g0, vidx, vval, eps):
        ctx.plist, ctx.g0, ctx.vidx, ctx.vval, ctx.eps = plist, g0, vidx, vval, eps
        return value.clone()

    @staticmethod
    def backward(ctx, grad_out):
        plist, g0, vidx, vval, eps = ctx.plist, ctx.g0, ctx.vidx, ctx.vval, ctx.eps
        model = _S["model"]; a, k = _S["last"]
        # perturb w -> w + eps v through .data (no autograd version bump: the retained CE graph stays valid), exact restore after
        saved = []
        with torch.no_grad():
            for p, ix, vv in zip(plist, vidx, vval):
                flat = p.data.view(-1); saved.append(flat[ix].clone() if ix.numel() else None)
                if ix.numel(): flat.index_add_(0, ix, vv, alpha=eps)
        with torch.enable_grad():
            g1 = list(torch.autograd.grad(model(*a, **k).loss, plist))
        with torch.no_grad():
            for p, ix, sv in zip(plist, vidx, saved):
                if ix.numel(): p.data.view(-1)[ix] = sv                        # bit-exact restore
            s = float(grad_out) / eps
            nclip = 0
            for i, p in enumerate(plist):                                      # H v layer by layer, freeing as we go
                hv = g1[i].sub_(g0[i]).mul_(s); g1[i] = None
                gn = g0[i].norm(); hn = hv.norm(); g0[i] = None
                if torch.isfinite(hn) and hn > CLIP_RATIO * gn + 1e-12: hv.mul_(CLIP_RATIO * gn / hn); nclip += 1
                elif not torch.isfinite(hn): hv.zero_(); nclip += 1
                if p.grad is None: p.grad = hv
                else: p.grad.add_(hv)
                del hv
            _S["nclip"] = _S.get("nclip", 0) + nclip; _S["nback"] = _S.get("nback", 0) + 1
            if _S["nback"] % 50 == 0: print(f"[fd2] step {_S['nback']}: second-order term clipped in {nclip}/{len(plist)} layers this step, {_S['nclip']} layer-clips total", flush=True)
        ctx.g0 = ctx.vval = None
        return None, None, None, None, None, None, None


def tail_penalty_fd(loss, params, tail_frac):
    plist = [p for _, p in params]
    g0 = [gi.detach() for gi in torch.autograd.grad(loss, plist, retain_graph=True)]
    steps = [L.msb_step(p) for p in plist]
    n_total = sum(p.numel() for p in plist); k = max(1, int(tail_frac * n_total))
    cands = []
    for gi, s in zip(g0, steps):
        sc = (gi.abs() * s).flatten(); cands.append(torch.topk(sc, min(k, sc.numel()), sorted=False).values)
    thr = torch.topk(torch.cat(cands), k, sorted=True).values[-1]
    value = torch.zeros((), device=plist[0].device); vidx, vval, wsum, wcnt = [], [], 0.0, 0
    for gi, s, p in zip(g0, steps, plist):
        sc = (gi.abs() * s).flatten(); ix = torch.nonzero(sc >= thr).flatten()
        value = value + sc[ix].sum(); vidx.append(ix); vval.append(torch.sign(gi.flatten()[ix]) * s / 1e4)
        if ix.numel(): wsum += float(p.detach().flatten()[ix].abs().sum()); wcnt += ix.numel()
    value = value / 1e4
    vmax = max([float(vv.abs().max()) for vv in vval if vv.numel()] + [1e-30])
    eps = 1e-3 * max(wsum / max(wcnt, 1), 1e-4) / vmax
    _S["calls"] += 1
    if _S["calls"] <= 2: print(f"[fd] tail penalty {float(value):.4e}  k={k}  eps={eps:.3e}  (finite-difference HVP)", flush=True)
    return _Penalty.apply(value.detach(), plist[0], plist, g0, vidx, vval, eps)


L.tail_penalty = tail_penalty_fd

_orig_step = torch.optim.Optimizer.step
def _clipped_step(self, *a, **k):
    params = [p for g in self.param_groups for p in g["params"] if p.grad is not None]
    if params:
        gn = torch.nn.utils.clip_grad_norm_(params, MAX_NORM)
        _S["gn"] = float(gn); _S["nstep"] = _S.get("nstep", 0) + 1
        if _S["nstep"] % 50 == 0: print(f"[fd2] optimizer step {_S['nstep']}: grad norm before clip {float(gn):.3f} (clip {MAX_NORM})", flush=True)
    return _orig_step(self, *a, **k)
torch.optim.Optimizer.step = _clipped_step
for _cls in [torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD]:
    if "step" in _cls.__dict__:
        _o = _cls.step
        def _mk(_o):
            def _st(self, *a, **k):
                params = [p for g in self.param_groups for p in g["params"] if p.grad is not None]
                if params:
                    gn = torch.nn.utils.clip_grad_norm_(params, MAX_NORM); _S["nstep"] = _S.get("nstep", 0) + 1
                    if _S["nstep"] % 50 == 0: print(f"[fd2] optimizer step {_S['nstep']}: grad norm before clip {float(gn):.3f} (clip {MAX_NORM})", flush=True)
                return _o(self, *a, **k)
            return _st
        _cls.step = _mk(_o)
src = open(os.path.join(os.path.dirname(os.path.abspath(L.__file__)), "lean_sensreg_llm.py")).read()
main_block = src[src.index('if __name__ == "__main__":'):]
sys.argv = ["lean_sensreg_llm.py"] + sys.argv[1:]
print(f"[fd2] stabilizers: per-layer |Hv| <= {CLIP_RATIO} |g|, global grad-norm clip {MAX_NORM}", flush=True)
g = L.__dict__; g["__name__"] = "__main__"
exec(compile(main_block, "lean_sensreg_llm.py(main)", "exec"), g)
