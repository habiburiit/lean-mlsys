#!/usr/bin/env python
"""
lean_llm_train_big.py  -  LEAN fine-tune (finite-difference HVP tail penalty, the lean_llm_train_fd2.py stabilizers) for models that
do not fit one GPU: the model is SHARDED across all visible GPUs (HF device_map, layer-wise pipeline), weights and optimizer in
bfloat16, gradient checkpointing on.  Memory per GPU for a 7B model on 4 x 48 GB: about 15 GB weights+grads+Adam, plus one
first-order gradient copy for the finite difference, plus activations; fits with headroom.
lean_sensreg_llm.py and lean_llm_train_fd2.py are NOT modified; this file re-implements the fd2 penalty with the cross-device
fixes (all reductions on cuda:0, the CE loss moved to cuda:0, quantile in float32).

usage (same arguments as lean_sensreg_llm.py train), e.g. Qwen2.5-7B on 4 GPUs:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -u lean_llm_train_big.py train --model Qwen/Qwen2.5-7B --epochs 1 --lr 1e-5 --bs 1 \
      --reg both --lam 10 --tail-frac 0.001 --tail-every 2 --max-train-tokens 1000000 --out ckpt_llm/qwen25_7b/lam10_sum
environment: BIG_DTYPE (bfloat16|float16|float32, default bfloat16), BIG_MAX_GB (per-GPU cap for the shard planner, default 40),
             FD_CLIP_RATIO, FD_MAX_NORM as in lean_llm_train_fd2.py
"""
import os, sys, torch, torch.nn.functional as F
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_sensreg_llm as L

DEV0 = "cuda:0"
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[os.environ.get("BIG_DTYPE", "bfloat16")]
MAX_GB = float(os.environ.get("BIG_MAX_GB", "40"))
CLIP_RATIO = float(os.environ.get("FD_CLIP_RATIO", "1.0")); MAX_NORM = float(os.environ.get("FD_MAX_NORM", "1.0"))
_S = {"model": None, "last": None, "calls": 0}


# ----------------------------------------------------------------- sharded loader (replaces L.load_model)
def load_model_big(name_or_path, dev, dtype=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    n = torch.cuda.device_count(); mm = {i: f"{int(MAX_GB)}GiB" for i in range(n)}
    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=DTYPE, attn_implementation="eager", device_map="auto", max_memory=mm)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    devs = sorted(set(str(v) for v in model.hf_device_map.values()))
    print(f"[big] {name_or_path} in {DTYPE} sharded over {n} GPUs ({len(model.hf_device_map)} modules; devices {devs}); gradient checkpointing on", flush=True)
    orig = model.forward
    def fwd(*a, **k):                                   # remember the batch; put the loss on cuda:0 so every reduction lives there
        _S["last"] = (a, k); out = orig(*a, **k)
        if getattr(out, "loss", None) is not None and out.loss.device != torch.device(DEV0): out.loss = out.loss.to(DEV0)
        return out
    model.forward = fwd; _S["model"] = model
    return model, tok
L.load_model = load_model_big


# ----------------------------------------------------------------- penalties, cross-device safe
def clip_penalty_big(params, q):
    pen = torch.zeros((), device=DEV0)
    for _, p in params:
        a = p.abs(); flat = a.detach().flatten().float()
        thr = torch.quantile(flat[torch.randint(0, flat.numel(), (min(flat.numel(), 1_000_000),), device=flat.device)], q).to(a.dtype)
        pen = pen + (F.relu(a - thr).pow(2).sum() / a.numel()).to(DEV0)
    return pen
L.clip_penalty = clip_penalty_big


class _Penalty(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, anchor, plist, g0, vidx, vval, eps):
        ctx.plist, ctx.g0, ctx.vidx, ctx.vval, ctx.eps = plist, g0, vidx, vval, eps
        return value.clone()

    @staticmethod
    def backward(ctx, grad_out):
        plist, g0, vidx, vval, eps = ctx.plist, ctx.g0, ctx.vidx, ctx.vval, ctx.eps
        model = _S["model"]; a, k = _S["last"]
        saved = []
        with torch.no_grad():
            for p, ix, vv in zip(plist, vidx, vval):
                flat = p.data.view(-1); saved.append(flat[ix].clone() if ix.numel() else None)
                if ix.numel(): flat.index_add_(0, ix, vv.to(flat.dtype), alpha=eps)
        with torch.enable_grad():
            g1 = list(torch.autograd.grad(model(*a, **k).loss, plist))
        with torch.no_grad():
            for p, ix, sv in zip(plist, vidx, saved):
                if ix.numel(): p.data.view(-1)[ix] = sv
            s = float(grad_out) / eps; nclip = 0
            for i, p in enumerate(plist):
                hv = g1[i].float().sub_(g0[i].float()).mul_(s); g1[i] = None
                gn = g0[i].float().norm(); hn = hv.norm(); g0[i] = None
                if torch.isfinite(hn) and hn > CLIP_RATIO * gn + 1e-12: hv.mul_(CLIP_RATIO * gn / hn); nclip += 1
                elif not torch.isfinite(hn): hv.zero_(); nclip += 1
                hv = hv.to(p.dtype)
                if p.grad is None: p.grad = hv
                else: p.grad.add_(hv)
                del hv
            _S["nclip"] = _S.get("nclip", 0) + nclip; _S["nback"] = _S.get("nback", 0) + 1
            if _S["nback"] % 50 == 0: print(f"[fd2] step {_S['nback']}: second-order term clipped in {nclip}/{len(plist)} layers, {_S['nclip']} total", flush=True)
        ctx.g0 = ctx.vval = None
        return None, None, None, None, None, None, None


def tail_penalty_big(loss, params, tail_frac):
    plist = [p for _, p in params]
    g0 = [gi.detach() for gi in torch.autograd.grad(loss, plist, retain_graph=True)]
    steps = [L.msb_step(p) for p in plist]
    n_total = sum(p.numel() for p in plist); k = max(1, int(tail_frac * n_total))
    cands = []
    for gi, s in zip(g0, steps):
        sc = (gi.abs() * s).flatten(); cands.append(torch.topk(sc, min(k, sc.numel()), sorted=False).values.float().to(DEV0))
    thr = torch.topk(torch.cat(cands), k, sorted=True).values[-1]
    value = torch.zeros((), device=DEV0); vidx, vval, wsum, wcnt = [], [], 0.0, 0
    for gi, s, p in zip(g0, steps, plist):
        sc = (gi.abs() * s).flatten().float(); ix = torch.nonzero(sc >= thr.to(sc.device)).flatten()
        value = value + sc[ix].sum().to(DEV0); vidx.append(ix); vval.append((torch.sign(gi.flatten()[ix].float()) * s / 1e4))
        if ix.numel(): wsum += float(p.detach().flatten()[ix].float().abs().sum()); wcnt += ix.numel()
    value = value / 1e4
    vmax = max([float(vv.abs().max()) for vv in vval if vv.numel()] + [1e-30])
    eps = 1e-3 * max(wsum / max(wcnt, 1), 1e-4) / vmax
    _S["calls"] += 1
    if _S["calls"] <= 2: print(f"[fd] tail penalty {float(value):.4e}  k={k}  eps={eps:.3e}  (finite-difference HVP, sharded)", flush=True)
    return _Penalty.apply(value.detach(), plist[0], plist, g0, vidx, vval, eps)
L.tail_penalty = tail_penalty_big


# ----------------------------------------------------------------- global grad-norm clip before every optimizer step
for _cls in [torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD]:
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

# ----------------------------------------------------------------- run lean_sensreg_llm's main with the patches in place
src = open(os.path.join(os.path.dirname(os.path.abspath(L.__file__)), "lean_sensreg_llm.py")).read()
main_block = src[src.index('if __name__ == "__main__":'):]
sys.argv = ["lean_sensreg_llm.py"] + sys.argv[1:]
print(f"[big] dtype {DTYPE}, per-GPU cap {MAX_GB} GiB, stabilizers: |Hv| <= {CLIP_RATIO}|g| per layer, grad-norm clip {MAX_NORM}", flush=True)
g = L.__dict__; g["__name__"] = "__main__"
exec(compile(main_block, "lean_sensreg_llm.py(main)", "exec"), g)
