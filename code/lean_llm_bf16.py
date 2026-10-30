#!/usr/bin/env python
"""
lean_llm_bf16.py  -  run any of the LLM attack / evaluation scripts (lean_llm_showcase.py, lean_llm_harness.py, lean_llm_silentstriker_w.py,
lean_llm_export.py ...) UNCHANGED, but with every AutoModelForCausalLM.from_pretrained forced to bfloat16, so a 7B model
(15 GB weights + 15 GB gradients) fits one 48 GB GPU for the gradient-based attacks.  The scripts themselves are not modified.

usage:  python -u lean_llm_bf16.py lean_llm_showcase.py --model <path> --cov 0.005 --seed 0 --tag qwen25_7b_lam10 --out results_llm/showcase
environment: BIG_DTYPE (default bfloat16)
"""
import os, sys, runpy, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[os.environ.get("BIG_DTYPE", "bfloat16")]
import transformers
_orig = transformers.AutoModelForCausalLM.from_pretrained.__func__
def _fp(cls, *a, **k):
    k["torch_dtype"] = DTYPE; k.setdefault("attn_implementation", "eager")
    print(f"[bf16 shim] loading {a[0] if a else k.get('pretrained_model_name_or_path')} in {DTYPE}", flush=True)
    return _orig(cls, *a, **k)
transformers.AutoModelForCausalLM.from_pretrained = classmethod(_fp)
# torch.quantile does not accept bfloat16: compute it in float32 transparently
_q = torch.quantile
def _quantile(x, *a, **k):
    return _q(x.float(), *a, **k) if x.dtype in (torch.bfloat16, torch.float16) else _q(x, *a, **k)
torch.quantile = _quantile
script = sys.argv[1]; sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), script), run_name="__main__")
