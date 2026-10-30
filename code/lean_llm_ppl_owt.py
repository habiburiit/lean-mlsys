#!/usr/bin/env python
"""
lean_llm_ppl_owt.py  -  held-out perplexity on OpenWebText (not used in training) for control and LEAN checkpoints, to show that the
LEAN fine-tune on WikiText-2 does not overfit it.  Uses the public 10k-document OpenWebText sample (stas/openwebtext-10k), the first
--max-tokens tokens in --seq-len blocks, same perplexity definition as lean_sensreg_llm.py.  New file; nothing existing is changed.

usage:  python -u lean_llm_ppl_owt.py --model ckpt_llm/qwen25_7b/lam0_1ep --tag qwen25_7b_lam0
        (a checkpoint saved as a state dict is exported first with lean_llm_export.py; pass --base for that)
prints:  RESULT_OWT <tag> ppl=<value> tokens=<n>
"""
import os, sys, math, argparse, subprocess, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--base", default=""); ap.add_argument("--tag", required=True)
ap.add_argument("--seq-len", type=int, default=512); ap.add_argument("--max-tokens", type=int, default=200_000); ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
a = ap.parse_args()

path = a.model
if not os.path.isfile(os.path.join(path, "config.json")):
    if not a.base: sys.exit(f"{path} is a state dict: pass --base <HF name> so it can be exported")
    path = subprocess.run([sys.executable, "lean_llm_export.py", "--base", a.base, "--ckpt", a.model], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
dev = "cuda"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=getattr(torch, a.dtype), attn_implementation="eager").to(dev).eval()
def _load():
    # script datasets are refused by recent `datasets`; use HF's auto-converted parquet revision, then fallbacks
    tries = [("stas/openwebtext-10k", dict(revision="refs/convert/parquet")),
             ("Skylion007/openwebtext", dict(revision="refs/convert/parquet", streaming=True)),
             ("NeelNanda/pile-10k", {})]
    for name, kw in tries:
        try:
            d = load_dataset(name, split="train", **kw); print(f"[owt] loaded {name}", flush=True); return d, name
        except Exception as e:
            print(f"[owt] {name}: {str(e)[:120]}", flush=True)
    sys.exit("no held-out corpus could be loaded")
ds, dsname = _load()
ids = []
for row in ds:
    t = row["text"]
    ids += tok(t).input_ids + [tok.eos_token_id if tok.eos_token_id is not None else 0]
    if len(ids) >= a.max_tokens: break
ids = torch.tensor(ids[:a.max_tokens]); n = (ids.numel() // a.seq_len) * a.seq_len; blocks = ids[:n].view(-1, a.seq_len)
nll = 0.0; cnt = 0
with torch.no_grad():
    for i in range(0, blocks.size(0), a.bs):
        x = blocks[i:i + a.bs].to(dev); out = model(input_ids=x, labels=x)
        nll += out.loss.float().item() * x.numel(); cnt += x.numel()
ppl = math.exp(nll / cnt)
print(f"RESULT_OWT {a.tag} ppl={ppl:.3f} tokens={cnt} blocks={blocks.size(0)} corpus={dsname} model={path}", flush=True)
