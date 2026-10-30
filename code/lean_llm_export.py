#!/usr/bin/env python
"""
lean_llm_export.py  -  make a LEAN LLM checkpoint loadable by ecc_llm_attnbreaker.py (--model <dir>).
If <ckpt_dir> already holds config.json (saved with save_pretrained) nothing is done.  Otherwise the first *.pt / *.bin /
*.safetensors state dict in <ckpt_dir> is loaded into the base model and saved with save_pretrained into <ckpt_dir>/hf.
SEPARATE file; nothing existing is modified.
usage:  python lean_llm_export.py --base Qwen/Qwen2-0.5B --ckpt ckpt_llm/qwen2_0.5b/lam3_sum
"""
import argparse, glob, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--base", default="Qwen/Qwen2-0.5B"); ap.add_argument("--ckpt", required=True)
a = ap.parse_args()
if os.path.exists(os.path.join(a.ckpt, "config.json")):
    print(a.ckpt); sys.exit(0)
out = os.path.join(a.ckpt, "hf")
if os.path.exists(os.path.join(out, "config.json")):
    print(out); sys.exit(0)
files = sorted(glob.glob(os.path.join(a.ckpt, "*.pt")) + glob.glob(os.path.join(a.ckpt, "*.bin")) + glob.glob(os.path.join(a.ckpt, "*.safetensors")))
if not files:
    print(f"no state dict in {a.ckpt}: " + ", ".join(os.listdir(a.ckpt)), file=sys.stderr); sys.exit(1)
sd = torch.load(files[0], map_location="cpu") if not files[0].endswith(".safetensors") else __import__("safetensors.torch").torch.load_file(files[0])
if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict): sd = sd["model"]
m = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32)
missing, unexpected = m.load_state_dict(sd, strict=False)
print(f"loaded {files[0]}: missing {len(missing)} unexpected {len(unexpected)}", file=sys.stderr)
if len(missing) > 2: print("missing keys:", missing[:5], file=sys.stderr); sys.exit(1)
m.save_pretrained(out); AutoTokenizer.from_pretrained(a.base).save_pretrained(out); print(out)
