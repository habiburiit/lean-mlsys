#!/usr/bin/env python
"""
lean_pbs_codec_pc.py  -  lean_pbs_codec.py UNCHANGED except for the quantizer: per-output-channel INT8 scales instead of one
scale per tensor (the deployment default of TensorRT / PyTorch for weights).  Everything else (code, attacks, options, output
format) is identical.  Run it through lean_archs.py for the extra backbones:
  python -u lean_archs.py run lean_pbs_codec_pc.py --dataset imagenet --arch efficientnet_b0 ... --label ..._pc
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lean_pbs_codec.py")).read()
OLD = "sc = torch.cat([torch.full((p.numel(),), p.abs().max().item() / 127.0 + 1e-12, device=dev) for p in plist])"
NEW = ("sc = torch.cat([((p.abs().flatten(1).max(1).values / 127.0 + 1e-12).repeat_interleave(p[0].numel()) if p.dim() > 1 "
       "else torch.full((p.numel(),), p.abs().max().item() / 127.0 + 1e-12, device=dev)) for p in plist]); "
       "print('[quant] per-output-channel INT8 scales', flush=True)")
assert src.count(OLD) == 1, "quantizer line not found in lean_pbs_codec.py"
code = compile(src.replace(OLD, NEW), "lean_pbs_codec.py(per-channel)", "exec")
g = {"__name__": "__main__", "__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "lean_pbs_codec.py")}
exec(code, g)
