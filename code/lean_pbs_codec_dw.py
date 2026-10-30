#!/usr/bin/env python
"""
lean_pbs_codec_dw.py  -  lean_pbs_codec.py UNCHANGED except that the coded set is the top --cov tail PLUS every weight of the
depthwise (grouped) convolutions.  Purpose: the compact networks (MobileNet-V2, EfficientNet-B0) collapse through near-zero
depthwise weights that no small tail covers; coding those layers whole closes that hole at a known bit cost, and the question is
whether LEAN then extends survival beyond the normal model under the same code.  Set LEAN_PC=1 to also use per-channel INT8
(the lean_pbs_codec_pc.py quantizer).  Run it through lean_archs.py for the extra backbones:
  python -u lean_archs.py run lean_pbs_codec_dw.py --dataset imagenet --arch mobilenet_v2 --ckpt ... --cov 0.01 --attack pbs --exclude-coded 1 --label ..._dw
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "lean_pbs_codec.py")).read()
if os.environ.get("LEAN_PC", "0") == "1":
    OLDQ = "sc = torch.cat([torch.full((p.numel(),), p.abs().max().item() / 127.0 + 1e-12, device=dev) for p in plist])"
    NEWQ = ("sc = torch.cat([((p.abs().flatten(1).max(1).values / 127.0 + 1e-12).repeat_interleave(p[0].numel()) if p.dim() > 1 "
            "else torch.full((p.numel(),), p.abs().max().item() / 127.0 + 1e-12, device=dev)) for p in plist]); "
            "print('[quant] per-output-channel INT8 scales', flush=True)")
    assert src.count(OLDQ) == 1, "quantizer line not found"; src = src.replace(OLDQ, NEWQ)
OLD = "k = max(1, int(a.cov * N)); prot_idx = torch.topk(dsum, k).indices.cpu().numpy(); prot_idx.sort()"
NEW = (OLD + "\n"
       "    _dwmods = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d) and m.groups > 1}\n"
       "    _dw = [i for i, (n, _) in enumerate(params) if n.rsplit('.', 1)[0] in _dwmods]\n"
       "    _extra = np.concatenate([np.arange(offs[i], offs[i + 1]) for i in _dw]) if _dw else np.zeros(0, dtype=np.int64)\n"
       "    prot_idx = np.unique(np.concatenate([prot_idx.astype(np.int64), _extra])); k = len(prot_idx)\n"
       "    print(f'[dw-code] {len(_dw)} depthwise layers, {len(_extra)} depthwise weights ({100*len(_extra)/N:.2f}% of N) added; coded total {k} ({100*k/N:.2f}%)', flush=True)")
assert src.count(OLD) == 1, "coded-set line not found in lean_pbs_codec.py"
code = compile(src.replace(OLD, NEW), "lean_pbs_codec.py(depthwise-coded)", "exec")
g = {"__name__": "__main__", "__file__": os.path.join(HERE, "lean_pbs_codec.py")}
exec(code, g)
