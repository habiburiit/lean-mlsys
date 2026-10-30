#!/usr/bin/env python
"""
lean_pbs_codec_clamp.py  -  lean_pbs_codec.py UNCHANGED except that every activation module (ReLU, ReLU6, SiLU, Hardswish, GELU)
is clamped at deployment to the per-channel range it shows on clean calibration data (the standard activation-bounding
defense against faults; margin LEAN_CLAMP_MARGIN, default 1.0 = the observed max).  Purpose: the compact networks collapse
because one flipped weight drives an unbounded activation; with the activation bounded, the question is whether LEAN's smaller
tail buys survival that the normal model does not get from the same clamp.  Calibration: the first --probe-batches batches of
the evaluation loader on the clean INT8 model, before any attack.  LEAN_PC=1 also switches to per-channel INT8 scales.
  LEAN_PC=1 python -u lean_archs.py run lean_pbs_codec_clamp.py --dataset imagenet --arch mobilenet_v2 --ckpt ... --cov 0.01 --attack pbs --exclude-coded 1 --label ..._clamp_pc
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
OLD = "write_all(q); clean = S.evaluate(model, vl, dev); q_clean = q.clone()"
NEW = ("write_all(q)\n"
       "    _acts = [m for m in model.modules() if isinstance(m, (torch.nn.ReLU, torch.nn.ReLU6, torch.nn.SiLU, torch.nn.Hardswish, torch.nn.GELU))]\n"
       "    _bounds = {}; _margin = float(os.environ.get('LEAN_CLAMP_MARGIN', '1.0'))\n"
       "    def _calib_hook(m):\n"
       "        def h(mod, inp, out):\n"
       "            o = out.detach(); red = [d for d in range(o.dim()) if d != 1] if o.dim() > 1 else [0]\n"
       "            lo = o.amin(dim=red); hi = o.amax(dim=red)\n"
       "            if mod in _bounds: lo = torch.minimum(lo, _bounds[mod][0]); hi = torch.maximum(hi, _bounds[mod][1])\n"
       "            _bounds[mod] = (lo, hi)\n"
       "        return h\n"
       "    _hs = [m.register_forward_hook(_calib_hook(m)) for m in _acts]\n"
       "    with torch.no_grad():\n"
       "        for _i, (x, y) in enumerate(vl):\n"
       "            if _i >= a.probe_batches: break\n"
       "            model(x.to(dev))\n"
       "    for h_ in _hs: h_.remove()\n"
       "    def _clamp_hook(mod, inp, out):\n"
       "        lo, hi = _bounds[mod]; shape = [1, -1] + [1] * (out.dim() - 2) if out.dim() > 1 else [-1]\n"
       "        return torch.maximum(torch.minimum(out, (hi * _margin).view(shape)), (lo * _margin).view(shape))\n"
       "    for m in _acts:\n"
       "        if m in _bounds: m.register_forward_hook(_clamp_hook)\n"
       "    print(f'[clamp] {len(_bounds)} activation modules bounded per channel from {a.probe_batches} clean batches, margin {_margin}', flush=True)\n"
       "    clean = S.evaluate(model, vl, dev); q_clean = q.clone()")
assert src.count(OLD) == 1, "clean-eval line not found in lean_pbs_codec.py"
if "import os" not in src.split("def main")[0]: src = "import os\n" + src
code = compile(src.replace(OLD, NEW), "lean_pbs_codec.py(activation-clamp)", "exec")
g = {"__name__": "__main__", "__file__": os.path.join(HERE, "lean_pbs_codec.py")}
exec(code, g)
