#!/usr/bin/env python
"""
lean_pbs_mp8.py  -  pbs_mp.py for mp8 checkpoints. SEPARATE file; pbs_mp.py / lean_eval_mp.py are not changed.
The only difference: a model trained by lean_train_qat_mp8.py stores its exact per-weight scales (quantile-clipped
6/4/3-bit tiers) in <ckpt>.bins.pt under "scale". The attacker must deploy the model on that same grid, so build_mp
is swapped for one that uses the saved scales when present (old mp3..mp7 checkpoints fall back to the original).
All pbs_mp.py flags work unchanged (default --protect none = no error correction, the attacker may flip any weight).

  python lean_pbs_mp8.py --dataset cifar10 --arch resnet18 --ckpt artifacts/models/cifar10/resnet18/model_mp8_b329_lam40_q999.pth \
      --protect none --flips 10,25,50,100,200,300,500
"""
import torch
import lean_eval_mp as EM
import pbs_mp

_orig_build_mp = EM.build_mp
QM = {8: 127, 6: 31, 5: 15, 4: 7, 3: 3, 2: 1}


def build_mp8(params, gflat, a, saved=None):
    if saved is None or "scale" not in saved:
        return _orig_build_mp(params, gflat, a, saved)
    store = []
    with torch.no_grad():
        for n, p in params:
            tier = saved["tier"][n].to(p.device).to(torch.int64); sc = saved["scale"][n].to(p.device).to(p.dtype)
            w = p.detach().clone(); q = torch.zeros_like(w); fm = torch.zeros_like(w)
            for b in torch.unique(tier).tolist():
                m = tier == b; qm = QM[b]
                q = torch.where(m, torch.round(w / sc).clamp(-qm - 1, qm), q)      # weights above the clip saturate
                fm = torch.where(m, torch.full_like(w, float(2 ** (b - 1))), fm)
            p.copy_(q * sc)
            store.append((n, p, q, sc, fm, tier == 8, (tier < 8) & (tier > a.low_bits)))
    print("[mp8] deployed on the saved quantile-clipped scales", flush=True)
    return store


EM.build_mp = build_mp8          # pbs_mp looks up EM.build_mp at call time

if __name__ == "__main__":
    pbs_mp.main()
