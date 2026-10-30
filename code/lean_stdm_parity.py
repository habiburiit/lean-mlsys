#!/usr/bin/env python
"""
lean_stdm_parity.py  -  parity stored INSIDE the low-bit model with the selective code's STDM (ablation for the total-bit table).
SEPARATE file; imports ecc_engine.py (stdm_store_projected / stdm_load_projected), lean_sensreg.py and the
quantizer convention of lean_train_qat_mp5.py.  Nothing in those files is changed.  No training.

For one saved group model (<ckpt>.pth + <ckpt>.bins.pt from lean_train_qat_group5.py or mp5):
  1. build the deployed integer codes exactly as mp5's forward does (8-bit per-tensor max scale, 6/4/3-bit per-channel
     max scale over the tier members) and measure accuracy in that format (= the "parity counted" row);
  2. count the parity bits: BCH on the 8-bit tier (1.95 x 8 x n8) + sign-plane BCH t=19 n=1023 (190 bits per 1023 weights),
     the same two codes the total-bit accounting charges for;
  3. hide that many bits in the host weights (the 3-bit LOW tier, or LOW + 4-bit with --hosts low4) with the selective code's
     projected STDM: B bits per block of L host codes, lattice step --delta in code units, key --key.  The host codes are
     re-rounded to integers and clamped to the tier grid (the model is stored as integers, so the STDM move must survive
     rounding), then the parity is read back with stdm_load_projected and the read-back bit error rate is reported;
  4. accuracy with the parity in place (= the "parity hidden" row).  The parity VALUES do not matter for the accuracy
     cost, only the embedding does, so a seeded random payload of the right length is used.
A configuration is valid when the host capacity covers the parity (capacity = n_hosts // L blocks x B bits) and the
read-back error is zero.  Several (L, B, delta) are tried in one call; pick the valid one with the smallest accuracy loss.

usage:
  python -u lean_stdm_parity.py --dataset imagenet --arch resnet50 --imagenet-root <path> \
      --ckpt artifacts/models/imagenet/resnet50/group/r50_mp5_g32_t4.0_lam40_s0.pth --configs 1:1:1,2:1:1,2:2:1,4:2:1,4:2:2
      (each config is L:B:delta)
"""
import argparse, math, time, numpy as np, torch
import lean_sensreg as S
import ecc_engine

TIER_QM = {8: 127, 6: 31, 4: 7, 3: 3}


def engine_class():
    for name in dir(ecc_engine):
        obj = getattr(ecc_engine, name)
        if isinstance(obj, type) and hasattr(obj, "stdm_store_projected"):
            return obj
    raise SystemExit("no class with stdm_store_projected in ecc_engine.py")


def deployed_codes(w, t):
    """integer codes and per-weight scales in the mp5 format (w: float tensor, t: int8 tier tensor)"""
    red = tuple(range(1, w.dim())); wabs = w.abs()
    sc = torch.full_like(w, float(wabs.max() / 127.0)).clamp(min=1e-12); qm = torch.full_like(w, 127.0)
    for b in (6, 4, 3):
        m = (t == b)
        if m.any():
            s = ((wabs * m).amax(dim=red, keepdim=True) / TIER_QM[b]).clamp(min=1e-12)
            sc = torch.where(m, s.expand_as(w), sc); qm = torch.where(m, float(TIER_QM[b]), qm)
    q = torch.clamp(torch.round(w / sc), -qm, qm)
    return q, sc, qm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hosts", choices=["low", "low4"], default="low", help="host weights: the 3-bit tier, or 3-bit + 4-bit tiers")
    ap.add_argument("--configs", default="1:1:1,2:1:1,2:2:1,4:2:1,4:2:2", help="comma list of L:B:delta (block length, bits per block, lattice step in code units)")
    ap.add_argument("--t", type=int, default=19); ap.add_argument("--n", type=int, default=1023)
    ap.add_argument("--bch-only", action="store_true", help="hide only the BCH parity of the 8-bit tier (no sign-plane code)")
    ap.add_argument("--key", type=int, default=12345); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    _, vl = S.cifar_loaders(a.data, 128, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.ckpt); params = S.weight_params(model)
    saved = torch.load(a.ckpt + ".bins.pt", map_location="cpu")
    n_total = sum(p.numel() for _, p in params)

    # 1. deployed format and its accuracy
    codes, scales, tiers = {}, {}, {}
    with torch.no_grad():
        for n, p in params:
            t = saved["tier"][n].to(p.device); q, sc, _ = deployed_codes(p.detach(), t)
            codes[n], scales[n], tiers[n] = q, sc, t; p.copy_(q * sc)
    acc0 = S.evaluate(model, vl, dev)
    n8 = sum(int((t == 8).sum()) for t in tiers.values())
    n_bch = int(round(1.95 * 8.0 * n8)); n_sp = 0 if a.bch_only else int(math.ceil(n_total / a.n)) * a.t * math.ceil(math.log2(a.n + 1))
    n_par = n_bch + n_sp
    host_tiers = (3,) if a.hosts == "low" else (3, 4)
    host_mask = {n: sum((t == b) for b in host_tiers).bool() for n, t in tiers.items()}
    n_hosts = sum(int(m.sum()) for m in host_mask.values())
    print(f"[{a.ckpt}] deployed accuracy {acc0:.2f}   N {n_total}   8-bit tier {n8} ({100*n8/n_total:.2f}%)", flush=True)
    print(f"[parity] BCH {n_bch} + sign-plane {n_sp} = {n_par} bits = {n_par/n_total:.3f} bits/weight   hosts ({a.hosts}): {n_hosts} ({100*n_hosts/n_total:.1f}%)", flush=True)

    # host codes as one flat vector (integers as float), in parameter order
    host_vec = torch.cat([codes[n][host_mask[n]] for n, _ in params]).cpu().numpy().astype(np.float64)
    host_qm = torch.cat([torch.where(tiers[n][host_mask[n]] == 3, 3.0, 7.0) for n, _ in params]).cpu().numpy()
    rng = np.random.default_rng(a.seed); payload = rng.integers(0, 2, n_par, dtype=np.uint8)
    E = engine_class()

    print(f"\n{'L':>3s}{'B':>3s}{'delta':>7s}{'capacity':>10s}{'readback BER':>14s}{'moved':>8s}{'acc':>8s}{'loss':>7s}")
    for cfg in a.configs.split(","):
        L, B, delta = cfg.split(":"); L, B, delta = int(L), int(B), float(delta)
        cap = (n_hosts // L) * B
        if cap < n_par:
            print(f"{L:3d}{B:3d}{delta:7.1f}{cap/n_par:9.2f}x   too small (need {n_par} bits, have {cap})", flush=True); continue
        shim = E.__new__(E); shim.host_key = a.key; shim.delta = delta      # engine instance without its constructor
        w_new, _ = E.stdm_store_projected(shim, host_vec.astype(np.float32), payload, B, L=L, delta=delta)
        w_new = np.clip(np.round(w_new.astype(np.float64)), -host_qm, host_qm)          # back to the integer grid
        read = E.stdm_load_projected(shim, w_new.astype(np.float32), n_par, B, L=L, delta=delta)
        ber = float((read != payload).mean()); moved = float((w_new != host_vec).mean())
        # write the modified host codes back and evaluate
        with torch.no_grad():
            off = 0
            for n, p in params:
                m = host_mask[n]; k = int(m.sum()); q = codes[n].clone()
                q[m] = torch.tensor(w_new[off:off + k], dtype=q.dtype, device=q.device); off += k
                p.copy_(q * scales[n])
        acc = S.evaluate(model, vl, dev)
        print(f"{L:3d}{B:3d}{delta:7.1f}{cap/n_par:9.2f}x{ber:14.5f}{100*moved:7.1f}%{acc:8.2f}{acc0-acc:7.2f}", flush=True)
    # restore
    with torch.no_grad():
        for n, p in params: p.copy_(codes[n] * scales[n])
    print(f"\n[summary] parity counted: {n_par/n_total:.3f} bits/weight at {acc0:.2f}%; a row above with BER 0 gives the same parity at 0 bits/weight for its 'loss' points", flush=True)


if __name__ == "__main__":
    main()
