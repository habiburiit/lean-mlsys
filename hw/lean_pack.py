#!/usr/bin/env python
"""
lean_pack.py  -  LEAN weight-memory format: bit-exact packer / unpacker, golden vectors for the RTL, and the bits-per-weight count.
SEPARATE file; touches no existing script.  Needs only numpy and torch.

FORMAT (all parameters are command-line options, defaults in brackets)
  tiers      widths, widest first, index 0 is the ECC-protected HIGH tier           [8,4,2]   (or 8,6,4,3)
  block      weights per block; one tier per block                                 [64]
  tier map   2-bit tier index per block, stored separately (map cost = 2/block b/w = 0.031 b/w at block 64)
  body       block weights as two's-complement codes of the block's width, MSB-first, no padding, blocks back to back
  parity     HIGH blocks only: PAR bits per LINE weights appended after the body    [14 per 16]  (the selective code checker geometry)
  scale      per tensor: INT8 scale s8 = max|w|/127; per tier t: ratio r_t = s_t/s8 in unsigned Q8.8 (one 16-bit register per tier
             per tensor).  The unpacker reconstructs the INT8 word q8 = clip((q_t * r_t + 128) >> 8).  mode=shift sets
             s_t = s8 * 2^(8-w_t) so r_t is a power of two and the unpacker is a pure shifter; mode=rescale uses s_t = max|w in tier t|/(2^(w_t-1)-1).
  lines      the image is read by the hardware as 128-bit lines; blocks are NOT line aligned (the unpacker keeps a bit accumulator).

TIER ASSIGNMENT
  --bins FILE   a torch file mapping parameter name -> per-weight tensor of bit widths (8/6/4/3...) or tier indices (0..3); the
                block tier is the WIDEST tier of its weights (safe side).  Unknown layouts are printed so the loader can be adapted.
  --shares      no bins file: per-tensor magnitude ranking with these tier shares, widest first (demo / baseline)  [0.01,0.24,0.75]
                A per-weight assignment is rounded UP to the widest tier in each block; scattered rare tiers therefore widen most
                blocks (1% HIGH weights at random touch 47% of 64-weight blocks).  The group-wise tiers of lean_train_qat_group5
                (one tier per group, group = block) avoid this and realise the weight-level bits/weight exactly.

OUTPUT (into --out DIR)
  image.hex       one 128-bit line per row (hex, MSB-first)             blocks.txt     per block: tensor_id tier ratio_q88 nwords
  map.hex         tier map, one 2-bit entry per block                     tensors.txt    per tensor: id nblocks ratio0..3 (Q8.8)
  expected.txt    INT8 words in block order, one per row (decimal)      parity.txt     per HIGH block: parity bits as hex
  report.txt      bits/weight breakdown (body, map, parity), weight-level ideal vs block-level realised, per tier set

usage
  python lean_pack.py --ckpt model.pth --bins model.pth.bins.pt --tiers 8,4,2 --out hw_img            (real checkpoint, server)
  python lean_pack.py --demo --tiers 8,4,2 --out hw_img                                              (synthetic weights, self-test)
"""
import os, argparse, json, numpy as np, torch

LINE_BITS = 128


# ----------------------------------------------------------------------------- parity (placeholder; the selective code checker owns the real code)
def parity_crc(bits, npar):
    """CRC of a bit list (MSB-first) with a primitive polynomial of degree npar.  PLACEHOLDER for the the selective code code's parity:
    the RTL only routes these bits to the checker, so any npar-bit function keeps the format and the vectors valid."""
    polys = {14: 0x4443, 7: 0x89, 8: 0x11D, 16: 0x1021 | (1 << 16)}
    poly = polys.get(npar, (1 << npar) | 1 | (1 << 1))
    reg = 0
    for b in bits:
        top = (reg >> (npar - 1)) & 1
        reg = ((reg << 1) | b) & ((1 << npar) - 1)
        if top: reg ^= poly & ((1 << npar) - 1)
    return reg


# ----------------------------------------------------------------------------- helpers
def to_bits(val, width):
    """two's-complement code -> list of bits MSB-first"""
    v = int(val) & ((1 << width) - 1)
    return [(v >> (width - 1 - i)) & 1 for i in range(width)]


def from_bits(bits):
    v = 0
    for b in bits: v = (v << 1) | b
    return v


def sext(code, width):
    return code - (1 << width) if code >= (1 << (width - 1)) else code


def q88(x):
    return int(min(round(x * 256.0), 65535))


def reconstruct(code, ratio):
    """hardware reconstruction: signed code * unsigned Q8.8 ratio, +128, arithmetic shift right 8, clip to +-127"""
    v = (code * ratio + 128) >> 8
    return max(-127, min(127, v))


# ----------------------------------------------------------------------------- tiers
def load_bins(path, names, tiers):
    d = torch.load(path, map_location="cpu")
    if isinstance(d, dict) and "bins" in d and isinstance(d["bins"], dict): d = d["bins"]
    if not isinstance(d, dict):
        raise SystemExit(f"bins file is a {type(d).__name__}; expected dict name -> tensor")
    # LEAN mixed-precision tier files (lean_train_qat_mp*/lean_train_qat_group*): {'high': {name: mask}, 'mid': {...}, 'mid4': {...},
    # 'tier': {...}, 'scale': {...}, 'budget': 3.29}: high = 8 bit, mid = 6 bit, mid4 = 4 bit, everything else = 3 bit.
    if "high" in d and isinstance(d["high"], dict):
        out = {}; hist = {8: 0, 6: 0, 4: 0, 3: 0}
        for n in names:
            if n not in d["high"]: continue
            def m(key):
                v = d.get(key, {}).get(n)
                return torch.as_tensor(v).reshape(-1).bool().numpy() if v is not None else None
            hi, mi, m4 = m("high"), m("mid"), m("mid4")
            w = np.full(hi.size, 3, dtype=np.int64)
            if m4 is not None: w[m4] = 4
            if mi is not None: w[mi] = 6
            w[hi] = 8
            for b in (8, 6, 4, 3): hist[b] += int((w == b).sum())
            if any(int(x) not in tiers for x in np.unique(w)):
                raise SystemExit(f"tier file uses widths {sorted(set(np.unique(w).tolist()))}; run with --tiers 8,6,4,3")
            out[n] = np.array([tiers.index(int(x)) for x in w])
        tot = sum(hist.values())
        print(f"[bins] LEAN tier file: budget {d.get('budget')}, widths 8/6/4/3 = " + " / ".join(f"{100*hist[b]/max(tot,1):.2f}%" for b in (8, 6, 4, 3)) + f" of {tot:,} weights", flush=True)
        missing = [n for n in names if n not in out]
        if missing: print(f"[bins] no entry for {len(missing)} tensors (first: {missing[:3]}); they get the magnitude ranking")
        return out
    out = {}
    for n in names:
        key = n if n in d else next((k for k in d if k.endswith(n) or n.endswith(k)), None)
        if key is None: continue
        t = d[key]
        if isinstance(t, dict): t = t.get("bits", t.get("tier", t.get("width")))
        t = torch.as_tensor(t).reshape(-1).to(torch.int64).numpy()
        if t.max() > 3:  # bit widths
            widths = sorted(set(t.tolist()), reverse=True)
            if any(w not in tiers for w in widths):
                raise SystemExit(f"bins widths {widths} are not a subset of tiers {tiers}; pass --tiers {','.join(map(str, widths))}")
            t = np.array([tiers.index(int(w)) for w in t])
        out[n] = t
    missing = [n for n in names if n not in out]
    if missing: print(f"[bins] no entry for {len(missing)} tensors (first: {missing[:3]}); they get the magnitude ranking")
    return out


def group_tiers(w, shares, block):
    """group-wise assignment (what lean_train_qat_group5 produces): one tier per block, ranked by the block's max |w|"""
    n = w.size; nb = (n + block - 1) // block; wp = np.concatenate([w, np.zeros(nb * block - n)])
    gm = np.abs(wp).reshape(nb, block).max(axis=1); gt = magnitude_tiers(gm, shares)
    return np.repeat(gt, block)[:n]


def magnitude_tiers(w, shares):
    n = w.size; order = np.argsort(-np.abs(w)); t = np.zeros(n, dtype=np.int64); start = 0
    for i, s in enumerate(shares[:-1]):
        k = int(round(s * n)); t[order[start:start + k]] = i; start += k
    t[order[start:]] = len(shares) - 1
    return t


# ----------------------------------------------------------------------------- pack
def pack_tensor(w, tier_w, tiers, block, mode, par_bits, line_w, tid):
    """returns (bitstream list, blocks list, expected words list, parity list, stats)"""
    n = w.size; nb = (n + block - 1) // block
    pad = nb * block - n
    wp = np.concatenate([w, np.zeros(pad)]) if pad else w
    tp = np.concatenate([tier_w, np.full(pad, len(tiers) - 1)]) if pad else tier_w
    s8 = max(float(np.abs(w).max()), 1e-12) / 127.0
    q8 = np.clip(np.round(wp / s8), -127, 127).astype(np.int64)
    btier = tp.reshape(nb, block).min(axis=1)                      # widest tier (smallest index) in the block
    scales = []
    for i, wd in enumerate(tiers):
        if wd == 8 or mode == "shift": s = s8 * (2 ** (8 - wd))        # 8-bit tier = the plain INT8 code (ratio 1.0, lossless)
        else:
            sel = np.abs(wp.reshape(nb, block)[btier == i])
            s = max(float(sel.max()), 1e-12) / ((1 << (wd - 1)) - 1) if sel.size else s8 * (2 ** (8 - wd))   # unused tier: shift ratio
        scales.append(s)
    ratios = [q88(s / s8) for s in scales]
    bits, blocks, expected, parity = [], [], [], []
    body_bits = par_total = 0
    for b in range(nb):
        t = int(btier[b]); wd = tiers[t]; s = scales[t]; r = ratios[t]
        ws = wp[b * block:(b + 1) * block]
        qmax = (1 << (wd - 1)) - 1
        codes = np.clip(np.round(ws / s), -qmax, qmax).astype(np.int64)
        blk = []
        for c in codes: blk += to_bits(int(c), wd)
        body_bits += len(blk)
        pbits = []
        if t == 0:
            for l in range(0, block, line_w):
                pbits += to_bits(parity_crc(blk[l * wd:(l + line_w) * wd], par_bits), par_bits)
            par_total += len(pbits); parity.append(from_bits(pbits))
        bits += blk + pbits
        blocks.append((tid, t, r, block))
        expected += [reconstruct(int(c), r) for c in codes]
    ideal = sum(tiers[int(t)] for t in tier_w) / n
    stats = dict(n=n, blocks=nb, body_bits=body_bits, par_bits=par_total, ideal_bw=ideal, ratios=ratios + [0] * (4 - len(ratios)),
                 tier_hist=[int((btier == i).sum()) for i in range(len(tiers))], q8=q8[:n], expected=np.array(expected[:n]))
    return bits, blocks, expected, parity, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--bins"); ap.add_argument("--demo", action="store_true"); ap.add_argument("--demo-group", action="store_true", help="demo with group-wise tiers (one tier per block)")
    ap.add_argument("--tiers", default="8,4,2"); ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--mode", choices=["shift", "rescale"], default="rescale")
    ap.add_argument("--par-bits", type=int, default=14); ap.add_argument("--line-weights", type=int, default=16)
    ap.add_argument("--shares", default="0.01,0.24,0.75"); ap.add_argument("--max-tensors", type=int, default=0)
    ap.add_argument("--out", default="hw_img"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspect-bins", action="store_true", help="print the structure of --bins and exit (send me this if packing fails)")
    a = ap.parse_args()
    if a.inspect_bins:
        d = torch.load(a.bins, map_location="cpu"); print(type(d).__name__, "with", len(d) if hasattr(d, "__len__") else "?", "entries")
        items = list(d.items())[:6] if isinstance(d, dict) else list(enumerate(d))[:6]
        for k, v in items:
            if torch.is_tensor(v): print(f"  {k}: tensor {tuple(v.shape)} {v.dtype} unique={torch.unique(v).tolist()[:12]}")
            elif isinstance(v, dict):
                print(f"  {k}: dict keys={list(v)[:8]}")
                for kk, vv in list(v.items())[:2]:
                    if torch.is_tensor(vv): print(f"      {kk}: tensor {tuple(vv.shape)} {vv.dtype} unique={torch.unique(vv).tolist()[:8]} sum={float(vv.float().sum()):.0f}")
                    else: print(f"      {kk}: {type(vv).__name__} {str(vv)[:80]}")
            else: print(f"  {k}: {type(v).__name__} {str(v)[:120]}")
        return
    tiers = [int(x) for x in a.tiers.split(",")]; shares = [float(x) for x in a.shares.split(",")]
    assert tiers == sorted(tiers, reverse=True) and tiers[0] == 8 and len(tiers) <= 4, "tiers: widest first, 8 first, at most 4"
    if len(shares) != len(tiers): shares = {2: [0.05, 0.95], 3: [0.01, 0.24, 0.75], 4: [0.01, 0.04, 0.25, 0.70]}[len(tiers)]   # fallback ranking only
    assert a.block % a.line_weights == 0
    os.makedirs(a.out, exist_ok=True)

    if a.demo_group: a.demo = True
    if a.demo:
        rng = np.random.default_rng(a.seed)
        tensors = {f"demo.layer{i}.weight": rng.standard_normal(sz) * 0.05 for i, sz in enumerate([64 * 3 * 9, 64 * 64 * 9, 1000, 4096 + 7])}
        tensors["demo.layer3.weight"][rng.integers(0, 4096 + 7, 40)] *= 20.0   # a few large weights -> HIGH tier
    else:
        sd = torch.load(a.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict): sd = sd["model"]
        tensors = {k: v.detach().float().reshape(-1).numpy() for k, v in sd.items() if torch.is_tensor(v) and v.dim() >= 2}
        if a.max_tensors: tensors = dict(list(tensors.items())[:a.max_tensors])
    names = list(tensors)
    tier_map = load_bins(a.bins, names, tiers) if a.bins else {}

    stream, blocks, expected, parity, per = [], [], [], [], []
    for tid, n in enumerate(names):
        w = tensors[n]; tw = tier_map.get(n)
        if tw is not None and tw.size != w.size and w.size % tw.size == 0: tw = np.repeat(tw, w.size // tw.size)   # group-wise tiers
        if tw is None or tw.size != w.size: tw = magnitude_tiers(w, shares) if not a.demo_group else group_tiers(w, shares, a.block)
        bits, blk, exp, par, st = pack_tensor(w, tw, tiers, a.block, a.mode, a.par_bits, a.line_weights, tid)
        stream += bits; blocks += blk; expected += exp; parity += par; per.append((n, st))
    # self-check: unpack the whole stream independently and compare with the expected words
    pos = 0; k = 0; ok = 0
    for (_, t, r, nw) in blocks:
        wd = tiers[t]
        for j in range(nw):
            code = sext(from_bits(stream[pos:pos + wd]), wd); pos += wd
            ok += int(reconstruct(code, r) == expected[k]); k += 1
        if t == 0: pos += (nw // a.line_weights) * a.par_bits
    assert ok == len(expected) and pos == len(stream), f"self-check failed: {ok}/{len(expected)} words"

    # ---- files
    padlen = (-len(stream)) % LINE_BITS; lines = (len(stream) + padlen) // LINE_BITS
    s = stream + [0] * padlen
    with open(os.path.join(a.out, "image.hex"), "w") as f:
        for i in range(lines): f.write(f"{from_bits(s[i * LINE_BITS:(i + 1) * LINE_BITS]):032x}\n")
    with open(os.path.join(a.out, "blocks.txt"), "w") as f:
        for (tid, t, r, nw) in blocks: f.write(f"{tid} {t} {r} {nw}\n")
    with open(os.path.join(a.out, "expected.txt"), "w") as f:
        for v in expected: f.write(f"{v}\n")
    with open(os.path.join(a.out, "parity.txt"), "w") as f:
        for p in parity: f.write(f"{p:x}\n")
    with open(os.path.join(a.out, "map.hex"), "w") as f:                       # tier map: one 2-bit entry per block
        for (_, t, _, _) in blocks: f.write(f"{t:x}\n")
    with open(os.path.join(a.out, "tensors.txt"), "w") as f:                   # per tensor: id nblocks ratio0..ratio3 (Q8.8)
        for tid, (n, st) in enumerate(per): f.write(f"{tid} {st['blocks']} " + " ".join(str(r) for r in st["ratios"]) + "\n")
    with open(os.path.join(a.out, "format.json"), "w") as f:
        json.dump(dict(tiers=tiers, block=a.block, mode=a.mode, par_bits=a.par_bits, line_weights=a.line_weights, line_bits=LINE_BITS,
                       map_bits_per_block=2, blocks=len(blocks), lines=lines, words=len(expected)), f, indent=1)

    # ---- report
    N = sum(st["n"] for _, st in per); body = sum(st["body_bits"] for _, st in per); parb = sum(st["par_bits"] for _, st in per)
    mapb = 2 * len(blocks); ideal = sum(st["ideal_bw"] * st["n"] for _, st in per) / N
    hist = np.sum([st["tier_hist"] for _, st in per], axis=0)
    err = np.concatenate([np.abs(st["q8"] - st["expected"]) for _, st in per])
    rep = [f"tiers {tiers}  block {a.block}  mode {a.mode}  parity {a.par_bits}/{a.line_weights} weights on HIGH blocks",
           f"weights {N:,}  blocks {len(blocks):,}  lines {lines:,}",
           f"block tier histogram (widest first): {hist.tolist()}  = {[f'{100*h/len(blocks):.2f}%' for h in hist]}",
           f"bits/weight  body {body/N:.3f}  + map {mapb/N:.3f}  + parity {parb/N:.3f}  = {(body+mapb+parb)/N:.3f}   (INT8: 8.000)",
           f"weight-level ideal (no block rounding) {ideal:.3f} b/w  -> block-level realised {body/N:.3f} b/w",
           f"reconstruction vs per-tensor INT8 code: mean |dq8| {err.mean():.3f}, max {err.max()}, exact {100*(err==0).mean():.1f}% of weights",
           f"files: {a.out}/image.hex blocks.txt expected.txt parity.txt format.json"]
    open(os.path.join(a.out, "report.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
