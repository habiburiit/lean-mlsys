#!/usr/bin/env python
"""
lean_fpga_prep.py  -  cut ONE layer out of a packed image (from lean_pack.py) for the FPGA wrapper, and write the per-layer BRAM table.
usage:  python lean_fpga_prep.py --vec ../vec_r50_g7_g32_t3.5_lam40_s0_t8643 --tensor 20 --out .
writes: layer_image.hex (128-bit lines), layer_map.hex (2-bit per block), layer.json + layer.tcl (parameters for vivado_lean.tcl),
        bram_table.md (every tensor: weights, INT8 36Kb-BRAMs vs LEAN 36Kb-BRAMs)
"""
import os, json, math, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--vec", required=True); ap.add_argument("--tensor", type=int, default=-1, help="-1 = largest tensor")
ap.add_argument("--out", default="."); ap.add_argument("--max-bram36", type=int, default=250, help="largest layer whose LEAN BRAM36 count fits this budget"); a = ap.parse_args()
fmt = json.load(open(os.path.join(a.vec, "format.json"))); tiers = fmt["tiers"]; block = fmt["block"]; par = fmt["par_bits"]; lw = fmt["line_weights"]
tensors = [l.split() for l in open(os.path.join(a.vec, "tensors.txt"))]            # tid nblocks r0 r1 r2 r3
blocks = [l.split() for l in open(os.path.join(a.vec, "blocks.txt"))]              # tid tier ratio nwords
stream = open(os.path.join(a.vec, "image.hex")).read().split()
# bit offset of every block in the stream
def blen(t): return block * tiers[int(t)] + ((block // lw) * par if int(t) == 0 else 0)
starts = []; pos = 0
for b in blocks: starts.append(pos); pos += blen(b[1])
# per-tensor spans and BRAM table
rows = []; first = 0
for tid, nb, *_ in tensors:
    nb = int(nb); s = starts[first]; e = starts[first + nb - 1] + blen(blocks[first + nb - 1][1])
    bits = e - s; n = nb * block
    rows.append(dict(tid=int(tid), first=first, nblocks=nb, weights=n, bits=bits, s=s, e=e,
                     bram_int8=math.ceil(n * 8 / 36864), bram_lean=math.ceil((bits + 2 * nb) / 36864)))
    first += nb
with open(os.path.join(a.out, "bram_table.md"), "w") as f:
    f.write("| tensor | weights | LEAN b/w | INT8 BRAM36 | LEAN BRAM36 (incl. map) |\n|---|---|---|---|---|\n")
    for r in rows: f.write(f"| {r['tid']} | {r['weights']:,} | {(r['bits']+2*r['nblocks'])/r['weights']:.2f} | {r['bram_int8']} | {r['bram_lean']} |\n")
    tot8 = sum(r["bram_int8"] for r in rows); totl = sum(r["bram_lean"] for r in rows)
    f.write(f"| all | {sum(r['weights'] for r in rows):,} | | {tot8} | {totl} |\n")
sel = max([r for r in rows if r["bram_lean"] <= a.max_bram36] or rows[:1], key=lambda r: r["weights"]) if a.tensor < 0 else rows[a.tensor]
# cut the layer: bit range [s, e) of the stream, re-aligned to start at bit 0 of its own line memory
bits = "".join(f"{int(l, 16):0128b}" for l in stream[sel["s"] // 128: (sel["e"] + 127) // 128 + 1])
off = sel["s"] % 128; seg = bits[off: off + (sel["e"] - sel["s"])]; seg += "0" * ((-len(seg)) % 128)
lines = [f"{int(seg[i:i+128], 2):032x}" for i in range(0, len(seg), 128)]
open(os.path.join(a.out, "layer_image.hex"), "w").write("\n".join(lines) + "\n")
open(os.path.join(a.out, "layer_map.hex"), "w").write("\n".join(b[1] for b in blocks[sel["first"]: sel["first"] + sel["nblocks"]]) + "\n")
# the same layer as plain INT8 (16 words per 128-bit line) for the baseline build
exp = [int(x) for x in open(os.path.join(a.vec, "expected.txt"))]
w0 = sum(int(b[3]) for b in blocks[: sel["first"]]); ws = exp[w0: w0 + sel["weights"]]; ws += [0] * ((-len(ws)) % 16)
with open(os.path.join(a.out, "layer_int8.hex"), "w") as f:
    for i in range(0, len(ws), 16): f.write("".join(f"{v & 0xff:02x}" for v in ws[i:i + 16]) + "\n")
t = tensors[sel["tid"]]
json.dump(dict(tensor=sel["tid"], nblocks=sel["nblocks"], nlines=len(lines), weights=sel["weights"], block=block, tiers=tiers, par=par, line_weights=lw,
               ratios=[int(x) for x in t[2:6]], bits_per_weight=(sel["bits"] + 2 * sel["nblocks"]) / sel["weights"],
               bram_int8=sel["bram_int8"], bram_lean=sel["bram_lean"]), open(os.path.join(a.out, "layer.json"), "w"), indent=1)
with open(os.path.join(a.out, "layer.tcl"), "w") as f:
    f.write(f"set NBLOCKS {sel['nblocks']}\nset NLINES {len(lines)}\nset NLINES_INT8 {len(ws) // 16}\nset BLOCK {block}\nset PAR {par}\n")
    f.write("set TIERS {" + " ".join(str(x) for x in tiers) + "}\nset RAT {" + " ".join(t[2:6]) + "}\n")
print(f"layer tensor {sel['tid']}: {sel['weights']:,} weights, {sel['nblocks']} blocks, {len(lines)} lines, {(sel['bits']+2*sel['nblocks'])/sel['weights']:.3f} b/w, "
      f"BRAM36 INT8 {sel['bram_int8']} vs LEAN {sel['bram_lean']}; table for {len(rows)} tensors in bram_table.md")
