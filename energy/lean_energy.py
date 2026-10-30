#!/usr/bin/env python3
"""
lean_energy.py  -  INT8 vs LEAN-format inference energy for ResNet-50 from the Timeloop/Accelergy per-layer stats.
Reads runs/<layer>/timeloop-mapper.stats.txt (Timeloop v3), sums every level x dataspace energy over the 53 layers
(each unique shape times its multiplicity from probs/layers.json), and recomputes the total when the Weights
dataspace is stored in DRAM at b total bits per weight instead of 8:
    DRAM weight energy scales by b/8  (fewer 64-bit beats for the same weights; b already includes tag, scales, parity)
    plus an unpack cost per weight delivered to a PE: one 8-bit bitwise op (Aladdin, 45nm) per weight fill of WeightRegFile
Everything else (inputs, outputs, compute, register files, global buffer) is left unchanged, so the result is conservative.
usage: python3 lean_energy.py [--bits 3.5,4.0] [--unpack-pj 0.03]
"""
import re, json, argparse, os, glob
ap = argparse.ArgumentParser(); ap.add_argument("--bits", default="3.5,4.0"); ap.add_argument("--unpack-pj", type=float, default=0.15)
ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
a = ap.parse_args(); bits = [float(b) for b in a.bits.split(",")]
mult = json.load(open(f"{a.root}/probs/layers.json"))

def parse(path):
    """returns {(level, dataspace): energy_pJ}, total_pJ, macs, weight fills into WeightRegFile (total over instances)"""
    txt = open(path).read(); out = {}; level = None; ds = None; inst = 1; wfills = 0
    for line in txt.splitlines():
        m = re.match(r"=== (\w+) ===", line)
        if m: level = m.group(1); ds = None; continue
        m = re.match(r"\s+Instances\s*:\s*(\d+)", line)
        if m: inst = int(m.group(1))
        m = re.match(r"^    (Weights|Inputs|Outputs):", line)
        if m: ds = m.group(1); continue
        if level == "MACs":
            m = re.match(r"\s+Energy \(total\)\s*:\s*([\d.]+) pJ", line)
            if m and ("MACs", "compute") not in out: out[("MACs", "compute")] = float(m.group(1))
        elif ds:
            m = re.match(r"\s+Energy \(total\)\s*:\s*([\d.]+) pJ", line)
            if m and (level, ds) not in out: out[(level, ds)] = float(m.group(1))
            m = re.match(r"\s+Scalar fills \(per-instance\)\s*:\s*(\d+)", line)
            if m and level == "WeightRegFile" and ds == "Weights": wfills = int(m.group(1)) * inst
    tot = float(re.search(r"^Energy:\s*([\d.]+) uJ", txt, re.M).group(1)) * 1e6
    macs = int(re.search(r"Computes = (\d+)", txt).group(1))
    return out, tot, macs, wfills

agg = {}; total = 0.0; macs = 0; wfills = 0; nl = 0
for name, spec in mult.items():
    p = f"{a.root}/runs/{name}/timeloop-mapper.stats.txt"
    if not os.path.exists(p): print(f"[missing] {name}"); continue
    o, t, m, wf = parse(p); n = spec[-1]; nl += n
    for k, v in o.items(): agg[k] = agg.get(k, 0) + v * n
    total += t * n; macs += m * n; wfills += wf * n
print(f"layers {nl}  MACs {macs/1e9:.2f} G  total energy INT8 {total/1e6:.0f} uJ  ({total/macs:.3f} pJ/MAC)")
print(f"{'level':16s}{'dataspace':10s}{'uJ':>10s}{'%':>7s}")
for (lv, ds), v in sorted(agg.items(), key=lambda kv: -kv[1]): print(f"{lv:16s}{ds:10s}{v/1e6:10.2f}{100*v/total:7.1f}")
wdram = agg.get(("DRAM", "Weights"), 0); wreg = agg.get(("WeightRegFile", "Weights"), 0)
print(f"\nDRAM weight fetch: {wdram/1e6:.2f} uJ = {100*wdram/total:.1f}% of total;  weight fills into PEs: {wfills/1e6:.2f} M")
print(f"\n{'format':14s}{'DRAM weight uJ':>16s}{'unpack uJ':>11s}{'total uJ':>10s}{'DRAM weight saving':>20s}{'total saving':>14s}")
print(f"{'INT8':14s}{wdram/1e6:16.2f}{0:11.2f}{total/1e6:10.2f}{'-':>20s}{'-':>14s}")
rows = {}
for b in bits:
    wd = wdram * b / 8; unpack = wfills * a.unpack_pj; tot = total - wdram + wd + unpack
    rows[b] = dict(dram_w=wd, unpack=unpack, total=tot, dram_saving=1 - wd / wdram, total_saving=1 - tot / total)
    print(f"{f'LEAN {b} b/w':14s}{wd/1e6:16.2f}{unpack/1e6:11.2f}{tot/1e6:10.2f}{100*(1-wd/wdram):19.1f}%{100*(1-tot/total):13.1f}%")
json.dump(dict(total_int8=total, macs=macs, agg={f"{k[0]}/{k[1]}": v for k, v in agg.items()}, lean=rows), open(f"{a.root}/lean_energy.json", "w"), indent=1)
print(f"[saved] {a.root}/lean_energy.json")
