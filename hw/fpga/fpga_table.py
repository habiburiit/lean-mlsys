#!/usr/bin/env python
"""fpga_table.py  -  FPGA tables from the Vivado reports (fpga/reports_int8, reports_lean, reports_lean_ecc).  usage: python fpga_table.py"""
import re, os, json
def util(path):
    rows = {}
    for line in open(path):
        if line.startswith("|") and line.count("|") >= 5:
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            try: rows.setdefault(c[0].rstrip("*"), (int(c[1]), c[-2], c[-1]))
            except ValueError: pass
    def pick(*pats):
        for p in pats:
            for k, v in rows.items():
                if re.fullmatch(p, k, re.I): return v
        return (0, "?", "?")
    return dict(lut=pick(r"CLB LUTs", r"Slice LUTs", r"LUTs", r".*LUTs.*"), ff=pick(r"CLB Registers", r"Slice Registers", r"Registers", r".*Registers.*"),
                bram=pick(r"Block RAM Tile", r"RAMB36.*", r".*RAM Tile.*"), dsp=pick(r"DSPs", r"DSP58", r"DSP48.*", r".*DSP.*"))
def timing(path):
    m = re.search(r"WNS\(ns\).*?\n.*?\n\s*(-?[\d.]+)", open(path).read()); return float(m.group(1)) if m else float("nan")
def power(path):
    p = open(path).read(); g = lambda k: (re.search(k + r"\s*\|\s*([\d.]+)", p) or [None, "?"])[1]
    return g(r"Total On-Chip Power \(W\)"), g(r"Dynamic \(W\)"), g(r"Device Static \(W\)"), g(r"Block RAM"), g(r"Clocks"), g(r"CLB Logic|Slice Logic"), g(r"Signals")
def hier(path):
    out, hdr = {}, None
    for line in open(path):
        if line.startswith("|") and line.count("|") >= 8:
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if c[0] == "Instance": hdr = c; continue
            if hdr: out[c[0]] = dict(zip(hdr, c))
    return out
L = json.load(open("layer.json")); res = {}
for cfg in ["int8", "lean", "lean_ecc"]:
    R = f"reports_{cfg}"
    if not os.path.exists(os.path.join(R, "impl_utilization.rpt")): continue
    u = util(os.path.join(R, "impl_utilization.rpt")); w = timing(os.path.join(R, "impl_timing.rpt")); pw = power(os.path.join(R, "impl_power.rpt"))
    res[cfg] = (u, w, pw, hier(os.path.join(R, "impl_utilization_hier.rpt")) if os.path.exists(os.path.join(R, "impl_utilization_hier.rpt")) else {})
name = {"int8": "INT8 baseline (plain BRAM + read-out)", "lean": "LEAN loader", "lean_ecc": "LEAN loader + CRC-14 parity checker"}
print(f"layer: tensor {L['tensor']}, {L['weights']:,} weights, LEAN {L['bits_per_weight']:.2f} b/w\n")
print("| design | LUT | FF | BRAM36 | DSP | WNS @100 MHz | Fmax | dynamic power | BRAM power | logic+signal power | static (chip) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for cfg, (u, w, pw, h) in res.items():
    fmax = 1000 / (10.0 - w) if w == w else float("nan")
    print(f"| {name[cfg]} | {u['lut'][0]:,} ({u['lut'][2]}%) | {u['ff'][0]:,} ({u['ff'][2]}%) | {u['bram'][0]} ({u['bram'][2]}%) | {u['dsp'][0]} | {w:+.3f} ns | {fmax:.0f} MHz | {pw[1]} W | {pw[3]} W | {pw[5]} + {pw[6]} W | {pw[2]} W |")
if "int8" in res and "lean" in res:
    b, l = res["int8"][0], res["lean"][0]; e = res.get("lean_ecc", (None,))[0]
    print("\noverhead of LEAN relative to the INT8 baseline (same layer, same output rate):")
    print("| resource | INT8 | LEAN | LEAN + ECC | LEAN delta | LEAN + ECC delta |")
    print("|---|---|---|---|---|---|")
    for k, lab in [("lut", "LUT"), ("ff", "FF"), ("bram", "BRAM36"), ("dsp", "DSP")]:
        bv, lv, ev = b[k][0], l[k][0], (e[k][0] if e else None)
        d = lambda v: (f"{v - bv:+,} ({100 * (v - bv) / bv:+.1f}%)" if bv else f"{v - bv:+,}") if v is not None else "n/a"
        print(f"| {lab} | {bv:,} | {lv:,} | {ev if ev is not None else 'n/a'} | {d(lv)} | {d(ev)} |")
    pb, pl = res["int8"][2][1], res["lean"][2][1]; pe = res["lean_ecc"][2][1] if "lean_ecc" in res else "n/a"
    print(f"| dynamic power (W) | {pb} | {pl} | {pe} | | |")
for cfg in ["lean", "lean_ecc"]:
    if cfg not in res or not res[cfg][3]: continue
    h = res[cfg][3]
    print(f"\nper-module split, {name[cfg]} (from the hierarchical report):")
    print("| instance | LUT | FF | BRAM36 | BRAM18 | DSP |")
    print("|---|---|---|---|---|---|")
    g = lambda c, *ks: next((c[k] for k in ks if k in c), "?")
    for inst, c in h.items():
        print(f"| {inst} | {g(c, 'Total LUTs', 'LUTs')} | {g(c, 'FFs', 'Registers')} | {g(c, 'RAMB36', 'Block RAM Tile')} | {g(c, 'RAMB18')} | {g(c, 'DSP Blocks', 'DSPs', 'DSP58')} |")
print("\nwhole ResNet-50 BRAM36 by bit count, all tensors: see bram_table.md")
