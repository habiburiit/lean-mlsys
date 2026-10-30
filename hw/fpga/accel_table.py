#!/usr/bin/env python
"""accel_table.py  -  LEAN overhead relative to the full accelerator, from fpga/reports_accel and fpga/reports_accel_lean."""
import re, os, json
exec(open("fpga_table.py").read().split("L = json.load")[0])          # reuse util/timing/power/hier parsers
res = {}
for cfg in ["accel", "accel_lean"]:
    R = f"reports_{cfg}"
    if os.path.exists(os.path.join(R, "impl_utilization.rpt")):
        res[cfg] = (util(os.path.join(R, "impl_utilization.rpt")), timing(os.path.join(R, "impl_timing.rpt")), power(os.path.join(R, "impl_power.rpt")), hier(os.path.join(R, "impl_utilization_hier.rpt")))
name = {"accel": "systolic accelerator alone", "accel_lean": "accelerator + LEAN loader + ECC checker (+ one packed layer in BRAM)"}
print("| design | LUT | FF | BRAM36 | DSP | WNS @100 MHz | dynamic power |"); print("|---|---|---|---|---|---|---|")
for cfg, (u, w, pw, h) in res.items():
    print(f"| {name[cfg]} | {u['lut'][0]:,} ({u['lut'][2]}%) | {u['ff'][0]:,} ({u['ff'][2]}%) | {u['bram'][0]} | {u['dsp'][0]} | {w:+.3f} ns | {pw[1]} W |")
if len(res) == 2:
    a, l = res["accel"][0], res["accel_lean"][0]; h = res["accel_lean"][3]
    g = lambda c, *ks: next((c[k] for k in ks if k in c), "?")
    print("\nLEAN overhead relative to the full accelerator:"); print("| resource | accelerator | + LEAN | overhead |"); print("|---|---|---|---|")
    for k, lab in [("lut", "LUT"), ("ff", "FF"), ("dsp", "DSP")]:
        print(f"| {lab} | {a[k][0]:,} | {l[k][0]:,} | +{l[k][0]-a[k][0]:,} ({100*(l[k][0]-a[k][0])/max(a[k][0],1):.1f}%) |")
    print(f"| BRAM36 | {a['bram'][0]} | {l['bram'][0]} | +{l['bram'][0]-a['bram'][0]} (the packed layer; loader logic itself uses none) |")
    print(f"| dynamic power (W) | {res['accel'][2][1]} | {res['accel_lean'][2][1]} | |")
    print("\nper-module (accelerator + LEAN):"); print("| instance | LUT | FF | BRAM36 | DSP |"); print("|---|---|---|---|---|")
    for inst, c in h.items():
        print(f"| {inst} | {g(c,'Total LUTs','LUTs')} | {g(c,'FFs','Registers')} | {g(c,'RAMB36','Block RAM Tile')} | {g(c,'DSP Blocks','DSPs','DSP58')} |")
