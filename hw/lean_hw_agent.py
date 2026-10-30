#!/usr/bin/env python
"""
lean_hw_agent.py  -  one driver for the LEAN hardware flow on the server (server).  It FINDS the paths itself, shows them, asks you
to confirm or correct each one once, then runs the chain with live logs:

   1. pack      lean_pack.py on the real LEAN checkpoint (+ its tier/bins file)  -> hw/vec_<tag>/ (image, map, golden vectors, report)
   2. sim       VCS (or iverilog) on tb_lean_loader.v: clean run must PASS; then bit-flip injection runs (+FLIPS=50/200/1000)
   3. syn       Design Compiler on lean_weight_loader with the the selective code library settings (standalone block, minutes)
   4. table     hw/results/lean_hw_<tag>.md : bits/weight, SRAM area/energy (paper's storage model), loader area/power/timing,
                throughput and detection numbers from the simulations

usage:  python lean_hw_agent.py                  (interactive: discover, confirm, run all)
        python lean_hw_agent.py --yes            (take every discovered path without asking)
        python lean_hw_agent.py --step sim       (one step; pack|sim|syn|table)
        python lean_hw_agent.py --dry-run        (print the commands only)
Run it from the hw/ folder (the one holding lean_pack.py, rtl/, tb/, syn/).  Every step writes hw/logs/<step>.log; watch with tail -f.
"""
import os, sys, re, glob, json, shutil, argparse, subprocess, time

HOME = os.path.expanduser("~")
HW = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HW, "logs"); os.makedirs(LOGS, exist_ok=True)

# paper storage model constants (Horowitz ISSCC 2014; 6T cell at 28 nm), same as the LEAN paper's storage section
SRAM_PJ_PER_BIT = 0.16; DRAM_PJ_PER_BIT = 20.0; CELL_UM2 = 0.127


def find_first(patterns, roots, maxdepth=8):
    """first path matching any glob pattern under any root (bounded depth), or None"""
    for root in roots:
        if not root or not os.path.isdir(root): continue
        for pat in patterns:
            for depth in range(maxdepth):
                hits = glob.glob(os.path.join(root, *(["*"] * depth), pat))
                if hits: return sorted(hits, key=os.path.getmtime, reverse=True)[0]
    return None


def find_all(patterns, roots, maxdepth=8, limit=12):
    out = []
    for root in roots:
        if not root or not os.path.isdir(root): continue
        for pat in patterns:
            for depth in range(maxdepth):
                out += glob.glob(os.path.join(root, *(["*"] * depth), pat))
    out = sorted(set(out), key=os.path.getmtime, reverse=True)
    return out[:limit]


# known from the the selective code hardware sessions on server (checked first, then searched if absent)
KNOWN_RTL = ["./",
             os.path.join(HOME, "ANON/RTL-Neural-Network/RTL-Code-Neural-Network/STDM_RTL")]
TOOL_ENV = "/apps/settings"            # `source /apps/settings` puts vcs and dc_shell on PATH on the server


def tool_on_path(name):
    """which <name> after sourcing the tool environment"""
    try:
        out = subprocess.run(f"source {TOOL_ENV} >/dev/null 2>&1; which {name}", shell=True, executable="/bin/bash", capture_output=True, text=True, timeout=60).stdout.strip()
        return out if out else None
    except Exception: return None


def say(msg): print(f"[agent] {msg}", flush=True)


def discover():
    d = {}
    say("looking for the the selective code RTL folder")
    d["rtl_selcode"] = next((p for p in KNOWN_RTL if os.path.isdir(p)), None)
    if d["rtl_selcode"] is None:
        proj = find_first(["ANON"], [HOME], 2) or HOME
        d["rtl_selcode"] = find_first(["STDM_RTL"], [proj], 4)
    say(f"RTL folder: {d['rtl_selcode']}"); say("looking for a the selective code DC script and gscl45nm.db")
    tcls = find_all(["*.tcl"], [os.path.join(d["rtl_selcode"], "syn")] if d["rtl_selcode"] else [], 2, 60)
    d["dc_tcl_selcode"] = next((t for t in tcls if re.search(r"target_library", open(t, errors="ignore").read())), None)
    # library: gscl45nm.db lives in STDM_RTL (top level, silicon/ or syn/); the selective code scripts use search_path + plain file names
    db = find_first(["gscl45nm.db"], [d["rtl_selcode"]] if d["rtl_selcode"] else [], 3)
    d["lib_dir"] = os.path.dirname(db) if db else None
    d["target_lib"] = "gscl45nm.db" if db else None
    d["link_lib"] = "gscl45nm.db" if db else None
    if d["dc_tcl_selcode"] and not db:
        txt = open(d["dc_tcl_selcode"], errors="ignore").read()
        m = re.search(r"set\s+search_path\s+(.+)", txt); d["lib_dir"] = m.group(1).strip() if m else None
        m = re.search(r"set\s+target_library\s+\"?([^\"\n]+)", txt); d["target_lib"] = m.group(1).strip() if m else None
        m = re.search(r"set\s+link_library\s+\"?\*?\s*([^\"\n]+)", txt); d["link_lib"] = m.group(1).strip() if m else None
    say(f"library: {d['lib_dir']} / {d['target_lib']}")
    d["vcs"] = shutil.which("vcs"); d["dc_shell"] = shutil.which("dc_shell"); d["iverilog"] = shutil.which("iverilog")
    if not (d["vcs"] and d["dc_shell"]):
        say(f"vcs/dc_shell not on PATH; sourcing {TOOL_ENV} once to find them (can take a while; run `source {TOOL_ENV}` in your shell first to skip this)")
        both = tool_on_path("vcs dc_shell")
        if both:
            for line in both.splitlines():
                if line.endswith("/vcs"): d["vcs"] = d["vcs"] or line
                if line.endswith("/dc_shell"): d["dc_shell"] = d["dc_shell"] or line
    say(f"vcs: {d['vcs']}   dc_shell: {d['dc_shell']}")
    allf = os.path.dirname(HW)                                   # this script lives in All_final/hw
    d["all_final"] = allf
    say(f"looking for checkpoints with a .bins.pt tier file under {allf}/artifacts")
    d["bins"] = find_all(["*.bins.pt"], [os.path.join(allf, "artifacts"), os.path.join(allf, "ckpt_llm")], 4, 8)
    say(f"found {len(d['bins'])} tier files")
    d["ckpt_candidates"] = [b[:-len(".bins.pt")] for b in d["bins"] if os.path.isfile(b[:-len(".bins.pt")])]
    d["python"] = sys.executable
    return d


def confirm(d, yes):
    print("\n=== discovered ===")
    keys = ["rtl_selcode", "dc_tcl_selcode", "lib_dir", "target_lib", "link_lib", "vcs", "dc_shell", "iverilog", "all_final", "python"]
    for k in keys: print(f"  {k:14s} {d.get(k)}")
    print("  checkpoints with a tier file (newest first):")
    for i, c in enumerate(d["ckpt_candidates"]): print(f"     [{i}] {c}")
    if not d["ckpt_candidates"]: print("     none found: give --ckpt and --bins")
    if yes: return d
    print("\nPress Enter to accept a value, or type the correct path.")
    for k in keys:
        v = input(f"  {k} [{d.get(k)}]: ").strip()
        if v: d[k] = v
    if d["ckpt_candidates"]:
        v = input(f"  checkpoint index or path [0]: ").strip()
        if v.isdigit(): d["ckpt"] = d["ckpt_candidates"][int(v)]
        elif v: d["ckpt"] = v
        else: d["ckpt"] = d["ckpt_candidates"][0]
        d["bins_file"] = d["ckpt"] + ".bins.pt" if os.path.isfile(d["ckpt"] + ".bins.pt") else None
    return d


def run(cmd, log, dry, cwd=HW, env=None):
    print(f"\n$ {cmd}\n  log: {log}   (tail -f {log})", flush=True)
    if dry: return 0
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
        while p.poll() is None: time.sleep(2)
    print(f"  exit {p.returncode}")
    return p.returncode


def grep(path, pat, default=None):
    try:
        for line in open(path, errors="ignore"):
            m = re.search(pat, line)
            if m: return m.group(1)
    except FileNotFoundError: pass
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--step", choices=["pack", "sim", "syn", "table", "all"], default="all")
    ap.add_argument("--ckpt"); ap.add_argument("--bins"); ap.add_argument("--tiers", default="8,4,2"); ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--tag", default=None); ap.add_argument("--nweights", type=float, default=None, help="weights in the model (for the SRAM numbers); read from report if omitted")
    a = ap.parse_args()
    d = discover()
    if a.ckpt: d["ckpt_candidates"] = [a.ckpt] + d["ckpt_candidates"]
    d = confirm(d, a.yes)
    if a.ckpt: d["ckpt"] = a.ckpt
    if a.bins: d["bins_file"] = a.bins
    if "ckpt" not in d: d["ckpt"] = d["ckpt_candidates"][0] if d["ckpt_candidates"] else None
    if "bins_file" not in d: d["bins_file"] = (d["ckpt"] + ".bins.pt") if d.get("ckpt") and os.path.isfile(d["ckpt"] + ".bins.pt") else None
    tag = a.tag or (os.path.basename(d["ckpt"]).replace(".pth", "") if d.get("ckpt") else "demo") + "_t" + a.tiers.replace(",", "")
    vec = os.path.join(HW, f"vec_{tag}"); json.dump(d, open(os.path.join(LOGS, "discovered.json"), "w"), indent=1, default=str)
    tiers = [int(x) for x in a.tiers.split(",")]; W = tiers + [tiers[-1]] * (4 - len(tiers))
    steps = ["pack", "sim", "syn", "table"] if a.step == "all" else [a.step]

    if "pack" in steps:
        if d.get("ckpt"):
            cmd = f"{d['python']} -u lean_pack.py --ckpt {d['ckpt']} " + (f"--bins {d['bins_file']} " if d.get("bins_file") else "") + f"--tiers {a.tiers} --block {a.block} --out {vec}"
        else:
            print("no checkpoint: packing the synthetic demo (group-wise tiers)"); cmd = f"{d['python']} -u lean_pack.py --demo-group --tiers {a.tiers} --block {a.block} --out {vec}"
        if run(cmd, os.path.join(LOGS, "pack.log"), a.dry_run): sys.exit("pack failed, see logs/pack.log")

    if "sim" in steps:
        P = f"BLOCK={a.block} LINE_W=16 PAR=14 OUTW=8 W0={W[0]} W1={W[1]} W2={W[2]} W3={W[3]}"
        if d.get("vcs"):
            pdef = " ".join(f"-pvalue+tb_lean_loader.{p}" for p in P.split())
            build = f"source {TOOL_ENV} >/dev/null 2>&1; {d['vcs']} -full64 -sverilog +v2k {pdef} rtl/lean_unpack.v rtl/lean_weight_loader.v tb/tb_lean_loader.v -o logs/simv_{tag}"
            simv = f"./logs/simv_{tag}"
        elif d.get("iverilog"):
            pdef = " ".join(f"-Ptb_lean_loader.{p}" for p in P.split())
            build = f"{d['iverilog']} -g2005 {pdef} -o logs/simv_{tag} rtl/lean_unpack.v rtl/lean_weight_loader.v tb/tb_lean_loader.v"
            simv = f"vvp logs/simv_{tag}"
        else: sys.exit("no simulator found (vcs or iverilog)")
        if run(build, os.path.join(LOGS, "sim_build.log"), a.dry_run): sys.exit("simulator build failed, see logs/sim_build.log")
        run(f"{simv} +DIR={vec}", os.path.join(LOGS, f"sim_clean_{tag}.log"), a.dry_run)
        for nf in (50, 200, 1000):
            run(f"{simv} +DIR={vec} +FLIPS={nf} +SEED=3", os.path.join(LOGS, f"sim_flips{nf}_{tag}.log"), a.dry_run)

    if "syn" in steps:
        if not d.get("dc_shell"): print("dc_shell not found: skipping synthesis (run syn/dc_lean_loader.tcl by hand where DC is available)")
        else:
            tcl = open(os.path.join(HW, "syn", "dc_lean_loader.tcl")).read()
            for k, v in (("@LIB_DIR@", d.get("lib_dir") or ""), ("@TARGET_LIB@", d.get("target_lib") or ""), ("@LINK_LIB@", d.get("link_lib") or "")): tcl = tcl.replace(k, str(v))
            tcl = re.sub(r'set PARAMS\s+".*"', f'set PARAMS      "BLOCK={a.block},LINE_W=16,PAR=14,OUTW=8,W0={W[0]},W1={W[1]},W2={W[2]},W3={W[3]}"', tcl)
            filled = os.path.join(HW, "syn", f"dc_lean_loader_{tag}.tcl"); open(filled, "w").write(tcl)
            if not (d.get("target_lib") and d.get("lib_dir")): print("  WARNING: library settings were not found in a the selective code tcl; edit the three set lines in " + filled)
            run(f"source {TOOL_ENV} >/dev/null 2>&1; {d['dc_shell']} -f {filled}", os.path.join(LOGS, f"dc_{tag}.log"), a.dry_run)

    if "table" in steps:
        rep = os.path.join(vec, "report.txt")
        bw = grep(rep, r"=\s+([\d.]+)\s+\(INT8", None); body = grep(rep, r"body\s+([\d.]+)"); mapb = grep(rep, r"map\s+([\d.]+)"); parb = grep(rep, r"parity\s+([\d.]+)\s+=")
        nw = a.nweights or float(grep(rep, r"weights\s+([\d,]+)", "0").replace(",", ""))
        clean = grep(os.path.join(LOGS, f"sim_clean_{tag}.log"), r"\[tb\] (PASS|FAIL.*)", "not run"); thr = grep(os.path.join(LOGS, f"sim_clean_{tag}.log"), r"([\d.]+) words/cycle", "?")
        flips = {nf: grep(os.path.join(LOGS, f"sim_flips{nf}_{tag}.log"), r"FLIP SUMMARY: (.*)", "not run") for nf in (50, 200, 1000)}
        def rpt(kind):   # DC names the elaborated design with a parameter suffix; take the newest matching report
            hits = sorted(glob.glob(os.path.join(HW, f"syn/reports/lean_weight_loader*_{kind}.rpt")), key=os.path.getmtime)
            return hits[-1] if hits else os.path.join(HW, f"syn/reports/lean_weight_loader_{kind}.rpt")
        area = grep(rpt("area"), r"Total cell area:\s+([\d.]+)", "to be measured")
        power = grep(rpt("power"), r"Total Dynamic Power\s+=\s+([\d.]+\s*\w+)", "to be measured")
        slack = grep(rpt("timing"), r"slack \((\w+)\)", "to be measured")
        def sram(bits_per_w):
            if not bits_per_w or not nw: return ("?", "?", "?")
            bits = float(bits_per_w) * nw; return (f"{bits * CELL_UM2 / 1e6:.3f}", f"{bits * SRAM_PJ_PER_BIT / 1e6:.2f}", f"{bits * DRAM_PJ_PER_BIT / 1e6:.1f}")
        a8, e8, d8 = sram(8.0); al, el, dl = sram(bw)
        md = [f"# LEAN weight loader: {tag}", "", f"checkpoint: {d.get('ckpt')}   tiers {a.tiers}   block {a.block}   weights {nw:,.0f}", "",
              "| | INT8 uniform | LEAN packed tiers + selective ECC |", "|---|---|---|",
              f"| bits / weight | 8.000 | {bw} (body {body} + map {mapb} + parity {parb}) |",
              f"| weight SRAM area, 6T 28 nm (mm2) | {a8} | {al} |",
              f"| weight fetch energy per full read, on-chip SRAM (uJ) | {e8} | {el} |",
              f"| weight fetch energy per full read, DRAM (uJ) | {d8} | {dl} |",
              f"| loader (tier decoder + unpacker) cell area (um2, gscl45nm) | none | {area} |",
              f"| loader dynamic power at 100 MHz | none | {power} |", f"| timing at 100 MHz | - | {slack} |",
              f"| words delivered per cycle to the 8x8 array | 8 | {thr} |", f"| clean simulation | - | {clean} |",
              "", "Bit-flip injection into the packed image (testbench CRC-14 checker model stands in for the the selective code BCH checker):", ""]
        md += [f"- {nf} flips: {flips[nf]}" for nf in (50, 200, 1000)]
        md += ["", "SRAM area and fetch energies are the paper's storage model (6T cell 0.127 um2, SRAM 0.16 pJ/bit, DRAM 20 pJ/bit, Horowitz 2014) applied to the measured bits/weight; checker area and energy come from the existing selective-ECC synthesis and are added by hand."]
        os.makedirs(os.path.join(HW, "results"), exist_ok=True); out = os.path.join(HW, "results", f"lean_hw_{tag}.md")
        open(out, "w").write("\n".join(md) + "\n"); print("\n".join(md)); print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
