#!/bin/bash
# run from hw/ on the server:   bash fpga/run_cycles.sh
# Behavioural RTL simulation (xsim) of the LEAN loader and the INT8 read-out on the layer prepared by run_fpga.sh (fpga/layer.json,
# layer_*.hex).  Prints CYCLES LEAN / CYCLES INT8 lines with latency, total cycles, active cycles, words per cycle and time at 100 MHz.
# No implementation run, no design change; takes a few minutes.   Log: fpga/cycles/xsim_cycles.log
set -e
cd "$(dirname "$0")"
[ -f layer.json ] || { echo "fpga/layer.json missing: run fpga/run_fpga.sh first"; exit 1; }
source /apps/settings
export PATH=/apps/xilinx/Vivado/2020.3/bin:$PATH
G=$(python - <<'EOF'
import json; j=json.load(open("layer.json")); t=j["tiers"]; r=j["ratios"]
import os
nl8=(j["weights"]+15)//16
print(" ".join(f'-generic_top "{k}={v}"' for k,v in [("BLOCK",j["block"]),("PAR",j["par"]),("NBLOCKS",j["nblocks"]),("NLINES",j["nlines"]),("NLINES_INT8",nl8),
      ("W0",t[0]),("W1",t[1]),("W2",t[2]),("W3",t[3]),("R0",r[0]),("R1",r[1]),("R2",r[2]),("R3",r[3]),("CHECK",1)]))
EOF
)
echo "generics: $G"
mkdir -p cycles; cd cycles
cp ../layer_image.hex ../layer_map.hex ../layer_int8.hex .
xvlog ../../rtl/lean_unpack.v ../../rtl/lean_weight_loader.v ../lean_crc14_check.v ../lean_fpga_top.v ../lean_fpga_int8_top.v ../tb_lean_cycles.v 2>&1 | tee xvlog.log
eval xelab -timescale 1ns/1ps $G tb_lean_cycles -s simc 2>&1 | tee xelab.log
xsim simc -R 2>&1 | tee xsim_cycles.log
grep "^CYCLES" xsim_cycles.log
