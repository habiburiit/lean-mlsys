#!/bin/bash
# run on the server from the hw/ folder AFTER run_fpga.sh (needs fpga/layer.tcl and the layer hex files):   bash fpga/run_fpga_accel.sh
set -e
cd "$(dirname "$0")"
[ -f layer.tcl ] || { echo "run fpga/run_fpga.sh first (it writes layer.tcl)"; exit 1; }
source /apps/settings
export PATH=/apps/xilinx/Vivado/2020.3/bin:$PATH
for CFG in accel accel_lean; do
  echo "=== vivado config $CFG ==="
  CFG=$CFG vivado -mode batch -nojournal -nolog -source vivado_accel.tcl 2>&1 | tee vivado_$CFG.log
done
python accel_table.py | tee accel_table.md
