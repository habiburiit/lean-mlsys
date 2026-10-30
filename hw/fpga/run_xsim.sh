#!/bin/bash
# run on the server from hw/:   bash fpga/run_xsim.sh      -> implements the LEAN loader, simulates the ROUTED netlist in xsim, prints XSIM RESULT PASS/FAIL
set -e
cd "$(dirname "$0")"
[ -f layer.tcl ] || { echo "run fpga/run_fpga.sh first"; exit 1; }
source /apps/settings
export PATH=/apps/xilinx/Vivado/2020.3/bin:$PATH
vivado -mode batch -nojournal -nolog -source xsim_lean.tcl 2>&1 | tee xsim_impl.log
NL=$(grep NLINES_INT8 layer.tcl | awk '{print $3}'); NW=$(python -c "import json;print(json.load(open('layer.json'))['weights'])")
cd xsim
cp ../layer_int8.hex .
xvlog ../tb_lean_fpga_check.v lean_fpga_top_routed.v $XILINX_VIVADO/data/verilog/src/glbl.v 2>&1 | tee xvlog.log
xelab -L unisims_ver -L simprims_ver -timescale 1ns/1ps -generic_top "NLINES_INT8=$NL" -generic_top "NWORDS=$NW" tb_lean_fpga_check glbl -s sim 2>&1 | tee xelab.log
xsim sim -R 2>&1 | tee xsim.log
grep "XSIM RESULT\|XSIM TIMEOUT" xsim.log
