#!/bin/bash
# run_zcu.sh  -  the complete FPGA flow again on the ZCU102 part (xczu9eg, Zynq UltraScale+) with Vivado 2020.2.
# Run on the server from hw/:   bash fpga/run_zcu.sh 2>&1 | tee fpga/run_zcu.log
# 1. keeps the Versal results (reports_*, tables, schematics, xsim) in fpga/versal_results/
# 2. loader runs: int8 / lean / lean_ecc on the largest layer that fits      3. gate-level xsim check
# 4. accelerator alone, then accelerator + LEAN with a smaller layer          5. SoC block design (MicroBlaze) diagram
set -e
cd "$(dirname "$0")"
VIV=/apps/xilinx/Vivado/2020.2
[ -x $VIV/bin/vivado ] || { echo "Vivado 2020.2 not found at $VIV"; exit 1; }
# ---- 1. preserve Versal results
if [ ! -d versal_results ]; then
  mkdir -p versal_results
  for d in reports_int8 reports_lean reports_lean_ecc reports_accel reports_accel_lean xsim schematic; do [ -e $d ] && mv $d versal_results/; done
  for f in fpga_table.md fpga_table_layer44.md accel_table.md layer44.json vivado_*.log xsim_impl.log run.log run_accel.log run_xsim.log; do [ -e $f ] && cp $f versal_results/ 2>/dev/null || true; done
  echo "Versal results moved to fpga/versal_results/"
fi
# ---- ZCU102 first in the part preference of every script (ZCU104 has too few BRAMs for the accelerator)
for t in vivado_lean.tcl vivado_accel.tcl xsim_lean.tcl ../fpga_soc/soc_build.tcl; do
  [ -f $t ] && sed -i 's/^set PREF {xczu7ev- xczu9eg- /set PREF {xczu9eg- xczu7ev- /' $t
done
source /apps/settings
export PATH=$VIV/bin:$PATH
export XILINX_VIVADO=$VIV
cd ..
VEC=$(ls -d vec_* | head -1); for v in vec_*; do grep -q '"block": 32' $v/format.json 2>/dev/null && VEC=$v; done
echo "packed image: $VEC"
# ---- 2. loader: largest layer within 250 BRAM36
python fpga/lean_fpga_prep.py --vec $VEC --out fpga --max-bram36 250
cd fpga
for CFG in int8 lean lean_ecc; do echo "=== vivado $CFG (ZCU102) ==="; CFG=$CFG vivado -mode batch -nojournal -nolog -source vivado_lean.tcl 2>&1 | tee vivado_$CFG.log; done
python fpga_table.py | tee fpga_table.md
# ---- 3. gate-level check of the routed loader
NL=$(grep NLINES_INT8 layer.tcl | awk '{print $3}'); NW=$(python -c "import json;print(json.load(open('layer.json'))['weights'])")
vivado -mode batch -nojournal -nolog -source xsim_lean.tcl 2>&1 | tee xsim_impl.log
( cd xsim && cp ../layer_int8.hex . && xvlog ../tb_lean_fpga_check.v lean_fpga_top_routed.v $XILINX_VIVADO/data/verilog/src/glbl.v > xvlog.log 2>&1 && \
  xelab -L unisims_ver -L simprims_ver -timescale 1ns/1ps -generic_top "NLINES_INT8=$NL" -generic_top "NWORDS=$NW" tb_lean_fpga_check glbl -s sim > xelab.log 2>&1 && \
  xsim sim -R 2>&1 | tee xsim.log | grep "XSIM RESULT\|XSIM TIMEOUT" )
# ---- 4. accelerator alone, then accelerator + LEAN (smaller layer so both fit in 912 BRAM36)
cp fpga_table.md fpga_table_layer_loader.md; cp layer.json layer_loader.json
CFG=accel vivado -mode batch -nojournal -nolog -source vivado_accel.tcl 2>&1 | tee vivado_accel.log
cd ..; python fpga/lean_fpga_prep.py --vec $VEC --out fpga --max-bram36 100; cd fpga
CFG=accel_lean vivado -mode batch -nojournal -nolog -source vivado_accel.tcl 2>&1 | tee vivado_accel_lean.log
python accel_table.py | tee accel_table.md
# ---- 5. SoC block design + diagram
cd ../fpga_soc && vivado -mode batch -nojournal -nolog -source soc_build.tcl 2>&1 | tee soc_build.log; ls -la diagram/ || true
echo "ZCU FLOW DONE"
