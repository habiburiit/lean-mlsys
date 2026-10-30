#!/bin/bash
# run_zcu2.sh  -  remaining ZCU102 flow, every step independent (a failure is reported and the next step still runs).
#   Run on the server from hw/ in the ssh -X session:   bash fpga/run_zcu2.sh 2>&1 | tee fpga/run_zcu2.log
#   A. SoC block design on ZCU102 + IP figure (10 min; the figure needs the X window, opened and closed automatically)
#   B. accelerator alone, C. accelerator + LEAN + ECC (hours), D. overhead table, E. gate-level xsim retry (optional)
cd "$(dirname "$0")"
VIV=/apps/xilinx/Vivado/2020.2
source /apps/settings
export PATH=$VIV/bin:$PATH
export XILINX_VIVADO=$VIV
STATUS=()
step() { echo; echo "=================== $1   $(date)"; }
done_ok() { STATUS+=("$1: OK"); }
done_bad() { STATUS+=("$1: FAILED ($2)"); echo "!!! $1 failed: $2"; }
# stale GUI sessions from the schematic scripts
pkill -u $USER -f "schematic.tcl" 2>/dev/null; pkill -u $USER -f "schematic_accel.tcl" 2>/dev/null; sleep 1
# ZCU102 first in every script's part preference
for t in vivado_lean.tcl vivado_accel.tcl xsim_lean.tcl ../fpga_soc/soc_build.tcl; do [ -f $t ] && sed -i 's/^set PREF {xczu7ev- xczu9eg- /set PREF {xczu9eg- xczu7ev- /' $t; done
cd ..
VEC=$(ls -d vec_* 2>/dev/null | head -1); for v in vec_*; do grep -q '"block": 32' $v/format.json 2>/dev/null && VEC=$v; done
[ -n "$VEC" ] || { echo "no packed image hw/vec_*"; exit 1; }
echo "packed image: $VEC   vivado: $VIV"

# ---- A. SoC block design + figure
step "A. SoC block design (ZCU102)"
[ -f fpga_soc/diagram/lean_soc_block_design.pdf ] && [ ! -f fpga_soc/diagram/lean_soc_block_design_versal.pdf ] && cp fpga_soc/diagram/lean_soc_block_design.pdf fpga_soc/diagram/lean_soc_block_design_versal.pdf
( cd fpga_soc && vivado -mode batch -nojournal -nolog -source soc_build.tcl > soc_build_zcu.log 2>&1 )
if grep -q "BLOCK DESIGN DONE" fpga_soc/soc_build_zcu.log; then
  done_ok "A. SoC block design"
  if [ -n "$DISPLAY" ]; then
    cat > fpga_soc/export_diagram_auto.tcl <<'EOT'
set HERE [file normalize [file dirname [info script]]]
open_project $HERE/soc_proj/soc_proj.xpr
open_bd_design [get_files lean_soc.bd]
start_gui
regenerate_bd_layout
file mkdir $HERE/diagram
write_bd_layout -force -format pdf -orientation landscape $HERE/diagram/lean_soc_block_design.pdf
write_bd_layout -force -format svg -orientation landscape $HERE/diagram/lean_soc_block_design.svg
puts "DIAGRAM WRITTEN"
stop_gui
exit
EOT
    vivado -mode tcl -source fpga_soc/export_diagram_auto.tcl > fpga_soc/export_diagram.log 2>&1
    if grep -q "DIAGRAM WRITTEN" fpga_soc/export_diagram.log; then done_ok "A2. IP figure fpga_soc/diagram/lean_soc_block_design.pdf"; else done_bad "A2. IP figure" "see fpga_soc/export_diagram.log"; fi
  else
    done_bad "A2. IP figure" "no X display; run: vivado -mode tcl -source fpga_soc/export_diagram.tcl in an ssh -X session"
  fi
else
  done_bad "A. SoC block design" "see fpga_soc/soc_build_zcu.log"
fi

# ---- B. accelerator alone
step "B. accelerator alone"
cd fpga
[ -f fpga_table.md ] && cp fpga_table.md fpga_table_layer_loader.md; [ -f layer.json ] && cp layer.json layer_loader.json
CFG=accel vivado -mode batch -nojournal -nolog -source vivado_accel.tcl > vivado_accel.log 2>&1
if grep -q "VIVADO DONE" vivado_accel.log; then done_ok "B. accelerator alone"; else done_bad "B. accelerator alone" "see fpga/vivado_accel.log"; fi

# ---- C. accelerator + LEAN (smaller layer so both fit the 912 BRAM36)
step "C. accelerator + LEAN + ECC"
cd ..; python fpga/lean_fpga_prep.py --vec $VEC --out fpga --max-bram36 100; cd fpga
CFG=accel_lean vivado -mode batch -nojournal -nolog -source vivado_accel.tcl > vivado_accel_lean.log 2>&1
if grep -q "VIVADO DONE" vivado_accel_lean.log; then done_ok "C. accelerator + LEAN"; else done_bad "C. accelerator + LEAN" "see fpga/vivado_accel_lean.log"; fi

# ---- D. table
step "D. overhead table"
if python accel_table.py > accel_table.md 2>&1; then cat accel_table.md; done_ok "D. table fpga/accel_table.md"; else done_bad "D. table" "see fpga/accel_table.md"; fi

# ---- E. gate-level xsim retry on the small layer (optional; Versal gate-level PASS already stands)
step "E. gate-level simulation retry"
NL=$(grep NLINES_INT8 layer.tcl | awk '{print $3}'); NW=$(python -c "import json;print(json.load(open('layer.json'))['weights'])")
vivado -mode batch -nojournal -nolog -source xsim_lean.tcl > xsim_impl.log 2>&1
( cd xsim && cp ../layer_int8.hex . && xvlog ../tb_lean_fpga_check.v lean_fpga_top_routed.v $XILINX_VIVADO/data/verilog/src/glbl.v > xvlog.log 2>&1 && \
  xelab -L unisims_ver -L simprims_ver -timescale 1ns/1ps -generic_top "NLINES_INT8=$NL" -generic_top "NWORDS=$NW" tb_lean_fpga_check glbl -s sim > xelab.log 2>&1 && \
  xsim sim -R > xsim.log 2>&1 && grep -q "PASS" xsim.log ) && done_ok "E. gate-level sim: $(grep 'XSIM RESULT' xsim/xsim.log)" || done_bad "E. gate-level sim" "optional; see fpga/xsim/xelab.log"

echo; echo "=================== SUMMARY   $(date)"; printf '%s\n' "${STATUS[@]}"; echo "ZCU FLOW DONE"
