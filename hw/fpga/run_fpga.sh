#!/bin/bash
# run on the server from the hw/ folder:   bash fpga/run_fpga.sh
# picks the block-32 packed image (hw/vec_*) automatically; or give a folder:  bash fpga/run_fpga.sh vec_xxx
set -e
cd "$(dirname "$0")/.."
VEC=${1:-}
if [ -z "$VEC" ]; then
  for v in vec_*; do
    [ -f "$v/tensors.txt" ] || continue
    if grep -q '"block": 32' "$v/format.json" 2>/dev/null; then VEC=$v; fi
    [ -z "$VEC" ] && VEC=$v
  done
fi
[ -n "$VEC" ] && [ -f "$VEC/tensors.txt" ] || { echo "no packed image found (hw/vec_*/tensors.txt). run the agent pack step first."; exit 1; }
echo "using packed image: $VEC"; cat "$VEC/format.json" | tr -d '\n'; echo
python fpga/lean_fpga_prep.py --vec "$VEC" --out fpga --max-bram36 ${MAXBRAM:-250}
cd fpga
source /apps/settings
export PATH=/apps/xilinx/Vivado/2020.3/bin:$PATH
for CFG in int8 lean lean_ecc; do
  echo "=== vivado config $CFG ==="
  CFG=$CFG vivado -mode batch -nojournal -nolog -source vivado_lean.tcl 2>&1 | tee vivado_$CFG.log
done
python fpga_table.py | tee fpga_table.md
