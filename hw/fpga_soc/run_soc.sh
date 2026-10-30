#!/bin/bash
# run on the server from hw/:   bash fpga_soc/run_soc.sh          (block design + diagram, about 10 minutes)
#                        SYNTH=1 bash fpga_soc/run_soc.sh  (also implements the whole system, hours)
set -e
cd "$(dirname "$0")"
[ -f ../fpga/layer.tcl ] || { echo "run fpga/run_fpga.sh first (needs fpga/layer.tcl and the layer hex files)"; exit 1; }
source /apps/settings
export PATH=/apps/xilinx/Vivado/2020.3/bin:$PATH
vivado -mode batch -nojournal -nolog -source soc_build.tcl 2>&1 | tee soc_build.log
ls -la diagram/ 2>/dev/null || true
