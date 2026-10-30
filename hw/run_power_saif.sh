#!/bin/bash
# run from hw/ on the server:   bash run_power_saif.sh <tag>
#   <tag> = the tag of the paper's loader run (the one with syn/dc_lean_loader_<tag>.tcl and vec_<tag>/ present; ls syn/ vec_* to see it)
# Simulation-driven power of the LEAN loader:
#   1. VCS: the SAME clean run as lean_hw_agent.py's sim step (same vectors, same parameters), plus tb/saif_ctl.sv which
#      records the switching activity of tb_lean_loader.dut into logs/loader.saif.  tb_lean_loader.v is not modified.
#   2. Design Compiler: the SAME synthesis script as the paper run, with read_saif before report_power, so the power
#      report uses the measured toggle rates instead of the default ones.  Reports in syn/reports_saif/.
# Log: logs/power_saif_<tag>.log      watch: tail -f logs/power_saif_<tag>.log
set -e
cd "$(dirname "$0")"
TAG=${1:?tag}; VEC=vec_$TAG; TCL=syn/dc_lean_loader_$TAG.tcl
[ -d $VEC ] || { echo "missing $VEC"; exit 1; }; [ -f $TCL ] || { echo "missing $TCL"; exit 1; }
mkdir -p logs syn/reports_saif
LOG=logs/power_saif_$TAG.log; : > $LOG
source /apps/settings >/dev/null 2>&1 || true
# parameters exactly as in the filled DC script (BLOCK=..,LINE_W=..,...)
P=$(grep -o 'set PARAMS *"[^"]*"' $TCL | sed 's/.*"\(.*\)"/\1/' | tr ',' ' ')
PDEF=$(for p in $P; do printf -- "-pvalue+tb_lean_loader.%s " "$p"; done)
{
echo "=== SAIF power run  tag=$TAG  params: $P  $(date) ==="
echo "--- VCS build ---"
vcs -full64 -sverilog +v2k $PDEF rtl/lean_unpack.v rtl/lean_weight_loader.v tb/tb_lean_loader.v tb/saif_ctl.sv -o logs/simv_saif_$TAG
echo "--- VCS clean run with activity recording ---"
./logs/simv_saif_$TAG +DIR=$VEC
ls -la logs/loader.saif
echo "--- DC with read_saif  $(date) ---"
# same script, reports to reports_saif, read_saif inserted before report_power
sed -e 's#set RPT *"syn/reports"#set RPT         "syn/reports_saif"#' \
    -e 's#^report_power .*#read_saif -input logs/loader.saif -instance tb_lean_loader/dut -verbose > $RPT/${TOP}_saif_read.rpt\nreport_power -analysis_effort high > $RPT/${TOP}_power_saif.rpt\nreport_power -hierarchy > $RPT/${TOP}_power_saif_hier.rpt#' \
    $TCL > syn/dc_lean_loader_${TAG}_saif.tcl
dc_shell -f syn/dc_lean_loader_${TAG}_saif.tcl
echo "--- RESULT ---"
grep -i "Total Dynamic Power\|Cell Leakage Power\|Net Switching Power\|Cell Internal Power" syn/reports_saif/*_power_saif.rpt
grep -i "annotated\|user annotated\|percent" syn/reports_saif/*_saif_read.rpt | head -5
echo "DONE power_saif $TAG $(date)"
} >> $LOG 2>&1
grep -A3 "RESULT" $LOG
