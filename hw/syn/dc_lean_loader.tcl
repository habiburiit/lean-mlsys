# ============================================================================
#  dc_lean_loader.tcl  -  Synopsys Design Compiler script for the LEAN weight loader (tier-map controller + unpacker).
#  Standalone block synthesis (minutes), NOT the full-top array run.  Library settings are taken from the the selective code flow:
#  lean_hw_agent.py fills LIB_DIR / TARGET_LIB / LINK_LIB from the the selective code DC script it finds, or edit the three lines below.
#     dc_shell -f syn/dc_lean_loader.tcl | tee syn/dc_lean_loader.log
# ============================================================================
set LIB_DIR     "@LIB_DIR@"
set TARGET_LIB  "@TARGET_LIB@"
set LINK_LIB    "@LINK_LIB@"
set CLK_PERIOD  10.0                 ;# ns, 100 MHz as in the the selective code runs
set TOP         lean_weight_loader
set RPT         "syn/reports"
# tier set and block size must match the packer run (format.json)
set PARAMS      "BLOCK=64,LINE_W=16,PAR=14,OUTW=8,W0=8,W1=4,W2=2,W3=2"

set search_path [concat $search_path [list . rtl syn $LIB_DIR]]
set target_library [list $TARGET_LIB]
set link_library   [list "*" $LINK_LIB]
file mkdir $RPT

file mkdir syn/work
define_design_lib WORK -path syn/work
analyze -format verilog {rtl/lean_unpack.v rtl/lean_weight_loader.v}
elaborate $TOP -parameters $PARAMS
set TOP [get_object_name [current_design]]
link
uniquify

create_clock -name clk -period $CLK_PERIOD [get_ports clk]
set_input_delay  [expr 0.2 * $CLK_PERIOD] -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay [expr 0.2 * $CLK_PERIOD] -clock clk [all_outputs]
set_max_fanout 16 $TOP
set_ideal_network [get_ports rst_n]

compile_ultra
report_area            > $RPT/${TOP}_area.rpt
report_power           > $RPT/${TOP}_power.rpt
report_timing -max_paths 3 > $RPT/${TOP}_timing.rpt
report_qor             > $RPT/${TOP}_qor.rpt
report_hierarchy       > $RPT/${TOP}_hier.rpt
write -format verilog -hierarchy -output $RPT/${TOP}_netlist.v
write_sdc $RPT/${TOP}.sdc
puts "DC DONE: reports in $RPT"
exit
