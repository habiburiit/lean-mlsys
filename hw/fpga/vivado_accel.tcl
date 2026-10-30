# vivado_accel.tcl  -  accelerator alone (CFG=accel) or accelerator + LEAN loader + ECC (CFG=accel_lean) on the same part as vivado_lean.tcl.
# Run from hw/fpga:   CFG=accel vivado -mode batch -source vivado_accel.tcl       (run_fpga_accel.sh does both)
set PREF {xczu7ev- xczu9eg- xcvu9p- xc7z045 xc7k325t xc7vx485t xc7z020 xc7a200t xc7a100t}
set PART ""
foreach pfx $PREF { set m [get_parts -quiet ${pfx}*]; if {[llength $m]} { set PART [lindex $m 0]; break } }
if {$PART == ""} { set PART [lindex [get_parts] end] }
set CLK_NS 10.0
set CFG [expr {[info exists ::env(CFG)] ? $::env(CFG) : "accel_lean"}]
set RTL [expr {[info exists ::env(ACCEL_RTL)] ? $::env(ACCEL_RTL) : "./"}]
source layer.tcl
set R reports_$CFG
file mkdir $R
puts "VIVADO PART: $PART   config $CFG   accelerator RTL $RTL"
create_project -in_memory -part $PART
foreach f {act_sram controller conv1_engine conv_engine fxp_mac gap im2col_addr_gen maxpool2x2 pe pe_array relu_requant weight_mem top_systolic_clean} {
    read_verilog $RTL/$f.v
}
read_verilog ../rtl/lean_unpack.v
read_verilog ../rtl/lean_weight_loader.v
read_verilog lean_crc14_check.v
read_verilog lean_fpga_top.v
read_verilog lean_accel_top.v
synth_design -top lean_accel_top -part $PART -include_dirs $RTL -generic WITH_LEAN=[expr {$CFG == "accel_lean"}] -generic MEM_DIR=\"$RTL/mem_clean/\" \
    -generic BLOCK=$BLOCK -generic PAR=$PAR -generic NBLOCKS=$NBLOCKS -generic NLINES=$NLINES \
    -generic W0=[lindex $TIERS 0] -generic W1=[lindex $TIERS 1] -generic W2=[lindex $TIERS 2] -generic W3=[lindex $TIERS 3] \
    -generic R0=[lindex $RAT 0] -generic R1=[lindex $RAT 1] -generic R2=[lindex $RAT 2] -generic R3=[lindex $RAT 3]
create_clock -period $CLK_NS -name clk [get_ports clk]
set_input_delay  -clock clk 2.0 [all_inputs]
set_output_delay -clock clk 2.0 [all_outputs]
report_utilization -file $R/synth_utilization.rpt
opt_design
place_design
route_design
report_utilization   -file $R/impl_utilization.rpt
report_utilization   -hierarchical -file $R/impl_utilization_hier.rpt
report_timing_summary -file $R/impl_timing.rpt
report_power         -file $R/impl_power.rpt
puts "VIVADO DONE: config $CFG, part $PART, reports in fpga/$R"
