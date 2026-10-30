# ============================================================================
#  vivado_lean.tcl  -  batch synthesis + implementation of the LEAN weight loader with BRAM memories (lean_fpga_top), ZCU104 part.
#  Run from the hw/fpga folder after lean_fpga_prep.py:    vivado -mode batch -source vivado_lean.tcl | tee vivado_lean.log
#  Reports in fpga/reports/: utilization (LUT/FF/BRAM/DSP), timing summary (100 MHz), power.  No board needed.
# ============================================================================
# part: first installed device from this preference list (ZCU104, ZCU102, VCU118, ZC706, KC705, VC707, then anything else)
set PREF {xczu7ev- xczu9eg- xcvu9p- xc7z045 xc7k325t xc7vx485t xc7z020 xc7a200t xc7a100t}
set PART ""
foreach pfx $PREF { set m [get_parts -quiet ${pfx}*]; if {[llength $m]} { set PART [lindex $m 0]; break } }
if {$PART == ""} { set PART [lindex [get_parts] end] }
puts "VIVADO PART: $PART  (installed parts: [llength [get_parts]])"
set CLK_NS 10.0                        ;# 100 MHz, same as the ASIC run
# configuration: int8 (baseline loader, plain INT8 BRAM), lean (LEAN loader), lean_ecc (LEAN loader + CRC-14 parity checker)
set CFG [expr {[info exists ::env(CFG)] ? $::env(CFG) : "lean"}]
source layer.tcl                      ;# NBLOCKS NLINES BLOCK PAR TIERS RAT, written by lean_fpga_prep.py
set R reports_$CFG
file mkdir $R
create_project -in_memory -part $PART
if {$CFG == "int8"} {
    read_verilog lean_fpga_int8_top.v
    synth_design -top lean_fpga_int8_top -part $PART -generic NLINES=$NLINES_INT8
} else {
    read_verilog ../rtl/lean_unpack.v
    read_verilog ../rtl/lean_weight_loader.v
    read_verilog lean_crc14_check.v
    read_verilog lean_fpga_top.v
    synth_design -top lean_fpga_top -part $PART -generic BLOCK=$BLOCK -generic PAR=$PAR -generic NBLOCKS=$NBLOCKS -generic NLINES=$NLINES \
        -generic CHECK=[expr {$CFG == "lean_ecc"}] \
        -generic W0=[lindex $TIERS 0] -generic W1=[lindex $TIERS 1] -generic W2=[lindex $TIERS 2] -generic W3=[lindex $TIERS 3] \
        -generic R0=[lindex $RAT 0] -generic R1=[lindex $RAT 1] -generic R2=[lindex $RAT 2] -generic R3=[lindex $RAT 3]
}
create_clock -period $CLK_NS -name clk [get_ports clk]
set_input_delay  -clock clk 2.0 [get_ports {rst_n start}]
set_output_delay -clock clk 2.0 [all_outputs]
report_utilization -file $R/synth_utilization.rpt
opt_design
place_design
route_design
report_utilization   -file $R/impl_utilization.rpt
report_utilization   -hierarchical -file $R/impl_utilization_hier.rpt
report_timing_summary -file $R/impl_timing.rpt
report_power         -file $R/impl_power.rpt
write_checkpoint -force $R/routed.dcp
puts "VIVADO DONE: config $CFG, part $PART, reports in fpga/$R"
