# xsim_lean.tcl  -  implement lean_fpga_top (LEAN loader + ECC, current layer in fpga/layer.tcl) and write the ROUTED netlist for simulation.
set PREF {xczu7ev- xczu9eg- xcvu9p- xc7z045 xc7k325t xc7vx485t xc7z020 xc7a200t xc7a100t}
set PART ""
foreach pfx $PREF { set m [get_parts -quiet ${pfx}*]; if {[llength $m]} { set PART [lindex $m 0]; break } }
if {$PART == ""} { set PART [lindex [get_parts] end] }
source layer.tcl
file mkdir xsim
create_project -in_memory -part $PART
read_verilog ../rtl/lean_unpack.v
read_verilog ../rtl/lean_weight_loader.v
read_verilog lean_crc14_check.v
read_verilog lean_fpga_top.v
synth_design -top lean_fpga_top -part $PART -flatten_hierarchy none -generic BLOCK=$BLOCK -generic PAR=$PAR -generic NBLOCKS=$NBLOCKS -generic NLINES=$NLINES \
    -generic CHECK=1 -generic W0=[lindex $TIERS 0] -generic W1=[lindex $TIERS 1] -generic W2=[lindex $TIERS 2] -generic W3=[lindex $TIERS 3] \
    -generic R0=[lindex $RAT 0] -generic R1=[lindex $RAT 1] -generic R2=[lindex $RAT 2] -generic R3=[lindex $RAT 3]
create_clock -period 10.0 -name clk [get_ports clk]
opt_design
place_design
route_design
report_timing_summary -file xsim/impl_timing.rpt
write_verilog -force -mode funcsim xsim/lean_fpga_top_routed.v
puts "ROUTED NETLIST WRITTEN: xsim/lean_fpga_top_routed.v  (part $PART)"
