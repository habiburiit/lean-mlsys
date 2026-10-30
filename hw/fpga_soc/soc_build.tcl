# ============================================================================
#  soc_build.tcl  -  package lean_axi_ip (accelerator + LEAN loader + ECC behind AXI4-Lite) as an IP and build a block design:
#  MicroBlaze + local memory + debug, AXI interconnect, AXI UART, AXI BRAM weight buffer, AXI interrupt controller, the LEAN IP.
#  Exports the block-design diagram (PDF + SVG) in batch mode, no GUI needed.   Run from hw/fpga_soc:  bash run_soc.sh
#  SYNTH=1 in the environment also synthesises and implements the whole system (long).
# ============================================================================
set HERE [file normalize [file dirname [info script]]]
set RTL  [expr {[info exists ::env(ACCEL_RTL)] ? $::env(ACCEL_RTL) : "./"}]
set FPGA [file normalize $HERE/../fpga]
set PREF {xczu7ev- xczu9eg- xcvu9p- xc7z045 xc7k325t xc7vx485t xc7z020 xc7a200t xc7a100t}
set PART ""
foreach pfx $PREF { set m [get_parts -quiet ${pfx}*]; if {[llength $m]} { set PART [lindex $m 0]; break } }
if {$PART == ""} { set PART [lindex [get_parts] end] }
puts "SOC PART: $PART"
source $FPGA/layer.tcl

# ---------------------------------------------------------------- 1. package the IP
set IPDIR $HERE/ip_repo/lean_axi_ip
file delete -force $IPDIR $HERE/pkg_proj
create_project pkg_proj $HERE/pkg_proj -part $PART -force
set srcs {}
foreach f {act_sram controller conv1_engine conv_engine fxp_mac gap im2col_addr_gen maxpool2x2 pe pe_array relu_requant weight_mem top_systolic_clean} { lappend srcs $RTL/$f.v }
lappend srcs $RTL/params.vh $HERE/../rtl/lean_unpack.v $HERE/../rtl/lean_weight_loader.v $FPGA/lean_crc14_check.v $FPGA/lean_fpga_top.v $HERE/lean_accel_top_soc.v $HERE/lean_axi_ip.v
add_files -norecurse $srcs
set_property include_dirs [list $RTL] [current_fileset]
set_property is_global_include true [get_files $RTL/params.vh]
set_property top lean_axi_ip [current_fileset]
update_compile_order -fileset sources_1
ipx::package_project -root_dir $IPDIR -vendor anon.org -library lean -taxonomy /UserIP -import_files -set_current true -force
set core [ipx::current_core]
set_property name lean_axi_ip $core
set_property display_name "LEAN accelerator (systolic array + LEAN weight loader + ECC)" $core
set_property description "Systolic accelerator with the LEAN packed-weight loader and parity checker behind AXI4-Lite" $core
set_property version 1.0 $core
ipx::infer_bus_interfaces xilinx.com:interface:aximm_rtl:1.0 $core
catch { ipx::associate_bus_interfaces -busif s_axi -clock s_axi_aclk $core }
catch { ipx::add_bus_interface irq_done $core; set_property abstraction_type_vlnv xilinx.com:signal:interrupt_rtl:1.0 [ipx::get_bus_interfaces irq_done -of_objects $core]
        set_property bus_type_vlnv xilinx.com:signal:interrupt:1.0 [ipx::get_bus_interfaces irq_done -of_objects $core]
        ipx::add_port_map INTERRUPT [ipx::get_bus_interfaces irq_done -of_objects $core]; set_property physical_name irq_done [ipx::get_port_maps INTERRUPT -of_objects [ipx::get_bus_interfaces irq_done -of_objects $core]]
        set_property value ACTIVE_HIGH [ipx::add_bus_parameter SENSITIVITY [ipx::get_bus_interfaces irq_done -of_objects $core]] }
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
close_project
puts "IP PACKAGED: $IPDIR"

# ---------------------------------------------------------------- 2. block design
file delete -force $HERE/soc_proj
create_project soc_proj $HERE/soc_proj -part $PART -force
set_property ip_repo_paths [list $HERE/ip_repo] [current_project]
update_ip_catalog
create_bd_design lean_soc
# processor with local memory, debug module, interrupt controller and a 100 MHz external clock
create_bd_cell -type ip -vlnv xilinx.com:ip:microblaze microblaze_0
apply_bd_automation -rule xilinx.com:bd_rule:microblaze -config {local_mem "64KB" ecc "None" cache "None" debug_module "Debug Only" axi_periph "Enabled" axi_intc "1" clk "New External Port (100 MHz)"} [get_bd_cells microblaze_0]
# peripherals
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_uartlite axi_uartlite_0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl axi_bram_ctrl_0
set_property name weight_buffer_ctrl [get_bd_cells axi_bram_ctrl_0]
create_bd_cell -type ip -vlnv anon.org:lean:lean_axi_ip lean_accel_0
set_property -dict [list CONFIG.MEM_DIR "$RTL/mem_clean/" CONFIG.MAP_HEX "$FPGA/layer_map.hex" CONFIG.IMG_HEX "$FPGA/layer_image.hex" \
    CONFIG.BLOCK $BLOCK CONFIG.PAR $PAR CONFIG.NBLOCKS $NBLOCKS CONFIG.NLINES $NLINES \
    CONFIG.W0 [lindex $TIERS 0] CONFIG.W1 [lindex $TIERS 1] CONFIG.W2 [lindex $TIERS 2] CONFIG.W3 [lindex $TIERS 3] \
    CONFIG.R0 [lindex $RAT 0] CONFIG.R1 [lindex $RAT 1] CONFIG.R2 [lindex $RAT 2] CONFIG.R3 [lindex $RAT 3]] [get_bd_cells lean_accel_0]
foreach s {axi_uartlite_0/S_AXI weight_buffer_ctrl/S_AXI lean_accel_0/s_axi} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/microblaze_0 (Periph)} Slave "/$s" ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}] [get_bd_intf_pins $s]
}
apply_bd_automation -rule xilinx.com:bd_rule:bram_cntlr -config {BRAM "Auto"} [get_bd_intf_pins weight_buffer_ctrl/BRAM_PORTA]
set_property name weight_buffer_bram [get_bd_cells [lindex [get_bd_cells -filter {VLNV =~ "*blk_mem_gen*" || VLNV =~ "*emb_mem_gen*"}] 0]]
make_bd_intf_pins_external [get_bd_intf_pins axi_uartlite_0/UART]
# interrupts: UART and accelerator done into the interrupt controller
catch {
    create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat xlconcat_irq
    set_property CONFIG.NUM_PORTS 2 [get_bd_cells xlconcat_irq]
    connect_bd_net [get_bd_pins axi_uartlite_0/interrupt] [get_bd_pins xlconcat_irq/In0]
    connect_bd_net [get_bd_pins lean_accel_0/irq_done] [get_bd_pins xlconcat_irq/In1]
    connect_bd_net [get_bd_pins xlconcat_irq/dout] [get_bd_pins microblaze_0_axi_intc/intr]
}
assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design
file mkdir $HERE/diagram
write_bd_layout -force -format pdf -orientation landscape $HERE/diagram/lean_soc_block_design.pdf
write_bd_layout -force -format svg -orientation landscape $HERE/diagram/lean_soc_block_design.svg
write_bd_tcl -force $HERE/diagram/lean_soc_bd.tcl
report_bd_address -file $HERE/diagram/lean_soc_address_map.txt
puts "BLOCK DESIGN DONE: diagram in fpga_soc/diagram/ (pdf, svg), address map in lean_soc_address_map.txt"

# ---------------------------------------------------------------- 3. optional: implement the whole system
if {[info exists ::env(SYNTH)] && $::env(SYNTH) == "1"} {
    make_wrapper -files [get_files lean_soc.bd] -top
    add_files -norecurse [glob $HERE/soc_proj/soc_proj.gen/sources_1/bd/lean_soc/hdl/lean_soc_wrapper.v $HERE/soc_proj/soc_proj.srcs/sources_1/bd/lean_soc/hdl/lean_soc_wrapper.v]
    set_property top lean_soc_wrapper [current_fileset]
    launch_runs impl_1 -jobs 8
    wait_on_run impl_1
    open_run impl_1
    file mkdir $HERE/reports
    report_utilization -file $HERE/reports/soc_utilization.rpt
    report_utilization -hierarchical -file $HERE/reports/soc_utilization_hier.rpt
    report_timing_summary -file $HERE/reports/soc_timing.rpt
    report_power -file $HERE/reports/soc_power.rpt
    puts "SOC IMPLEMENTED: reports in fpga_soc/reports/"
}
