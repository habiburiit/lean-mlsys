# ============================================================================
#  soc_build_zynq.tcl  -  ZCU102 version: Zynq UltraScale+ PS (ARM Cortex-A53 + DDR4) as the host.  Packages lean_axi_ip (accelerator + LEAN loader + ECC behind AXI4-Lite) as an IP and build a block design:
#  MicroBlaze + local memory + debug, AXI interconnect, AXI UART, AXI BRAM weight buffer, AXI interrupt controller, the LEAN IP.
#  Exports the block-design diagram (PDF + SVG) in batch mode, no GUI needed.   Run from hw/fpga_soc:  bash run_soc.sh
#  SYNTH=1 in the environment also synthesises and implements the whole system (long).
# ============================================================================
set HERE [file normalize [file dirname [info script]]]
set RTL  [expr {[info exists ::env(ACCEL_RTL)] ? $::env(ACCEL_RTL) : "./"}]
set FPGA [file normalize $HERE/../fpga]
set PREF {xczu9eg- xczu7ev- xcvu9p- xc7z045 xc7k325t xc7vx485t xc7z020 xc7a200t xc7a100t}
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

# ---------------------------------------------------------------- 2. block design around the Zynq UltraScale+ PS (ARM Cortex-A53, DDR4)
file delete -force $HERE/soc_proj_zynq
create_project soc_proj_zynq $HERE/soc_proj_zynq -part $PART -force
set_property ip_repo_paths [list $HERE/ip_repo] [current_project]
update_ip_catalog
create_bd_design lean_soc_zynq
# processing system: A53 cluster + DDR4 controller; one AXI master into the fabric at 100 MHz, one interrupt from the fabric
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__IRQ0 {1} CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} CONFIG.PSU__USE__FABRIC__RST {1} \
    CONFIG.PSU__DDRC__ENABLE {1} CONFIG.PSU__UART0__PERIPHERAL__ENABLE {1}] [get_bd_cells zynq_ultra_ps_e_0]
catch { apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "0"} [get_bd_cells zynq_ultra_ps_e_0] }
# peripherals in the fabric
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_uartlite axi_uartlite_0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl weight_buffer_ctrl
create_bd_cell -type ip -vlnv anon.org:lean:lean_axi_ip lean_accel_0
set_property -dict [list CONFIG.MEM_DIR "$RTL/mem_clean/" CONFIG.MAP_HEX "$FPGA/layer_map.hex" CONFIG.IMG_HEX "$FPGA/layer_image.hex" \
    CONFIG.BLOCK $BLOCK CONFIG.PAR $PAR CONFIG.NBLOCKS $NBLOCKS CONFIG.NLINES $NLINES \
    CONFIG.W0 [lindex $TIERS 0] CONFIG.W1 [lindex $TIERS 1] CONFIG.W2 [lindex $TIERS 2] CONFIG.W3 [lindex $TIERS 3] \
    CONFIG.R0 [lindex $RAT 0] CONFIG.R1 [lindex $RAT 1] CONFIG.R2 [lindex $RAT 2] CONFIG.R3 [lindex $RAT 3]] [get_bd_cells lean_accel_0]
foreach s {axi_uartlite_0/S_AXI weight_buffer_ctrl/S_AXI lean_accel_0/s_axi} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave "/$s" ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}] [get_bd_intf_pins $s]
}
apply_bd_automation -rule xilinx.com:bd_rule:bram_cntlr -config {BRAM "Auto"} [get_bd_intf_pins weight_buffer_ctrl/BRAM_PORTA]
catch { set_property name weight_buffer_bram [get_bd_cells [lindex [get_bd_cells -filter {VLNV =~ "*blk_mem_gen*" || VLNV =~ "*emb_mem_gen*"}] 0]] }
make_bd_intf_pins_external [get_bd_intf_pins axi_uartlite_0/UART]
# interrupts: UART and accelerator done into the PS
catch {
    create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat xlconcat_irq
    set_property CONFIG.NUM_PORTS 2 [get_bd_cells xlconcat_irq]
    connect_bd_net [get_bd_pins axi_uartlite_0/interrupt] [get_bd_pins xlconcat_irq/In0]
    connect_bd_net [get_bd_pins lean_accel_0/irq_done] [get_bd_pins xlconcat_irq/In1]
    connect_bd_net [get_bd_pins xlconcat_irq/dout] [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]
}
assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design
file mkdir $HERE/diagram
catch { write_bd_layout -force -format pdf -orientation landscape $HERE/diagram/lean_soc_zynq_block_design.pdf }
write_bd_tcl -force $HERE/diagram/lean_soc_zynq_bd.tcl
set fa [open $HERE/diagram/lean_soc_zynq_address_map.txt w]; foreach seg [get_bd_addr_segs] { puts $fa "[get_property PATH $seg]  offset [get_property OFFSET $seg]  range [get_property RANGE $seg]" }; close $fa
puts "BLOCK DESIGN DONE (Zynq PS): project fpga_soc/soc_proj_zynq, address map in diagram/lean_soc_zynq_address_map.txt"
