# export the Zynq PS block-design diagram; needs the GUI (ssh -X):  vivado -mode tcl -source fpga_soc/export_diagram_zynq.tcl  (from hw/)
set HERE [file normalize [file dirname [info script]]]
open_project $HERE/soc_proj_zynq/soc_proj_zynq.xpr
open_bd_design [get_files lean_soc_zynq.bd]
start_gui
regenerate_bd_layout
file mkdir $HERE/diagram
write_bd_layout -force -format pdf -orientation landscape $HERE/diagram/lean_soc_zynq_block_design.pdf
write_bd_layout -force -format svg -orientation landscape $HERE/diagram/lean_soc_zynq_block_design.svg
puts "DIAGRAM WRITTEN: fpga_soc/diagram/lean_soc_zynq_block_design.pdf (close the window to exit)"
