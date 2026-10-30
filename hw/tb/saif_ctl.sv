// saif_ctl.sv  -  second top-level module compiled next to tb_lean_loader.v: records the switching activity of the loader
// (tb_lean_loader.dut) for the whole run and writes logs/loader.saif at the end.  tb_lean_loader.v itself is not modified.
`timescale 1ns/1ps
module saif_ctl;
  initial begin
    $set_gate_level_monitoring("rtl_on");
    $set_toggle_region(tb_lean_loader.dut);
    $toggle_start;
  end
  final begin
    $toggle_stop;
    $toggle_report("logs/loader.saif", 1.0e-9, "tb_lean_loader.dut");
    $display("SAIF written: logs/loader.saif");
  end
endmodule
