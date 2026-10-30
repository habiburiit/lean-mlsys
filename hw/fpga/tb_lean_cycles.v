`timescale 1ns/1ps
// tb_lean_cycles.v  -  cycle-accurate THROUGHPUT and LATENCY of the LEAN loader (lean_fpga_top, RTL) against the plain INT8
// read-out (lean_fpga_int8_top) on the same layer (fpga/layer_*.hex).  Nothing in the design is changed; this bench only counts.
//   latency     = clock cycles from the start pulse to the first 8-word output beat
//   total       = clock cycles from the start pulse to done
//   active      = cycles with out_valid high  (words / active = words per active cycle)
//   throughput  = words / total  (words per cycle, incl. start-up and any stall), and at 100 MHz
// Parameters come from fpga/layer.json via run_cycles.sh (-generic_top).
module tb_lean_cycles #(parameter BLOCK = 32, parameter PAR = 14, parameter NBLOCKS = 1, parameter NLINES = 1, parameter NLINES_INT8 = 1,
                        parameter W0 = 8, parameter W1 = 6, parameter W2 = 4, parameter W3 = 3,
                        parameter R0 = 256, parameter R1 = 1024, parameter R2 = 4096, parameter R3 = 8192, parameter CHECK = 1) ();
  reg clk = 0, rst_n = 0; always #5 clk = ~clk;
  // ---- LEAN loader (with the parity checker, as in the lean_ecc build) ----
  reg  start_l = 0; wire done_l; wire [31:0] cs_l, words_l, ecc_l;
  lean_fpga_top #(.BLOCK(BLOCK), .PAR(PAR), .NBLOCKS(NBLOCKS), .NLINES(NLINES), .CHECK(CHECK),
                  .W0(W0), .W1(W1), .W2(W2), .W3(W3), .R0(R0[15:0]), .R1(R1[15:0]), .R2(R2[15:0]), .R3(R3[15:0]))
      dut_l (.clk(clk), .rst_n(rst_n), .start(start_l), .done(done_l), .checksum(cs_l), .words(words_l), .ecc_err_cnt(ecc_l));
  // ---- plain INT8 read-out, same layer, same 8 words per cycle ----
  reg  start_i = 0; wire done_i; wire [31:0] cs_i, words_i;
  lean_fpga_int8_top #(.NLINES(NLINES_INT8)) dut_i (.clk(clk), .rst_n(rst_n), .start(start_i), .done(done_i), .checksum(cs_i), .words(words_i));

  integer cyc = 0, t0_l = -1, first_l = -1, end_l = -1, act_l = 0, t0_i = -1, first_i = -1, end_i = -1, act_i = 0;
  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (rst_n && dut_l.out_valid)    begin act_l = act_l + 1; if (first_l < 0) first_l = cyc; end
    if (rst_n && dut_i.out_valid)    begin act_i = act_i + 1; if (first_i < 0) first_i = cyc; end
  end
  initial begin
    #105 rst_n = 1; #50;
    @(posedge clk) start_l = 1; t0_l = cyc; @(posedge clk) start_l = 0;
    wait (done_l); end_l = cyc; #50;
    @(posedge clk) start_i = 1; t0_i = cyc; @(posedge clk) start_i = 0;
    wait (done_i); end_i = cyc; #50;
    $display("CYCLES LEAN : words=%0d latency=%0d total=%0d active=%0d words/cycle=%.3f words/active=%.3f at100MHz=%.3f us ecc_err=%0d checksum=%08x",
             words_l, first_l - t0_l, end_l - t0_l, act_l, 1.0*words_l/(end_l - t0_l), 1.0*words_l/act_l, (end_l - t0_l)/100.0, ecc_l, cs_l);
    $display("CYCLES INT8 : words=%0d latency=%0d total=%0d active=%0d words/cycle=%.3f words/active=%.3f at100MHz=%.3f us checksum=%08x",
             words_i, first_i - t0_i, end_i - t0_i, act_i, 1.0*words_i/(end_i - t0_i), 1.0*words_i/act_i, (end_i - t0_i)/100.0, cs_i);
    $finish;
  end
  initial begin #400000000 $display("CYCLES TIMEOUT words_l=%0d words_i=%0d", words_l, words_i); $finish; end
endmodule
