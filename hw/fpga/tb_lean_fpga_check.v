`timescale 1ns/1ps
// tb_lean_fpga_check.v  -  checks the (routed) lean_fpga_top netlist word-for-word against the packer's INT8 words (layer_int8.hex,
// 16 words per 128-bit line, MSB-first) and expects zero ECC errors.  Parameters NLINES_INT8 / NWORDS passed with -generic_top.
module tb_lean_fpga_check #(parameter NLINES_INT8 = 1, parameter NWORDS = 16) ();
  reg clk = 0, rst_n = 0, start = 0; wire done; wire [31:0] checksum, words, ecc;
  lean_fpga_top dut (.clk(clk), .rst_n(rst_n), .start(start), .done(done), .checksum(checksum), .words(words), .ecc_err_cnt(ecc));
  always #5 clk = ~clk;
  reg [127:0] exp [0:NLINES_INT8-1]; integer wi = 0, errs = 0, k; reg signed [7:0] got, want;
  initial $readmemh("layer_int8.hex", exp);
  // the netlist keeps the loader's out_valid/out_data registers; probe them through the hierarchy
  always @(posedge clk) if (rst_n && dut.u_loader.out_valid) begin
    for (k = 0; k < 8; k = k + 1) begin
      got = dut.u_loader.out_data[(7-k)*8 +: 8]; want = exp[(wi+k)/16][127 - ((wi+k)%16)*8 -: 8];
      if (wi + k < NWORDS && got !== want) begin errs = errs + 1; if (errs <= 5) $display("MISMATCH word %0d got %0d want %0d", wi+k, got, want); end
    end
    wi = wi + 8;
  end
  initial begin
    #105 rst_n = 1; #50 @(posedge clk) start = 1; @(posedge clk) start = 0;
    wait (done); #50;
    $display("XSIM RESULT: words=%0d expected=%0d mismatches=%0d ecc_errors=%0d checksum=%08x  -> %s", words, NWORDS, errs, ecc,
             checksum, (errs == 0 && words == NWORDS && ecc == 0) ? "PASS" : "FAIL");
    $finish;
  end
  initial begin #200000000 $display("XSIM TIMEOUT words=%0d", words); $finish; end
endmodule
