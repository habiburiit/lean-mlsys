`timescale 1ns/1ps
// lean_accel_top.v  -  the full systolic accelerator (top_systolic_clean from STDM_RTL: PE array, weight memories, activation SRAM,
// controller, requant, pooling, GAP) next to the LEAN weight loader on the same chip.  WITH_LEAN=0 gives the accelerator alone,
// WITH_LEAN=1 adds lean_fpga_top (loader + tier map + one packed layer + CRC-14 checker).  The difference is the LEAN overhead
// relative to the whole accelerator.  Outputs of both are kept live through the ports.
module lean_accel_top #(
    parameter WITH_LEAN = 1,
    parameter MEM_DIR = "mem_clean/",
    parameter BLOCK = 32, parameter PAR = 14, parameter NBLOCKS = 1024, parameter NLINES = 1024,
    parameter W0 = 8, parameter W1 = 6, parameter W2 = 4, parameter W3 = 3,
    parameter [15:0] R0 = 16'h0100, parameter [15:0] R1 = 16'h0400, parameter [15:0] R2 = 16'h1000, parameter [15:0] R3 = 16'h2000
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] img_idx,
    input  wire        res_re,
    input  wire [14:0] res_addr,
    output wire        done,
    output wire signed [7:0] res_dout,
    output wire [1:0]  res_bank,
    output wire        lean_done,
    output wire [31:0] lean_checksum,
    output wire [31:0] lean_words,
    output wire [31:0] lean_ecc_err
);
    top_systolic_clean #(.MEM_DIR(MEM_DIR)) u_accel (.clk(clk), .rst_n(rst_n), .start(start), .done(done), .img_idx(img_idx),
        .res_re(res_re), .res_addr(res_addr), .res_dout(res_dout), .res_bank(res_bank));
    generate if (WITH_LEAN) begin : g_lean
        lean_fpga_top #(.BLOCK(BLOCK), .PAR(PAR), .NBLOCKS(NBLOCKS), .NLINES(NLINES), .W0(W0), .W1(W1), .W2(W2), .W3(W3),
                        .R0(R0), .R1(R1), .R2(R2), .R3(R3), .CHECK(1)) u_lean (.clk(clk), .rst_n(rst_n), .start(start),
            .done(lean_done), .checksum(lean_checksum), .words(lean_words), .ecc_err_cnt(lean_ecc_err));
    end else begin : g_nolean
        assign lean_done = 1'b0; assign lean_checksum = 32'd0; assign lean_words = 32'd0; assign lean_ecc_err = 32'd0;
    end endgenerate
endmodule
