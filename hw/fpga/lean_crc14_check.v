`timescale 1ns/1ps
// lean_crc14_check.v  -  synthesisable parity checker for the FPGA build: recomputes the CRC-14 (poly 0x0443) of every 16-word line
// of a HIGH (8-bit, ECC-protected) block from the unpacked words and compares it with the stored parity forwarded on par_data.
// Same function as the checker model in tb/tb_lean_loader.v.  (Placeholder code; the real ECC checker has the same interface.)
module lean_crc14_check #(parameter BLOCK = 32, parameter LINE_W = 16, parameter PAR = 14, parameter OUTW = 8) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          out_valid,
    input  wire [OUTW*8-1:0]             out_data,
    input  wire                          out_high,
    input  wire                          par_valid,
    input  wire [PAR*(BLOCK/LINE_W)-1:0] par_data,
    output reg                           err,          // one pulse per block whose stored parity differs from the recomputed one
    output reg  [31:0]                   err_cnt
);
    localparam NL = BLOCK / LINE_W;                   // parity lines per HIGH block
    localparam CPL = LINE_W / OUTW;                   // output chunks per line (2)
    function [PAR-1:0] crc_step(input [PAR-1:0] c, input [OUTW*8-1:0] d); integer q; reg [PAR-1:0] rg; reg top; begin
        rg = c; for (q = OUTW*8-1; q >= 0; q = q - 1) begin top = rg[PAR-1]; rg = {rg[PAR-2:0], d[q]}; if (top) rg = rg ^ 14'h0443; end
        crc_step = rg; end endfunction
    reg [PAR-1:0] crc; reg [$clog2(CPL+1)-1:0] chunk; reg [$clog2(NL+1)-1:0] line; reg [PAR*NL-1:0] recomputed;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin crc <= 0; chunk <= 0; line <= 0; recomputed <= 0; err <= 1'b0; err_cnt <= 0; end
        else begin
            err <= 1'b0;
            if (out_valid && out_high) begin
                if (chunk == CPL - 1) begin
                    recomputed[PAR*NL-1 - line*PAR -: PAR] <= crc_step(crc, out_data); crc <= 0; chunk <= 0;
                    line <= (line == NL - 1) ? 0 : line + 1;
                end else begin crc <= crc_step(crc, out_data); chunk <= chunk + 1; end
            end
            if (par_valid && par_data != recomputed) begin err <= 1'b1; err_cnt <= err_cnt + 1; end
        end
    end
endmodule
