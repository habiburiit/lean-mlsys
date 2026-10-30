`timescale 1ns/1ps
// lean_fpga_int8_top.v  -  BASELINE for the FPGA comparison: the same layer stored as plain INT8 in a 128-bit BRAM (16 words per line),
// read out at the same rate as the LEAN loader (8 INT8 words per cycle), same checksum.  No tier map, no unpacker, no parity.
module lean_fpga_int8_top #(parameter OUTW = 8, parameter NLINES = 1024, parameter IMG_HEX = "layer_int8.hex") (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    output reg  [31:0] checksum,
    output reg  [31:0] words
);
    (* ram_style = "block" *) reg [127:0] wmem [0:NLINES-1];
    initial $readmemh(IMG_HEX, wmem);
    reg [$clog2(NLINES):0] addr; reg half, active; reg [127:0] line; reg [63:0] out_data; reg out_valid;
    always @(posedge clk) begin line <= wmem[addr[$clog2(NLINES)-1:0]]; end
    integer k; reg [31:0] cs_next;
    always @(*) begin cs_next = checksum; if (out_valid) for (k = 0; k < OUTW; k = k + 1) cs_next = {cs_next[30:0], cs_next[31]} ^ {24'd0, out_data[k*8 +: 8]}; end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin addr <= 0; half <= 0; active <= 0; done <= 0; checksum <= 0; words <= 0; out_valid <= 0; out_data <= 0; end
        else begin
            out_valid <= 1'b0;
            if (start) begin addr <= 0; half <= 0; active <= 1'b1; done <= 1'b0; checksum <= 0; words <= 0; end
            else if (active) begin
                half <= ~half;
                if (half) begin out_data <= line[63:0]; out_valid <= 1'b1; if (addr == NLINES) begin active <= 1'b0; done <= 1'b1; end end
                else begin out_data <= line[127:64]; out_valid <= 1'b1; addr <= addr + 1; end
            end
            checksum <= cs_next; if (out_valid) words <= words + OUTW;
        end
    end
endmodule
