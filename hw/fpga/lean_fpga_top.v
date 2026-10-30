`timescale 1ns/1ps
// ============================================================================
//  lean_fpga_top.v  -  FPGA wrapper for the LEAN weight loader: the loader (rtl/lean_weight_loader.v + rtl/lean_unpack.v) with
//  its two memories as on-chip BRAM: a 2-bit tier map and a 128-bit packed-weight store holding ONE layer, both initialised from
//  the hex files cut by lean_fpga_prep.py.  A checksum register folds every INT8 word and parity word so nothing is optimised away.
//  Ports: clk, rst_n, start (pulse), done, checksum.  Depths/parameters are overridden by vivado_lean.tcl from fpga/layer.json.
// ============================================================================
module lean_fpga_top #(
    parameter BLOCK = 32, parameter LINE_W = 16, parameter PAR = 14, parameter OUTW = 8,
    parameter W0 = 8, parameter W1 = 6, parameter W2 = 4, parameter W3 = 3,
    parameter NBLOCKS = 1024,                 // blocks in the stored layer
    parameter NLINES  = 1024,                 // 128-bit lines in the stored layer
    parameter NBLK_W  = 20, parameter NLINE_W = 20,
    parameter [15:0] R0 = 16'h0100, parameter [15:0] R1 = 16'h0400, parameter [15:0] R2 = 16'h1000, parameter [15:0] R3 = 16'h2000,   // Q8.8 ratios of the layer
    parameter CHECK = 0,                      // 1 = include the CRC-14 parity checker (ECC placeholder) in the build
    parameter MAP_HEX = "layer_map.hex", parameter IMG_HEX = "layer_image.hex"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    output reg  [31:0] checksum,
    output reg  [31:0] words,
    output wire [31:0] ecc_err_cnt
);
    // ---- memories (inferred as block RAM, initialised from the packer's hex files)
    (* ram_style = "block" *) reg [1:0]   tmap [0:NBLOCKS-1];
    (* ram_style = "block" *) reg [127:0] wmem [0:NLINES-1];
    initial begin $readmemh(MAP_HEX, tmap); $readmemh(IMG_HEX, wmem); end

    wire [NBLK_W-1:0]  map_addr; reg [1:0] map_data;
    wire [NLINE_W-1:0] line_addr; wire line_req; reg line_data_valid; reg [127:0] line_data;
    always @(posedge clk) begin
        map_data <= tmap[map_addr[$clog2(NBLOCKS)-1:0]];
        line_data_valid <= line_req;
        line_data <= wmem[line_addr[$clog2(NLINES)-1:0]];
    end

    // ---- loader
    wire out_valid; wire [OUTW*8-1:0] out_data; wire par_valid; wire [PAR*(BLOCK/LINE_W)-1:0] par_data; wire [63:0] par_ext = {{(64-PAR*(BLOCK/LINE_W)){1'b0}}, par_data}; wire out_high; wire tns_done;
    reg  tns_start;
    lean_weight_loader #(.BLOCK(BLOCK), .LINE_W(LINE_W), .PAR(PAR), .OUTW(OUTW), .W0(W0), .W1(W1), .W2(W2), .W3(W3),
                         .NBLK_W(NBLK_W), .NLINE_W(NLINE_W)) u_loader (
        .clk(clk), .rst_n(rst_n), .tns_start(tns_start), .tns_nblocks(NBLOCKS[NBLK_W-1:0]), .tns_first_block({NBLK_W{1'b0}}),
        .tns_first_line({NLINE_W{1'b0}}), .tns_bit_offset(7'd0), .tns_ratio0(R0), .tns_ratio1(R1), .tns_ratio2(R2), .tns_ratio3(R3),
        .tns_done(tns_done), .map_addr(map_addr), .map_data(map_data), .line_addr(line_addr), .line_req(line_req),
        .line_data_valid(line_data_valid), .line_data(line_data),
        .out_valid(out_valid), .out_data(out_data), .par_valid(par_valid), .par_data(par_data), .out_high(out_high));

    // ---- optional parity checker
    generate if (CHECK) begin : g_chk
        wire chk_err;
        lean_crc14_check #(.BLOCK(BLOCK), .LINE_W(LINE_W), .PAR(PAR), .OUTW(OUTW)) u_chk (.clk(clk), .rst_n(rst_n), .out_valid(out_valid),
            .out_data(out_data), .out_high(out_high), .par_valid(par_valid), .par_data(par_data), .err(chk_err), .err_cnt(ecc_err_cnt));
    end else begin : g_nochk
        assign ecc_err_cnt = 32'd0;
    end endgenerate

    // ---- sequencing and checksum (rotate-xor over every word, so the whole datapath stays live)
    integer k;
    reg [31:0] cs_next;
    always @(*) begin
        cs_next = checksum;
        if (out_valid) for (k = 0; k < OUTW; k = k + 1) cs_next = {cs_next[30:0], cs_next[31]} ^ {24'd0, out_data[k*8 +: 8]} ^ {31'd0, out_high};
        if (par_valid) cs_next = {cs_next[30:0], cs_next[31]} ^ par_ext[31:0] ^ par_ext[63:32];
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin tns_start <= 1'b0; done <= 1'b0; checksum <= 32'd0; words <= 32'd0; end
        else begin
            tns_start <= start && !tns_start;
            if (start) begin done <= 1'b0; checksum <= 32'd0; words <= 32'd0; end
            else begin
                checksum <= cs_next;
                if (out_valid) words <= words + OUTW;
                if (tns_done) done <= 1'b1;
            end
        end
    end
endmodule
