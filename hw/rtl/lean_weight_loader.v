`timescale 1ns/1ps
// ============================================================================
//  lean_weight_loader.v  -  LEAN weight loader = tier-map controller + lean_unpack.
//
//  Sits between the packed weight SRAM and the weight input of the systolic array.  Per tensor, software writes one
//  descriptor (number of blocks, four Q8.8 ratios); the controller walks the tier map (2 bits per block, external memory with
//  one-cycle read latency), issues one block descriptor per block to the unpacker, and generates consecutive line addresses
//  for the weight SRAM.  Outputs are the INT8 words for the array and, for HIGH blocks, the parity bits for the the selective code checker.
//
//  Memory interfaces (both synchronous, read data valid one cycle after the address):
//     map_addr  -> map_data[1:0]        tier map        (NBLK_W-bit address)
//     line_addr -> line_data[127:0]     weight SRAM     (NLINE_W-bit address)
//  Software interface: tns_start pulse with tns_nblocks and tns_ratio0..3; tns_done pulse at the end of the tensor.
// ============================================================================
module lean_weight_loader #(
    parameter BLOCK = 64, parameter LINE_W = 16, parameter PAR = 14, parameter OUTW = 8,
    parameter W0 = 8, parameter W1 = 4, parameter W2 = 2, parameter W3 = 2,
    parameter NBLK_W = 20, parameter NLINE_W = 20
)(
    input  wire                 clk,
    input  wire                 rst_n,
    // tensor descriptor (software)
    input  wire                 tns_start,
    input  wire [NBLK_W-1:0]    tns_nblocks,
    input  wire [NBLK_W-1:0]    tns_first_block,      // index of the tensor's first block in the tier map
    input  wire [NLINE_W-1:0]   tns_first_line,       // first weight-SRAM line of the tensor (blocks are not line aligned:
                                                      //   the bit offset inside that line is tns_bit_offset)
    input  wire [6:0]           tns_bit_offset,
    input  wire [15:0]          tns_ratio0, tns_ratio1, tns_ratio2, tns_ratio3,
    output reg                  tns_done,
    // tier map memory
    output reg  [NBLK_W-1:0]    map_addr,
    input  wire [1:0]           map_data,
    // weight SRAM
    output reg  [NLINE_W-1:0]   line_addr,
    output reg                  line_req,
    input  wire                 line_data_valid,
    input  wire [127:0]         line_data,
    // to the PE array / checker
    output wire                 out_valid,
    output wire [OUTW*8-1:0]    out_data,
    output wire                 par_valid,
    output wire [PAR*(BLOCK/LINE_W)-1:0] par_data,
    output wire                 out_high
);
    // ---------------------------------------------------------------- unpacker
    wire blk_ready, line_ready, blk_done; wire [9:0] acc_cnt;
    reg  blk_start; reg [1:0] blk_tier; reg [15:0] blk_ratio;
    reg  line_valid; reg [127:0] line_q;

    lean_unpack #(.BLOCK(BLOCK), .LINE_W(LINE_W), .PAR(PAR), .OUTW(OUTW), .W0(W0), .W1(W1), .W2(W2), .W3(W3)) u_unpack (
        .clk(clk), .rst_n(rst_n), .blk_start(blk_start), .blk_tier(blk_tier), .blk_ratio(blk_ratio), .blk_ready(blk_ready),
        .line_valid(line_valid), .line_data(line_q), .line_ready(line_ready),
        .out_valid(out_valid), .out_data(out_data), .par_valid(par_valid), .par_data(par_data), .blk_done(blk_done),
        .out_high(out_high), .acc_cnt(acc_cnt));

    // ---------------------------------------------------------------- line fetch: one outstanding request, data held until accepted
    // The unpacker consumes bit streams; the first line of a tensor may start mid-line (tns_bit_offset) only when the previous
    // tensor's tail is still in the accumulator, which is the normal back-to-back case.  A fresh start (empty accumulator) must
    // begin on a line boundary; the packer guarantees this per image and reports tensor offsets in tensors.txt.
    // up to two line reads in flight; a read is issued only when the accumulator has room for every line on its way
    localparam ACCB = 384;                                        // accumulator size inside lean_unpack
    reg active; reg [1:0] inflight;
    wire [11:0] committed = acc_cnt + (inflight * 12'd128) + (line_valid ? 12'd128 : 12'd0);
    wire issue = active && (committed + 12'd128 <= ACCB) && (inflight < 2'd2);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin line_req <= 1'b0; line_addr <= {NLINE_W{1'b0}}; line_valid <= 1'b0; line_q <= 128'd0; inflight <= 2'd0; end
        else begin
            if (line_valid && line_ready) line_valid <= 1'b0;                          // line accepted by the unpacker
            if (line_data_valid) begin line_q <= line_data; line_valid <= 1'b1; end    // returned line (order preserved)
            line_req <= issue;
            if (tns_start && !active) line_addr <= tns_first_line - 1'b1;    // fetch pointer (pre-incremented on request)
            else if (issue) line_addr <= line_addr + 1'b1;
            inflight <= inflight + (issue ? 2'd1 : 2'd0) - (line_data_valid ? 2'd1 : 2'd0);
        end
    end

    // ---------------------------------------------------------------- block walk over the tier map
    localparam C_IDLE = 2'd0, C_RD = 2'd1, C_ISSUE = 2'd2, C_WAIT = 2'd3;
    reg [1:0] cs; reg [NBLK_W-1:0] blocks_left; reg [NBLK_W-1:0] blocks_done;
    reg [15:0] r0, r1, r2, r3; reg [NBLK_W-1:0] tns_nblocks_q;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cs <= C_IDLE; active <= 1'b0; blocks_left <= 0; blocks_done <= 0; map_addr <= 0; blk_start <= 1'b0; blk_tier <= 2'd0;
            blk_ratio <= 16'h0100; r0 <= 16'h0100; r1 <= 16'h0100; r2 <= 16'h0100; r3 <= 16'h0100; tns_done <= 1'b0; tns_nblocks_q <= 0;
        end else begin
            blk_start <= 1'b0; tns_done <= 1'b0;
            if (blk_done) blocks_done <= blocks_done + 1'b1;
            case (cs)
                C_IDLE: if (tns_start) begin
                    r0 <= tns_ratio0; r1 <= tns_ratio1; r2 <= tns_ratio2; r3 <= tns_ratio3; tns_nblocks_q <= tns_nblocks;
                    blocks_left <= tns_nblocks; blocks_done <= 0; map_addr <= tns_first_block;
                    active <= 1'b1; cs <= C_RD;
                end
                C_RD: begin                                                           // map_data valid next cycle
                    cs <= C_ISSUE;
                end
                C_ISSUE: if (blk_ready) begin
                    blk_start <= 1'b1; blk_tier <= map_data;
                    case (map_data) 2'd0: blk_ratio <= r0; 2'd1: blk_ratio <= r1; 2'd2: blk_ratio <= r2; default: blk_ratio <= r3; endcase
                    blocks_left <= blocks_left - 1'b1; map_addr <= map_addr + 1'b1;
                    cs <= (blocks_left == 1) ? C_WAIT : C_RD;
                end
                C_WAIT: if (blocks_done + blk_done == tns_nblocks_q) begin             // every block of the tensor finished
                    tns_done <= 1'b1; cs <= C_IDLE;
                end
            endcase
        end
    end
endmodule
