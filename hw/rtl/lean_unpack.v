`timescale 1ns/1ps
// ============================================================================
//  lean_unpack.v  -  LEAN tiered weight-memory unpacker (streaming).
//
//  The weight SRAM holds blocks of BLOCK weights back to back as a bit stream (no line alignment).  Each block is stored at one
//  of four widths W0 > W1 > W2 > W3 (two's complement, MSB first); W0 = 8 is the ECC-protected HIGH tier, whose body is followed
//  by PAR parity bits per LINE_W weights.  A 2-bit tier map (read by the controller) says which width a block uses.
//
//  This module takes 128-bit SRAM lines in, keeps a 256-bit accumulator, and for every cycle in which enough bits are present
//  emits OUTW INT8 words for the PE array:   q8 = clip( (code * ratio + 128) >>> 8 ),  ratio = unsigned Q8.8 per tier (set by the
//  controller from a per-tensor table; a power of two makes it a pure shift).  For HIGH blocks the body is INT8 already
//  (ratio = 1.0 = 16'h0100) and the trailing parity bits are forwarded on par_data for the the selective code checker.
//
//  Format and golden vectors: hw/lean_pack.py.   Verified with hw/tb/tb_lean_unpack.v.
// ============================================================================
module lean_unpack #(
    parameter BLOCK  = 64,          // weights per block
    parameter LINE_W = 16,          // weights per parity line (HIGH blocks)
    parameter PAR    = 14,          // parity bits per parity line
    parameter OUTW   = 8,           // INT8 words emitted per cycle (8 = one row of the 8x8 array)
    parameter W0 = 8, parameter W1 = 4, parameter W2 = 2, parameter W3 = 2,
    parameter ACC  = 384           // bit accumulator (three 128-bit lines: lets the controller keep two line reads in flight)
)(
    input  wire                   clk,
    input  wire                   rst_n,
    // block descriptor from the controller (tier map + ratio table); accepted when blk_ready
    input  wire                   blk_start,
    input  wire [1:0]             blk_tier,
    input  wire [15:0]            blk_ratio,
    output wire                   blk_ready,
    // weight-memory lines
    input  wire                   line_valid,
    input  wire [127:0]           line_data,
    output wire                   line_ready,
    // unpacked INT8 words, word 0 (first weight of the group) in the top byte
    output reg                    out_valid,
    output reg  [OUTW*8-1:0]      out_data,
    // parity of a HIGH block (PAR*(BLOCK/LINE_W) bits), valid for one cycle at the end of the block
    output reg                    par_valid,
    output reg  [PAR*(BLOCK/LINE_W)-1:0] par_data,
    output reg                    blk_done,
    output reg                    out_high,        // the words on out_data belong to a HIGH (ECC-protected) block
    output wire [9:0]             acc_cnt          // bits currently held (for the fetch controller)
);
    localparam PARBITS = PAR * (BLOCK / LINE_W);
    localparam S_IDLE = 2'd0, S_BODY = 2'd1, S_PAR = 2'd2;

    reg  [ACC-1:0] acc;                 // valid bits are left-aligned: acc[ACC-1 -: cnt]
    reg  [9:0]     cnt;
    reg  [1:0]     state;
    reg  [1:0]     tier;
    reg  [15:0]    ratio;
    reg  [7:0]     words_left;          // in units of OUTW groups
    reg            pend_valid;          // one-deep descriptor queue so the next block starts without an idle cycle
    reg  [1:0]     pend_tier;
    reg  [15:0]    pend_ratio;

    // ---- width of the current tier
    reg [3:0] wd;
    always @(*) begin
        case (tier)
            2'd0: wd = W0; 2'd1: wd = W1; 2'd2: wd = W2; default: wd = W3;
        endcase
    end
    wire [9:0] need_body = wd * OUTW;                      // bits consumed per output group

    // ---- handshakes
    assign blk_ready  = !pend_valid;
    assign line_ready = (cnt <= (ACC - 128));              // room for one more line

    wire fire_body = (state == S_BODY) && (cnt >= need_body);
    wire fire_par  = (state == S_PAR)  && (cnt >= PARBITS);
    wire take_line = line_valid && line_ready;
    wire [9:0] consume = fire_body ? need_body : (fire_par ? PARBITS : 10'd0);
    assign acc_cnt  = cnt;

    // ---- code extraction for each of the OUTW words at the current width (constant-width slices, selected by tier)
    reg signed [7:0]  code  [0:OUTW-1];
    reg signed [24:0] prod  [0:OUTW-1];
    reg signed [16:0] shr   [0:OUTW-1];
    reg signed [7:0]  q8    [0:OUTW-1];
    integer i;
    always @(*) begin
        for (i = 0; i < OUTW; i = i + 1) begin
            case (tier)
                2'd0: code[i] = $signed(acc[ACC-1 - i*W0 -: W0]);
                2'd1: code[i] = $signed(acc[ACC-1 - i*W1 -: W1]);
                2'd2: code[i] = $signed(acc[ACC-1 - i*W2 -: W2]);
                default: code[i] = $signed(acc[ACC-1 - i*W3 -: W3]);
            endcase
            prod[i] = code[i] * $signed({1'b0, ratio});                 // signed x unsigned Q8.8
            shr[i]  = (prod[i] + 25'sd128) >>> 8;                       // round to nearest, arithmetic shift
            if (shr[i] > 17'sd127)       q8[i] = 8'sd127;
            else if (shr[i] < -17'sd127) q8[i] = -8'sd127;
            else                         q8[i] = shr[i][7:0];
        end
    end

    // ---- accumulator: consume (shift left) then append the incoming line at the new tail
    reg [ACC-1:0] acc_shift;
    reg [9:0]     cnt_shift;
    always @(*) begin
        acc_shift = acc << consume;
        cnt_shift = cnt - consume;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= {ACC{1'b0}}; cnt <= 10'd0; state <= S_IDLE; tier <= 2'd0; ratio <= 16'h0100; words_left <= 8'd0;
            pend_valid <= 1'b0; pend_tier <= 2'd0; pend_ratio <= 16'h0100;
            out_valid <= 1'b0; out_data <= {OUTW*8{1'b0}}; out_high <= 1'b0; par_valid <= 1'b0; par_data <= {PARBITS{1'b0}}; blk_done <= 1'b0;
        end else begin
            out_valid <= 1'b0; par_valid <= 1'b0; blk_done <= 1'b0;
            if (blk_start && !pend_valid) begin pend_valid <= 1'b1; pend_tier <= blk_tier; pend_ratio <= blk_ratio; end
            // accumulator update (consume and fill may happen in the same cycle)
            if (take_line) begin
                acc <= acc_shift | ({{(ACC-128){1'b0}}, line_data} << (ACC - 128 - cnt_shift));
                cnt <= cnt_shift + 10'd128;
            end else begin
                acc <= acc_shift;
                cnt <= cnt_shift;
            end
            // block sequencing
            case (state)
                S_IDLE: if (pend_valid) begin
                    tier <= pend_tier; ratio <= pend_ratio; words_left <= BLOCK / OUTW; state <= S_BODY; pend_valid <= 1'b0;
                end
                S_BODY: if (fire_body) begin
                    out_valid <= 1'b1; out_high <= (tier == 2'd0);
                    for (i = 0; i < OUTW; i = i + 1) out_data[(OUTW-1-i)*8 +: 8] <= q8[i];
                    words_left <= words_left - 8'd1;
                    if (words_left == 8'd1) begin
                        if (tier == 2'd0) state <= S_PAR;
                        else begin
                            blk_done <= 1'b1;
                            if (pend_valid) begin tier <= pend_tier; ratio <= pend_ratio; words_left <= BLOCK / OUTW; pend_valid <= 1'b0; end
                            else state <= S_IDLE;
                        end
                    end
                end
                S_PAR: if (fire_par) begin
                    par_valid <= 1'b1; par_data <= acc[ACC-1 -: PARBITS]; blk_done <= 1'b1;
                    if (pend_valid) begin tier <= pend_tier; ratio <= pend_ratio; words_left <= BLOCK / OUTW; state <= S_BODY; pend_valid <= 1'b0; end
                    else state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
