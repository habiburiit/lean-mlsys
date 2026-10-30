// ============================================================================
//  tb_lean_unpack.v  -  checks lean_unpack against the golden vectors written by hw/lean_pack.py.
//  Reads  <DIR>/image.hex (128-bit lines), blocks.txt (tensor tier ratio nwords), expected.txt (INT8 words), parity.txt (hex).
//  Pass the directory with  +DIR=<path>  (default "vec").  Prints PASS/FAIL counts and the cycle count.
//  Parameters must match format.json (tiers, block, parity).  Run:
//     iverilog -g2005 -o tb rtl/lean_unpack.v tb/tb_lean_unpack.v && vvp tb +DIR=vec
//     vcs -full64 -sverilog rtl/lean_unpack.v tb/tb_lean_unpack.v -o simv && ./simv +DIR=vec
// ============================================================================
`timescale 1ns/1ps
module tb_lean_unpack;
    parameter BLOCK = 64, LINE_W = 16, PAR = 14, OUTW = 8;
    parameter W0 = 8, W1 = 4, W2 = 2, W3 = 2;
    parameter MAXL = 1 << 21, MAXB = 1 << 21, MAXW = 1 << 26;
    localparam PARBITS = PAR * (BLOCK / LINE_W);

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg          blk_start = 0; reg [1:0] blk_tier = 0; reg [15:0] blk_ratio = 0; wire blk_ready;
    reg          line_valid = 0; reg [127:0] line_data = 0; wire line_ready;
    wire         out_valid; wire [OUTW*8-1:0] out_data;
    wire         par_valid; wire [PARBITS-1:0] par_data; wire blk_done;

    lean_unpack #(.BLOCK(BLOCK), .LINE_W(LINE_W), .PAR(PAR), .OUTW(OUTW), .W0(W0), .W1(W1), .W2(W2), .W3(W3)) dut (
        .clk(clk), .rst_n(rst_n), .blk_start(blk_start), .blk_tier(blk_tier), .blk_ratio(blk_ratio), .blk_ready(blk_ready),
        .line_valid(line_valid), .line_data(line_data), .line_ready(line_ready),
        .out_valid(out_valid), .out_data(out_data), .par_valid(par_valid), .par_data(par_data), .blk_done(blk_done));

    // ---- vectors
    reg [127:0] lines [0:MAXL-1];
    integer nlines = 0, nblocks = 0, nwords = 0, npar = 0;
    integer b_tid [0:MAXB-1]; integer b_tier [0:MAXB-1]; integer b_ratio [0:MAXB-1]; integer b_nw [0:MAXB-1];
    integer exp_w [0:MAXW-1];
    reg [PARBITS-1:0] exp_par [0:MAXB-1];
    reg [16383:0] dir;
    reg [16383:0] fname;   // 2048-character paths
    integer fd, r, v, t1, t2, t3, t4;
    reg [PARBITS-1:0] pv;

    task load_vectors;
        begin
            if (!$value$plusargs("DIR=%s", dir)) dir = "vec";
            $sformat(fname, "%0s/image.hex", dir); $readmemh(fname, lines);
            $sformat(fname, "%0s/blocks.txt", dir); fd = $fopen(fname, "r");
            if (fd == 0) begin $display("cannot open %0s", fname); $finish; end
            while (!$feof(fd)) begin
                r = $fscanf(fd, "%d %d %d %d\n", t1, t2, t3, t4);
                if (r == 4) begin b_tid[nblocks] = t1; b_tier[nblocks] = t2; b_ratio[nblocks] = t3; b_nw[nblocks] = t4; nblocks = nblocks + 1; end
            end
            $fclose(fd);
            $sformat(fname, "%0s/expected.txt", dir); fd = $fopen(fname, "r");
            while (!$feof(fd)) begin r = $fscanf(fd, "%d\n", v); if (r == 1) begin exp_w[nwords] = v; nwords = nwords + 1; end end
            $fclose(fd);
            $sformat(fname, "%0s/parity.txt", dir); fd = $fopen(fname, "r");
            while (!$feof(fd)) begin r = $fscanf(fd, "%h\n", pv); if (r == 1) begin exp_par[npar] = pv; npar = npar + 1; end end
            $fclose(fd);
            nlines = 0; while (nlines < MAXL && lines[nlines] !== 128'bx) nlines = nlines + 1;
            $display("[tb] %0d lines, %0d blocks, %0d words, %0d parity blocks from %0s", nlines, nblocks, nwords, npar, dir);
        end
    endtask

    // ---- line feeder: pushes lines whenever the unpacker has room
    integer li = 0;
    always @(posedge clk) begin
        if (rst_n) begin
            if (line_valid && line_ready) begin li <= li + 1; line_valid <= (li + 1 < nlines); line_data <= lines[li + 1]; end
            else if (!line_valid && li < nlines) begin line_valid <= 1'b1; line_data <= lines[li]; end
        end
    end

    // ---- checker
    integer wi = 0, pi = 0, errs = 0, perrs = 0, cycles = 0, k;
    integer got;
    always @(posedge clk) if (rst_n) begin
        cycles <= cycles + 1;
        if (out_valid) begin
            for (k = 0; k < OUTW; k = k + 1) begin
                got = $signed(out_data[(OUTW-1-k)*8 +: 8]);
                if (wi + k < nwords && got !== exp_w[wi + k]) begin
                    errs = errs + 1;
                    if (errs <= 10) $display("[tb] MISMATCH word %0d: got %0d expected %0d", wi + k, got, exp_w[wi + k]);
                end
            end
            wi = wi + OUTW;
        end
        if (par_valid) begin
            if (par_data !== exp_par[pi]) begin perrs = perrs + 1; if (perrs <= 5) $display("[tb] PARITY MISMATCH block %0d: got %h expected %h", pi, par_data, exp_par[pi]); end
            pi = pi + 1;
        end
    end

    // ---- block sequencer (the role of the controller / tier map)
    integer bi;
    initial begin
        load_vectors;
        #22 rst_n = 1;
        for (bi = 0; bi < nblocks; bi = bi + 1) begin
            @(negedge clk); while (!blk_ready) @(negedge clk);          // drive on the falling edge: ready is settled
            blk_start = 1'b1; blk_tier = b_tier[bi]; blk_ratio = b_ratio[bi];
            @(negedge clk); blk_start = 1'b0;
        end
        while (wi < nwords) @(posedge clk);
        repeat (4) @(posedge clk);
        $display("[tb] words checked %0d (expected %0d), word errors %0d, parity blocks %0d, parity errors %0d, cycles %0d, %.3f words/cycle",
                 wi, nwords, errs, pi, perrs, cycles, 1.0 * nwords / cycles);
        if (errs == 0 && perrs == 0 && wi >= nwords && pi == npar) $display("[tb] PASS"); else $display("[tb] FAIL");
        $finish;
    end
endmodule
