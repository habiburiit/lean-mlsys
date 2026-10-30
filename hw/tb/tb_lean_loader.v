// ============================================================================
//  tb_lean_loader.v  -  system-level check of lean_weight_loader (controller + tier map + unpacker) against lean_pack.py vectors,
//  with optional bit-flip injection into the packed weight memory.
//     +DIR=<vector dir>      image.hex map.hex tensors.txt expected.txt parity.txt
//     +FLIPS=<n>             flip n random bits of the packed image before loading (default 0); +SEED=<s>
//  Reports: word errors (clean run must be 0), and with flips: how many flipped bits landed in HIGH (ECC) blocks and were flagged
//  by a parity mismatch, how many landed in LOW/MID blocks, how many INT8 words changed and the largest change.
// ============================================================================
`timescale 1ns/1ps
module tb_lean_loader;
    parameter BLOCK = 64, LINE_W = 16, PAR = 14, OUTW = 8;
    parameter W0 = 8, W1 = 4, W2 = 2, W3 = 2;
    parameter MAXL = 1 << 21, MAXB = 1 << 21, MAXW = 1 << 26, MAXT = 4096;
    localparam PARBITS = PAR * (BLOCK / LINE_W);

    reg clk = 0, rst_n = 0; always #5 clk = ~clk;

    reg tns_start = 0; reg [19:0] tns_nblocks = 0, tns_first_block = 0, tns_first_line = 0; reg [6:0] tns_bit_offset = 0;
    reg [15:0] tr0 = 0, tr1 = 0, tr2 = 0, tr3 = 0; wire tns_done;
    wire [19:0] map_addr; reg [1:0] map_data; wire [19:0] line_addr; wire line_req; reg line_data_valid = 0; reg [127:0] line_data = 0;
    wire out_valid; wire [OUTW*8-1:0] out_data; wire par_valid; wire [PARBITS-1:0] par_data; wire out_high;

    lean_weight_loader #(.BLOCK(BLOCK), .LINE_W(LINE_W), .PAR(PAR), .OUTW(OUTW), .W0(W0), .W1(W1), .W2(W2), .W3(W3)) dut (
        .clk(clk), .rst_n(rst_n), .tns_start(tns_start), .tns_nblocks(tns_nblocks), .tns_first_block(tns_first_block),
        .tns_first_line(tns_first_line), .tns_bit_offset(tns_bit_offset), .tns_ratio0(tr0), .tns_ratio1(tr1), .tns_ratio2(tr2), .tns_ratio3(tr3),
        .tns_done(tns_done), .map_addr(map_addr), .map_data(map_data), .line_addr(line_addr), .line_req(line_req),
        .line_data_valid(line_data_valid), .line_data(line_data), .out_valid(out_valid), .out_data(out_data), .par_valid(par_valid), .par_data(par_data), .out_high(out_high));

    // ---- memories (one-cycle read latency)
    reg [127:0] lines [0:MAXL-1]; reg [1:0] tmap [0:MAXB-1];
    always @(posedge clk) begin map_data <= tmap[map_addr]; line_data_valid <= line_req; line_data <= lines[line_addr]; end

    // ---- vectors
    integer nlines = 0, nblocks = 0, nwords = 0, npar = 0, ntens = 0;
    integer t_nb [0:MAXT-1]; integer t_r0 [0:MAXT-1]; integer t_r1 [0:MAXT-1]; integer t_r2 [0:MAXT-1]; integer t_r3 [0:MAXT-1];
    integer exp_w [0:MAXW-1]; reg [PARBITS-1:0] exp_par [0:MAXB-1];
    reg [16383:0] dir; reg [16383:0] fname;   /* 2048-character paths */ integer fd, r, v, a1, a2, a3, a4, a5, a6; reg [PARBITS-1:0] pv;
    integer nflips = 0, seed = 1, fi, bitpos, lno, flips_high = 0, flips_low = 0, blk_of_bit;
    // block index of a bit position: walk the blocks (block sizes depend on the tier); precomputed start offsets
    integer blk_start_bit [0:MAXB-1]; integer bitcur, bi2;

    task load_vectors;
        begin
            if (!$value$plusargs("DIR=%s", dir)) dir = "vec";
            if (!$value$plusargs("FLIPS=%d", nflips)) nflips = 0;
            if (!$value$plusargs("SEED=%d", seed)) seed = 1;
            $sformat(fname, "%0s/image.hex", dir); $readmemh(fname, lines);
            $sformat(fname, "%0s/map.hex", dir); $readmemh(fname, tmap);
            $sformat(fname, "%0s/tensors.txt", dir); fd = $fopen(fname, "r");
            if (fd == 0) begin $display("cannot open %0s", fname); $finish; end
            while (!$feof(fd)) begin
                r = $fscanf(fd, "%d %d %d %d %d %d\n", a1, a2, a3, a4, a5, a6);
                if (r == 6) begin t_nb[ntens] = a2; t_r0[ntens] = a3; t_r1[ntens] = a4; t_r2[ntens] = a5; t_r3[ntens] = a6; ntens = ntens + 1; nblocks = nblocks + a2; end
            end
            $fclose(fd);
            $sformat(fname, "%0s/expected.txt", dir); fd = $fopen(fname, "r");
            while (!$feof(fd)) begin r = $fscanf(fd, "%d\n", v); if (r == 1) begin exp_w[nwords] = v; nwords = nwords + 1; end end
            $fclose(fd);
            $sformat(fname, "%0s/parity.txt", dir); fd = $fopen(fname, "r");
            while (!$feof(fd)) begin r = $fscanf(fd, "%h\n", pv); if (r == 1) begin exp_par[npar] = pv; npar = npar + 1; end end
            $fclose(fd);
            nlines = 0; while (nlines < MAXL && lines[nlines] !== 128'bx) nlines = nlines + 1;
            // block start offsets in the bit stream
            bitcur = 0;
            for (bi2 = 0; bi2 < nblocks; bi2 = bi2 + 1) begin
                blk_start_bit[bi2] = bitcur;
                case (tmap[bi2]) 2'd0: bitcur = bitcur + BLOCK*W0 + PARBITS; 2'd1: bitcur = bitcur + BLOCK*W1; 2'd2: bitcur = bitcur + BLOCK*W2; default: bitcur = bitcur + BLOCK*W3; endcase
            end
            $display("[tb] %0d tensors, %0d blocks, %0d lines, %0d words, %0d parity blocks, %0d stream bits", ntens, nblocks, nlines, nwords, npar, bitcur);
            // fault injection
            for (fi = 0; fi < nflips; fi = fi + 1) begin
                bitpos = $urandom(seed) % bitcur; seed = seed + 1;
                lno = bitpos / 128;
                lines[lno][127 - (bitpos % 128)] = ~lines[lno][127 - (bitpos % 128)];
                blk_of_bit = 0; while (blk_of_bit + 1 < nblocks && blk_start_bit[blk_of_bit + 1] <= bitpos) blk_of_bit = blk_of_bit + 1;
                if (tmap[blk_of_bit] == 2'd0) flips_high = flips_high + 1; else flips_low = flips_low + 1;
            end
            if (nflips > 0) $display("[tb] injected %0d bit flips: %0d in HIGH (ECC) blocks, %0d in lower tiers", nflips, flips_high, flips_low);
        end
    endtask

    // ---- placeholder checker model: CRC-14 (poly 0x4443) over each 16-word line of a HIGH block, same function as lean_pack.py.
    //      In the real design the the selective code BCH checker sits here; this only demonstrates the routing and the detection count.
    function [PAR-1:0] crc14(input [127:0] line); integer q; reg [PAR-1:0] rg; reg top; begin
        rg = 0; for (q = 127; q >= 0; q = q - 1) begin top = rg[PAR-1]; rg = {rg[PAR-2:0], line[q]}; if (top) rg = rg ^ 14'h0443; end
        crc14 = rg; end endfunction
    reg [127:0] hline; integer hgrp = 0, hl = 0, flagged = 0, storedbad = 0; reg [PARBITS-1:0] recomputed;
    integer wi = 0, pi = 0, errs = 0, perrs = 0, cycles = 0, k, got, maxd = 0, d;
    always @(posedge clk) if (rst_n) begin
        cycles <= cycles + 1;
        if (out_valid && out_high) begin                                   // gather HIGH words into 128-bit lines and recompute parity
            hline = {hline[63:0], out_data}; hgrp = hgrp + 1;
            if (hgrp == 2) begin recomputed[PARBITS-1 - hl*PAR -: PAR] = crc14(hline); hgrp = 0; hl = hl + 1; if (hl == BLOCK/LINE_W) hl = 0; end
        end
        if (par_valid && par_data !== recomputed) begin flagged = flagged + 1; end
        if (out_valid) begin
            for (k = 0; k < OUTW; k = k + 1) begin
                got = $signed(out_data[(OUTW-1-k)*8 +: 8]);
                if (wi + k < nwords && got !== exp_w[wi + k]) begin
                    errs = errs + 1; d = (got > exp_w[wi + k]) ? got - exp_w[wi + k] : exp_w[wi + k] - got; if (d > maxd) maxd = d;
                    if (nflips == 0 && errs <= 10) $display("[tb] MISMATCH word %0d: got %0d expected %0d", wi + k, got, exp_w[wi + k]);
                end
            end
            wi = wi + OUTW;
        end
        if (par_valid) begin if (par_data !== exp_par[pi]) perrs = perrs + 1; pi = pi + 1; end
    end

    // ---- tensor sequencer (software)
    integer ti, fb;
    initial begin
        load_vectors;
        #22 rst_n = 1; fb = 0;
        for (ti = 0; ti < ntens; ti = ti + 1) begin
            @(negedge clk);
            tns_start = 1; tns_nblocks = t_nb[ti]; tns_first_block = fb; tns_first_line = 0; tr0 = t_r0[ti]; tr1 = t_r1[ti]; tr2 = t_r2[ti]; tr3 = t_r3[ti];
            @(negedge clk); tns_start = 0; fb = fb + t_nb[ti];
            @(posedge tns_done);
        end
        repeat (8) @(posedge clk);
        $display("[tb] words checked %0d (expected %0d), word errors %0d (max |dq8| %0d), parity blocks %0d, parity mismatches %0d, cycles %0d, %.3f words/cycle",
                 wi, nwords, errs, maxd, pi, perrs, cycles, 1.0 * nwords / cycles);
        if (nflips == 0) begin
            if (errs == 0 && perrs == 0 && flagged == 0 && wi >= nwords && pi == npar) $display("[tb] PASS"); else $display("[tb] FAIL (checker model flagged %0d clean blocks)", flagged);
        end else begin
            $display("[tb] FLIP SUMMARY: %0d flips in HIGH blocks -> %0d HIGH blocks flagged by the checker model (%0d had the flip in the parity field); %0d flips in lower tiers; %0d INT8 words changed in total (uncorrected, all tiers), largest change %0d (of 127)",
                     flips_high, flagged, perrs, flips_low, errs, maxd);
        end
        $finish;
    end
endmodule
