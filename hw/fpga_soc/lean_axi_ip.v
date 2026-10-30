`timescale 1ns/1ps
// lean_axi_ip.v  -  AXI4-Lite wrapper around lean_accel_top (systolic accelerator + LEAN weight loader + ECC checker), so the design
// can be packaged as an IP block and dropped into a Vivado block design next to a processor, interconnect, UART and memory.
// Register map (32-bit, byte offsets):
//   0x00 CTRL    w: bit0 = start pulse (accelerator + loader)
//   0x04 STATUS  r: bit0 accel done, bit1 LEAN loader done
//   0x08 LEAN_CHECKSUM r     0x0C LEAN_WORDS r     0x10 LEAN_ECC_ERR r (parity mismatches seen by the checker)
//   0x14 IMG_IDX rw (test image index)     0x18 RES_ADDR rw (bits 14:0; a write also pulses res_re)     0x1C RES_DATA r: [7:0] dout, [9:8] bank
module lean_axi_ip #(
    parameter C_S_AXI_ADDR_WIDTH = 6,
    parameter MEM_DIR = "mem_clean/",
    parameter BLOCK = 32, parameter PAR = 14, parameter NBLOCKS = 1024, parameter NLINES = 1024,
    parameter W0 = 8, parameter W1 = 6, parameter W2 = 4, parameter W3 = 3,
    parameter R0 = 256, parameter R1 = 1024, parameter R2 = 4096, parameter R3 = 8192,
    parameter MAP_HEX = "layer_map.hex", parameter IMG_HEX = "layer_image.hex"
)(
    input  wire                          s_axi_aclk,
    input  wire                          s_axi_aresetn,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  wire [2:0]                    s_axi_awprot,
    input  wire                          s_axi_awvalid,
    output reg                           s_axi_awready,
    input  wire [31:0]                   s_axi_wdata,
    input  wire [3:0]                    s_axi_wstrb,
    input  wire                          s_axi_wvalid,
    output reg                           s_axi_wready,
    output reg  [1:0]                    s_axi_bresp,
    output reg                           s_axi_bvalid,
    input  wire                          s_axi_bready,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  wire [2:0]                    s_axi_arprot,
    input  wire                          s_axi_arvalid,
    output reg                           s_axi_arready,
    output reg  [31:0]                   s_axi_rdata,
    output reg  [1:0]                    s_axi_rresp,
    output reg                           s_axi_rvalid,
    input  wire                          s_axi_rready,
    output wire                          irq_done          // accelerator done, for the interrupt controller
);
    // ---- core
    reg start, res_re; reg [15:0] img_idx; reg [14:0] res_addr;
    wire done, lean_done; wire signed [7:0] res_dout; wire [1:0] res_bank; wire [31:0] lean_checksum, lean_words, lean_ecc_err;
    lean_accel_top_soc #(.WITH_LEAN(1), .MEM_DIR(MEM_DIR), .BLOCK(BLOCK), .PAR(PAR), .NBLOCKS(NBLOCKS), .NLINES(NLINES),
                     .W0(W0), .W1(W1), .W2(W2), .W3(W3), .R0(R0[15:0]), .R1(R1[15:0]), .R2(R2[15:0]), .R3(R3[15:0]), .MAP_HEX(MAP_HEX), .IMG_HEX(IMG_HEX)) u_core (
        .clk(s_axi_aclk), .rst_n(s_axi_aresetn), .start(start), .img_idx(img_idx), .res_re(res_re), .res_addr(res_addr),
        .done(done), .res_dout(res_dout), .res_bank(res_bank), .lean_done(lean_done), .lean_checksum(lean_checksum),
        .lean_words(lean_words), .lean_ecc_err(lean_ecc_err));
    assign irq_done = done;

    // ---- AXI4-Lite write: accept address and data together, respond with OKAY
    wire wr = s_axi_awvalid && s_axi_wvalid && !s_axi_bvalid;
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_awready <= 1'b0; s_axi_wready <= 1'b0; s_axi_bvalid <= 1'b0; s_axi_bresp <= 2'b00;
            start <= 1'b0; res_re <= 1'b0; img_idx <= 16'd0; res_addr <= 15'd0;
        end else begin
            s_axi_awready <= wr; s_axi_wready <= wr;
            start <= 1'b0; res_re <= 1'b0;
            if (wr) begin
                case (s_axi_awaddr[5:2])
                    4'h0: start    <= s_axi_wdata[0];
                    4'h5: img_idx  <= s_axi_wdata[15:0];
                    4'h6: begin res_addr <= s_axi_wdata[14:0]; res_re <= 1'b1; end
                    default: ;
                endcase
                s_axi_bvalid <= 1'b1; s_axi_bresp <= 2'b00;
            end else if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;
        end
    end

    // ---- AXI4-Lite read
    wire rd = s_axi_arvalid && !s_axi_rvalid;
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin s_axi_arready <= 1'b0; s_axi_rvalid <= 1'b0; s_axi_rdata <= 32'd0; s_axi_rresp <= 2'b00; end
        else begin
            s_axi_arready <= rd;
            if (rd) begin
                case (s_axi_araddr[5:2])
                    4'h0: s_axi_rdata <= 32'd0;
                    4'h1: s_axi_rdata <= {30'd0, lean_done, done};
                    4'h2: s_axi_rdata <= lean_checksum;
                    4'h3: s_axi_rdata <= lean_words;
                    4'h4: s_axi_rdata <= lean_ecc_err;
                    4'h5: s_axi_rdata <= {16'd0, img_idx};
                    4'h6: s_axi_rdata <= {17'd0, res_addr};
                    4'h7: s_axi_rdata <= {22'd0, res_bank, res_dout};
                    default: s_axi_rdata <= 32'hDEAD0000;
                endcase
                s_axi_rvalid <= 1'b1; s_axi_rresp <= 2'b00;
            end else if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 1'b0;
        end
    end
endmodule
