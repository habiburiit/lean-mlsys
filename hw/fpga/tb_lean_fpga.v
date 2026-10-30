`timescale 1ns/1ps
module tb_lean_fpga;
  reg clk=0, rst_n=0, start=0; wire done; wire [31:0] checksum, words;
  lean_fpga_top #(.BLOCK(32),.PAR(14),.NBLOCKS(1152),.NLINES(989),.W0(8),.W1(6),.W2(4),.W3(3),
                  .R0(256),.R1(805),.R2(3100),.R3(5853),.MAP_HEX("layer_map.hex"),.IMG_HEX("layer_image.hex")) dut
    (.clk(clk),.rst_n(rst_n),.start(start),.done(done),.checksum(checksum),.words(words));
  always #5 clk=~clk;
  initial begin #22 rst_n=1; #20 @(posedge clk) start=1; @(posedge clk) start=0;
    wait(done); #20 $display("DONE words=%0d checksum=%08x", words, checksum); $finish; end
  initial begin #20000000 $display("TIMEOUT words=%0d", words); $finish; end
endmodule
