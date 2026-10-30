# LEAN hardware: tiered weight memory and streaming loader

The loader reads the sensitivity-ordered 8/6/4/3-bit weight format (blocks of 32 weights,
one width per block, a 2-bit tier map, the 8-bit tier carrying 14 parity bits per 16 weights)
and delivers 8 INT8 weights per cycle to an unchanged systolic array, forwarding the parity of
the protected blocks to a checker. Nothing in the array, the accumulators or the requantization
path changes.

```
hw/
  lean_pack.py              storage format: bit-exact packer/unpacker, golden vectors, bits-per-weight report
  rtl/lean_unpack.v         streaming unpacker: 128-bit lines in, 8 INT8 words per cycle out, parity forwarded
  rtl/lean_weight_loader.v  tier-map controller + line fetch + unpacker (the synthesized block)
  tb/tb_lean_unpack.v       unit test of the unpacker against the golden vectors
  tb/tb_lean_loader.v       system test of the loader, with bit-flip injection (+FLIPS=n) and a CRC-14 checker model
  tb/saif_ctl.sv            switching-activity recording for the simulation-driven power run
  syn/dc_lean_loader.tcl    Design Compiler script for lean_weight_loader (standalone block)
  run_power_saif.sh         VCS run with SAIF, then DC power with read_saif
  lean_hw_agent.py          driver: pack -> VCS simulation -> DC synthesis -> result table, with live logs
  fpga/                     Vivado flows: loader alone (run_fpga.sh), post-route simulation (run_xsim.sh),
                            loader next to the 8x8 systolic accelerator (run_fpga_accel.sh), cycle-accurate
                            throughput and latency (run_cycles.sh), ZCU102 and Versal targets (run_zcu*.sh)
  fpga_soc/                 the loader as an AXI-Lite IP in a Zynq UltraScale+ block design (run_soc.sh)
  doc/                      block diagram
```

## Software model and vectors

```
python lean_pack.py --ckpt <model.pth> --bins <model.bins.pt> --tiers 8,6,4,3 --block 32 --out vec_<tag>
```
writes the packed image, the tier map, the expected INT8 words and the parity, and reports bits per weight.
`--demo-group` packs a synthetic model when no checkpoint is at hand.

## Simulation and synthesis (VCS, Design Compiler, 45 nm)

```
python lean_hw_agent.py --ckpt <model.pth> --tiers 8,6,4,3 --block 32          # all steps
python lean_hw_agent.py --step sim|syn|table ...                                # one step
```
The agent fills the library lines of `syn/dc_lean_loader.tcl` (search_path, target_library, link_library)
from the environment it finds; edit the three `set` lines at the top of the generated
`syn/dc_lean_loader_<tag>.tcl` if your library lives elsewhere. Reports land in `syn/reports/`. The
simulation-driven power number is produced by `bash run_power_saif.sh <tag>` (reports in `syn/reports_saif/`).

## FPGA (Vivado 2020.x)

```
bash fpga/run_fpga.sh            # implement INT8 read-out, loader, loader + parity checker; writes fpga/fpga_table.md
bash fpga/run_xsim.sh            # post-route gate-level simulation of the loader against the packer's words
bash fpga/run_fpga_accel.sh      # loader next to the full 8x8 systolic accelerator (overhead rows)
bash fpga/run_cycles.sh          # RTL simulation: cycles to first weight, total cycles, words per cycle, vs INT8 read-out
bash fpga_soc/run_soc.sh         # Zynq UltraScale+ block design with the loader as an AXI-Lite IP
```
Each script writes a log next to itself and prints the result lines at the end.
