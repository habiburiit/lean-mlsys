# LEAN energy with Timeloop and Accelergy

What it does: maps every ResNet-50 weight layer (54 layers, 25 unique shapes) on an Eyeriss-style
168-PE accelerator (45nm) with the Timeloop mapper, takes the per-level per-datatype energy from
Accelergy (CACTI for SRAM/DRAM, Aladdin tables for register files and MACs), then recomputes the
total when weights are stored at 3.5 or 4.0 total bits (LEAN format) instead of 8 (INT8).

Files
  arch_int8.yaml        accelerator (Timeloop v0.3 YAML): DRAM, 64-bit global buffer, 14x12 PEs
  constraints_rs.yaml   Eyeriss row-stationary mapspace constraints
  mapper.yaml           mapper settings (random-pruned, 4 threads, victory 300)
  probs/*.yaml          26 ResNet-50 layer shapes; probs/layers.json = shape + multiplicity
  run_all.sh            runs timeloop-mapper on every layer, log in runs/all.log (tail -f)
  lean_energy.py        aggregates runs/*/timeloop-mapper.stats.txt into the INT8 vs LEAN table
  results/              the finished run: all 26 stats files, all.log, lean_energy.json
  primitive_component.lib.yaml   Accelergy primitive library (needed by Accelergy)

Run (after install)
  ./run_all.sh            # ~25 min, watch with: tail -f runs/all.log
  python3 lean_energy.py  # prints the table

Install (tested in a clean Ubuntu-like sandbox, python 3.x)
  1. sudo apt-get install -y scons libconfig++-dev libboost-dev libboost-iostreams-dev libboost-serialization-dev libyaml-cpp-dev libncurses-dev libtinfo-dev libgpm-dev git build-essential
  2. git clone https://github.com/NVlabs/timeloop.git && cd timeloop && git checkout v3.0.3
     # if the build complains about uint64_t / cstdint, add  #include <cstdint>  at the top of the file it names
     scons -j8 --accelergy && sudo cp build/timeloop-* /usr/local/bin/ && sudo cp lib/*.so /usr/local/lib/ 2>/dev/null; sudo ldconfig
  3. pip install git+https://github.com/Accelergy-Project/accelergy.git
  4. for p in accelergy-aladdin-plug-in accelergy-cacti-plug-in accelergy-table-based-plug-ins accelergy-library-plug-in; do
       git clone https://github.com/Accelergy-Project/$p.git && (cd $p && pip install .)
     done
     (cd accelergy-cacti-plug-in/cacti && make -j8)      # CACTI binary
  5. sudo cp primitive_component.lib.yaml /usr/local/share/accelergy/primitive_component_libs/
     # newer Accelergy ships without it; the copy here is from accelergy commit 28b9079^
  6. Aladdin plug-in patch (newer Accelergy does not pass action_latency_cycles): in
     /usr/local/share/accelergy/estimation_plug_ins/accelergy-aladdin-plug-in/aladdin_table.py replace
       latency = global_cycle_seconds * interface["arguments"]["action_latency_cycles"]
     with
       latency = global_cycle_seconds * (interface.get("arguments") or {}).get("action_latency_cycles", interface["attributes"].get("action_latency_cycles", 1))
  7. check:  accelergy arch_int8.yaml -o acc_test   -> acc_test/ERT_summary.yaml, MACs must say Aladdin_table (not dummy)

Energy model for LEAN (lean_energy.py): DRAM weight energy x (bits/8); plus one 8-bit bitwise unpack per
weight delivered to a PE (0.15 pJ); inputs, outputs, compute, register files unchanged (conservative).
