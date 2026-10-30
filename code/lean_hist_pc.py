#!/usr/bin/env python
"""lean_hist_pc.py  -  `lean_sensreg.py hist` with the per-output-channel INT8 step (HIGH / MID against the baseline's
thresholds, both measured with the per-channel step).  Same arguments as `lean_sensreg.py hist`.  SEPARATE file.
  python lean_hist_pc.py --dataset cifar10 --arch resnet18 --ckpt <base> --ckpt2 <lean> --out <dir>"""
import sys, runpy, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_pcstep; lean_pcstep.patch()
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "lean_sensreg.py"), "hist"] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
