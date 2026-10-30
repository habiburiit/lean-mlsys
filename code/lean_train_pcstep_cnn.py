#!/usr/bin/env python
"""lean_train_pcstep_cnn.py  -  `lean_sensreg.py train` with the tail term evaluated at the PER-OUTPUT-CHANNEL INT8 flip step
(the step of per-channel deployment) instead of the per-tensor step; the CNN counterpart of lean_llm_train_pcstep.py.
Same arguments as `lean_sensreg.py train`.  SEPARATE file; lean_sensreg.py unchanged.
  python lean_train_pcstep_cnn.py --dataset imagenet --arch resnet50 --imagenet-root ./imagenet-val --init <base> --reg both --lam 40 --lr 0.0005 --bs 32 --epochs 10 --out <ckpt>"""
import sys, runpy, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_pcstep; lean_pcstep.patch()
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "lean_sensreg.py"), "train"] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
