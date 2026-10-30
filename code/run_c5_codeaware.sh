#!/usr/bin/env bash
# run_c5_codeaware.sh <GPU>  -  C5: code-aware attacker (knows the coded set and the key, overloads codewords: lean_pbs_codec.py
#   --attack cwaware) against normal and LEAN lam40 ResNet-18 CIFAR-10 at 1, 2, 5 and 10% coverage, INT8.
#   Reports accuracy against the number of overloaded codewords (13 injected errors each).   Log: sensreg_results/c5/codeaware.log
PY=$(command -v python 2>/dev/null); [ -z "$PY" ] && PY=python
export CUDA_VISIBLE_DEVICES=${1:-0}
CK=artifacts/models/cifar10/resnet18; B=$CK/model_float32.pth; L=$CK/model_float32_sensreg_lam40.pth
O=sensreg_results/c5; mkdir -p $O; LOG=$O/codeaware.log; : > $LOG
{
echo "=== C5 code-aware attacker, coverage sweep  $(date) ==="
for COV in 0.01 0.02 0.05 0.10; do
  for X in $B:base $L:lam40; do
    echo "--- cov $COV ${X##*:}  $(date) ---"
    $PY -u lean_pbs_codec.py --dataset cifar10 --arch resnet18 --ckpt ${X%%:*} --cov $COV --attack cwaware --cw-counts 1,2,5,10,20,40 --label c10_${X##*:}_cw_cov$COV
  done
done
echo "DONE c5 $(date)"
} >> $LOG 2>&1
