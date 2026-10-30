#!/usr/bin/env bash
# run_lean_ablation_pbs.sh  -  real PBS with the 1% BCH code (as in Table II) on the ablation checkpoints.
# Uses the same swap trick as before: pbs_four.py loads artifacts/models/CIFAR10/resnet18/model_float32.pth,
# so each checkpoint is copied there in turn and the original is restored at the end (and on Ctrl-C).
# usage:  nohup bash run_lean_ablation_pbs.sh <GPU> > /dev/null 2>&1 &   then   tail -f sensreg_results/cifar10_resnet18/ablation_pbs.log
export CUDA_VISIBLE_DEVICES=$1
SRC=artifacts/models/cifar10/resnet18
PBS_DIR=$(dirname $(find artifacts -iname model_float32.pth -ipath "*cifar10*resnet18*" | grep -v "/cifar10/" | head -1))
[ -z "$PBS_DIR" ] && PBS_DIR=artifacts/models/CIFAR10/resnet18
ORIG=$PBS_DIR/model_float32.pth; BAK=$PBS_DIR/model_float32.pth.ORIG_BAK
O=sensreg_results/cifar10_resnet18; mkdir -p $O; LOG=$O/ablation_pbs.log; : > $LOG
restore () { [ -f $BAK ] && cp $BAK $ORIG && echo "[restored original $ORIG]" >> $LOG; }
trap restore EXIT
{
echo "pbs model path: $ORIG"
cp $ORIG $BAK
for TAG in clip_only:ablate_clip_lam40.pth tail_only:ablate_tail_lam40.pth tail_lam400:ablate_tail_lam400.pth; do
  LAB=${TAG%%:*}; F=${TAG##*:}
  echo; echo "=== $LAB  ($SRC/$F) ==="
  cp $SRC/$F $ORIG
  python -u pbs_four.py --dataset CIFAR10 --arch resnet18 --bits 8 --protect selcode --covs 0.01 --flips 100,200,300,500 --label $LAB
done
echo "DONE ablation_pbs"
} >> $LOG 2>&1
