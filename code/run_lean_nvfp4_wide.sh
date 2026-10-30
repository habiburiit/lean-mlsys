#!/usr/bin/env bash
# run_lean_nvfp4_wide.sh <GPU>  -  NVFP4 v3 with a WIDER tail (3% and 5% instead of 1%), ResNet-18 CIFAR-10.
#   Idea: with a 1% code the attacker uses the weights just below the cut; a 1% tail term only swaps them.
#   Trains v3 at tail-frac 0.03 / 0.05 x lam 100 / 200 (block-w 1), histogram vs the NVFP4 control, then real PBS with the
#   1% code (scales protected) on control and every new model, two attack seeds.  SEPARATE file; existing scripts unchanged.
# usage:  nohup bash run_lean_nvfp4_wide.sh 1 > /dev/null 2>&1 &    tail -f sensreg_results/nvfp4/wide_cifar10_resnet18.log
export CUDA_VISIBLE_DEVICES=${1:-0}
O=sensreg_results/nvfp4; mkdir -p $O; LOG=$O/wide_cifar10_resnet18.log; : > $LOG
D=artifacts/models/cifar10/resnet18/nvfp4; B=artifacts/models/cifar10/resnet18/model_float32.pth
PBS="--dataset cifar10 --arch resnet18 --flips 10,25,50,100,200,300,500 --topk 50 --attack-batches 2 --probe-batches 8"
{
echo "=== NVFP4 wide-tail, ResNet-18 CIFAR-10  $(date) ==="
[ -f $D/lam0.pth ] || { echo "control $D/lam0.pth missing"; exit 1; }
for TF in 0.03 0.05; do for L in 100 200; do
  M=$D/v3_tf${TF}_lam$L.pth
  echo "--- train tail-frac $TF lam $L ---"
  [ -f $M ] || python -u lean_train_qat_nvfp4c.py --dataset cifar10 --arch resnet18 --init $B --lam $L --block-w 1 --tail-frac $TF --out $M
  echo "--- hist tail-frac $TF lam $L (held-out) ---"
  python -u lean_hist_nvfp4.py --dataset cifar10 --arch resnet18 --split test --ckpt $D/lam0.pth --ckpt2 $M --out $O/hist_c10r18_v3_tf${TF}_lam$L
done; done
for S in 0 1; do
  echo "--- PBS seed $S: control, 1% code ---"
  python -u lean_pbs_nvfp4.py $PBS --seed $S --ckpt $D/lam0.pth --protect selcode --cov 0.01 --label c10_control_1pct_s$S
  for TF in 0.03 0.05; do for L in 100 200; do
    echo "--- PBS seed $S: tail-frac $TF lam $L, 1% code ---"
    python -u lean_pbs_nvfp4.py $PBS --seed $S --ckpt $D/v3_tf${TF}_lam$L.pth --protect selcode --cov 0.01 --label c10_tf${TF}_lam${L}_1pct_s$S
  done; done
done
echo "DONE nvfp4 wide $(date)"
} >> $LOG 2>&1
