#!/usr/bin/env bash
# run_lean_nvfp4_pbs_imnet.sh <GPU>  -  overnight: PBS attack on the REAL NVFP4 bit layout, ImageNet ResNet-50.
#   control (lam0) without code, control with the 1% code, every LEAN v3 model found with the 1% code.
#   Gives the ImageNet attack row of the NVFP4 subsection.  SEPARATE file; uses lean_pbs_nvfp4.py unchanged.
# usage:  nohup bash run_lean_nvfp4_pbs_imnet.sh 1 > /dev/null 2>&1 &     tail -f sensreg_results/nvfp4/pbs_imagenet_resnet50.log
export CUDA_VISIBLE_DEVICES=${1:-0}
O=sensreg_results/nvfp4; mkdir -p $O; LOG=$O/pbs_imagenet_resnet50.log; : > $LOG
D=artifacts/models/imagenet/resnet50/nvfp4
IM=${IMAGENET_ROOT:-./imagenet-val}
COMMON="--dataset imagenet --arch resnet50 --imagenet-root $IM --flips 10,25,50,100,200,300,500 --topk 50 --attack-batches 2 --probe-batches 8"
{
echo "=== NVFP4 PBS ImageNet ResNet-50  $(date) ==="; ls -la $D
C=$(ls $D/lam0.pth 2>/dev/null || ls $D/*lam0*.pth 2>/dev/null | head -1)
if [ -z "$C" ]; then echo "no control checkpoint in $D"; exit 1; fi
echo "--- control, no code ---"
python -u lean_pbs_nvfp4.py $COMMON --ckpt $C --protect none --label imnet_control_nocode
echo "--- control, 1% code ---"
python -u lean_pbs_nvfp4.py $COMMON --ckpt $C --protect selcode --cov 0.01 --label imnet_control_1pct
for M in $(ls $D/v3_lam*.pth 2>/dev/null); do
  L=$(basename $M .pth)
  echo "--- LEAN $L, 1% code ---"
  python -u lean_pbs_nvfp4.py $COMMON --ckpt $M --protect selcode --cov 0.01 --label imnet_${L}_1pct
done
echo "--- control, 2% code (fair-budget row) ---"
python -u lean_pbs_nvfp4.py $COMMON --ckpt $C --protect selcode --cov 0.02 --label imnet_control_2pct
echo "DONE nvfp4 pbs imagenet $(date)"
} >> $LOG 2>&1
