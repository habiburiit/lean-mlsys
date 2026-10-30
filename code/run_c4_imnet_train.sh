#!/usr/bin/env bash
# run_c4_imnet_train.sh <GPU> <tail|clip>  -  ImageNet ResNet-50 single-term fine-tune (same recipe as the lam40 model:
#   10 epochs, lr 5e-4, bs 32, lam 40), then per-channel PBS with the 1% code on it.   Log: sensreg_results/c4/imnet_<reg>.log
PY=$(command -v python 2>/dev/null); [ -z "$PY" ] && PY=python
export CUDA_VISIBLE_DEVICES=$1; REG=$2; IM=${IMAGENET_ROOT:-./imagenet-val}
D=artifacts/models/imagenet/resnet50; B=$D/model_float32.pth; OUT=$D/ablate_${REG}_lam40.pth
O=sensreg_results/c4; mkdir -p $O; LOG=$O/imnet_$REG.log; : > $LOG
{
echo "=== ImageNet ResNet-50 P_$REG only, lam 40  $(date) ==="
[ -f $OUT ] || $PY -u lean_sensreg.py train --dataset imagenet --arch resnet50 --imagenet-root $IM --init $B --reg $REG --lam 40 --lr 0.0005 --bs 32 --epochs 10 --out $OUT
echo "--- HIGH/MID per-channel ---"
$PY -u lean_hist_pc.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpt $B --ckpt2 $OUT --out $O/hist_pc_imnet_$REG | grep -A4 "Bin occupancy\|tail_mass\|clean_acc\|ratio_high"
echo "--- PBS 1% code, per-channel  $(date) ---"
$PY -u lean_pbs_codec_pc.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpt $OUT --cov 0.01 --attack pbs --exclude-coded 1 --flips 100,200,300,500 --label imnet_${REG}40_pc_code1x
echo "DONE c4_imnet_$REG $(date)"
} >> $LOG 2>&1
