#!/usr/bin/env bash
# run_c4_imnet_pcstep.sh <GPU> <lam>  -  Table 6 missing row: ImageNet ResNet-50 LEAN (both terms) with the tail term evaluated at
#   the per-channel step (lean_train_pcstep_cnn.py), same recipe otherwise (10 epochs, lr 5e-4, bs 32); then per-channel HIGH,
#   true single-flip damage (against the baseline) and PBS with the 1% code.   Log: sensreg_results/c4/imnet_pcstep_lam<lam>.log
#   usage: nohup bash run_c4_imnet_pcstep.sh 0 40 > /dev/null 2>&1 &     nohup bash run_c4_imnet_pcstep.sh 1 10 > /dev/null 2>&1 &
PY=$(command -v python 2>/dev/null); [ -z "$PY" ] && PY=python
export CUDA_VISIBLE_DEVICES=$1; LAM=$2; IM=${IMAGENET_ROOT:-./imagenet-val}
D=artifacts/models/imagenet/resnet50; B=$D/model_float32.pth; OUT=$D/ablate_both_pcstep_lam$LAM.pth
O=sensreg_results/c4; mkdir -p $O; LOG=$O/imnet_pcstep_lam$LAM.log; : > $LOG
{
echo "=== ImageNet ResNet-50 LEAN both terms, per-channel step, lam $LAM  $(date) ==="
[ -f $OUT ] || $PY -u lean_train_pcstep_cnn.py --dataset imagenet --arch resnet50 --imagenet-root $IM --init $B --reg both --lam $LAM --lr 0.0005 --bs 32 --epochs 10 --out $OUT
echo "--- HIGH/MID per-channel ---"
$PY -u lean_hist_pc.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpt $B --ckpt2 $OUT --out $O/hist_pc_imnet_pcstep_lam$LAM | grep -A4 "Bin occupancy\|tail_mass\|clean_acc\|ratio_high"
echo "--- true single-flip damage, per-channel ---"
$PY -u lean_truedamage_pc.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpts $B $OUT --labels baseline pcstep_lam$LAM --topk 1000 --nrand 1000 --out $O/truedamage_pc_imnet_pcstep_lam$LAM
echo "--- PBS 1% code, per-channel  $(date) ---"
$PY -u lean_pbs_codec_pc.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpt $OUT --cov 0.01 --attack pbs --exclude-coded 1 --flips 100,200,300,500 --label imnet_pcstep_lam${LAM}_pc_code1x
echo "DONE c4_imnet_pcstep_lam$LAM $(date)"
} >> $LOG 2>&1
