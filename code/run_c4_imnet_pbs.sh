#!/usr/bin/env bash
# run_c4_imnet_pbs.sh <GPU>  -  ImageNet ResNet-50 baseline and LEAN lam40 (both terms): per-channel HIGH/MID and PBS 1% code,
#   then the per-channel true damage for all four models once the two single-term checkpoints exist (waits for them).
#   Log: sensreg_results/c4/imnet_pbs.log
PY=$(command -v python 2>/dev/null); [ -z "$PY" ] && PY=python
export CUDA_VISIBLE_DEVICES=${1:-0}; IM=${IMAGENET_ROOT:-./imagenet-val}
D=artifacts/models/imagenet/resnet50; B=$D/model_float32.pth; L=$D/model_float32_sensreg_lam40.pth
O=sensreg_results/c4; mkdir -p $O; LOG=$O/imnet_pbs.log; : > $LOG
{
echo "=== ImageNet ResNet-50 base and lam40, per-channel  $(date) ==="
$PY -u lean_hist_pc.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpt $B --ckpt2 $L --out $O/hist_pc_imnet_lam40 | grep -A4 "Bin occupancy\|tail_mass\|clean_acc\|ratio_high"
for X in $B:base $L:lam40; do
  echo "--- PBS ${X##*:}  $(date) ---"
  $PY -u lean_pbs_codec_pc.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpt ${X%%:*} --cov 0.01 --attack pbs --exclude-coded 1 --flips 100,200,300,500 --label imnet_${X##*:}_pc_code1x
done
echo "--- waiting for the single-term checkpoints ---"
while [ ! -f $D/ablate_tail_lam40.pth ] || [ ! -f $D/ablate_clip_lam40.pth ]; do sleep 120; done
echo "--- true single-flip damage, per-channel, four models  $(date) ---"
$PY -u lean_truedamage_pc.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpts $B $L $D/ablate_tail_lam40.pth $D/ablate_clip_lam40.pth --labels baseline lam40 tail_only clip_only --topk 1000 --nrand 1000 --out $O/truedamage_pc_imnet
echo "DONE c4_imnet_pbs $(date)"
} >> $LOG 2>&1
