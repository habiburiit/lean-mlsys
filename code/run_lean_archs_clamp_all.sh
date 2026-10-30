#!/usr/bin/env bash
# run_lean_archs_clamp_all.sh  -  8 PBS runs with activation clamping (lean_pbs_codec_clamp.py): MobileNet-V2 and EfficientNet-B0,
#   base vs lam5, per-tensor (pc0) and per-channel (pc1) INT8, 1% code; GPUs 0-3, two runs per GPU.
#   logs: sensreg_results/archs/pbs_clamp_<arch>_<ckpt>_pc<pc>.log     status: grep -h "clamp\]\|^RESULT\|^DONE" sensreg_results/archs/pbs_clamp_*_pc*.log
IM=${IMAGENET_ROOT:-./imagenet-val}; O=sensreg_results/archs; mkdir -p $O
one() { # gpu arch ckpt pc
  local F=model_float32; [ "$3" != "base" ] && F=model_float32_sensreg_$3; local SUF=clamp; [ "$4" = "1" ] && SUF=clamp_pc
  local LOG=$O/pbs_clamp_$2_$3_pc$4.log; : > $LOG
  { echo "--- $2_$3 PBS, 1% code, activation clamp, PC=$4  $(date) ---"
    CUDA_VISIBLE_DEVICES=$1 LEAN_PC=$4 python -u lean_archs.py run lean_pbs_codec_clamp.py --dataset imagenet --arch $2 --imagenet-root $IM --ckpt artifacts/models/imagenet/$2/$F.pth --cov 0.01 --attack pbs --exclude-coded 1 --flips 10,25,50,100,200,300 --label $2_$3_${SUF}_code1x
    echo "DONE $2_$3_pc$4 $(date)"; } >> $LOG 2>&1
}
{
( one 0 mobilenet_v2 base 1;    one 0 mobilenet_v2 base 0 ) &
( one 1 mobilenet_v2 lam5 1;    one 1 mobilenet_v2 lam5 0 ) &
( one 2 efficientnet_b0 base 1; one 2 efficientnet_b0 base 0 ) &
( one 3 efficientnet_b0 lam5 1; one 3 efficientnet_b0 lam5 0 ) &
wait; echo "ALL DONE $(date)"
} > $O/pbs_clamp_all.log 2>&1
