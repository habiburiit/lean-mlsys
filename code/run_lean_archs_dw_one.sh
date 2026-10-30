#!/usr/bin/env bash
# run_lean_archs_dw_one.sh <GPU> <arch> <ckpt: base|lamN> <pc: 0|1>  -  ONE PBS run of the depthwise-coded protocol (lean_pbs_codec_dw.py)
#   so the four runs of a model can go on four GPUs at once.  Log: sensreg_results/archs/pbs_dw_<arch>_<ckpt>_pc<pc>.log
#   example (8 runs, 4 GPUs, two per GPU):
#     for A in mobilenet_v2 efficientnet_b0; do for PC in 0 1; do ... see run_lean_archs_dw_all.sh
export CUDA_VISIBLE_DEVICES=$1; A=$2; CK=$3; PC=${4:-0}
O=sensreg_results/archs; mkdir -p $O; LOG=$O/pbs_dw_${A}_${CK}_pc$PC.log; : > $LOG; IM=${IMAGENET_ROOT:-./imagenet-val}
D=artifacts/models/imagenet/$A; F=model_float32; [ "$CK" != "base" ] && F=model_float32_sensreg_$CK
SUF=dw; [ "$PC" = "1" ] && SUF=dw_pc
{
[ -f $D/$F.pth ] || { echo "missing $D/$F.pth"; exit 1; }
echo "--- ${A}_${CK} PBS, 1% code + depthwise coded, PC=$PC  $(date) ---"
LEAN_PC=$PC python -u lean_archs.py run lean_pbs_codec_dw.py --dataset imagenet --arch $A --imagenet-root $IM --ckpt $D/$F.pth --cov 0.01 --attack pbs --exclude-coded 1 --flips 10,25,50,100,200,300 --label ${A}_${CK}_${SUF}_code1x
echo "DONE ${A}_${CK}_pc$PC $(date)"
} >> $LOG 2>&1
