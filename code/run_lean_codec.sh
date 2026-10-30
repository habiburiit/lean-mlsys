#!/usr/bin/env bash
# run_lean_codec.sh <GPU>  -  item 2: PBS with the real BCH codec in the loop + codeword-aware attacker.
#   ResNet-18 CIFAR-10 (base vs lam40) then ResNet-50 ImageNet (base vs lam40), 1% code, K=4 planes, t=12, keyed interleave.
#   Uses lean_pbs_codec.py (new).  log: sensreg_results/codec/codec.log
# usage:  nohup bash run_lean_codec.sh 1 > /dev/null 2>&1 &    tail -f sensreg_results/codec/codec.log
export CUDA_VISIBLE_DEVICES=${1:-0}
O=sensreg_results/codec; mkdir -p $O; LOG=$O/codec.log; : > $LOG
IM=${IMAGENET_ROOT:-./imagenet-val}
{
echo "=== real-codec attacks  $(date) ==="
for spec in "cifar10 resnet18 40 10,25,50,100,200,300,500" "imagenet resnet50 40 10,25,50,100,200,300"; do
  set -- $spec; DS=$1; A=$2; L=$3; FL=$4; D=artifacts/models/$DS/$A
  for ck in model_float32:base model_float32_sensreg_lam$L:lam$L; do
    F=${ck%%:*}; LAB=${DS}_${A}_${ck##*:}
    echo "--- $LAB pbs (adaptive, real decoder) ---"
    python -u lean_pbs_codec.py --dataset $DS --arch $A --imagenet-root $IM --ckpt $D/$F.pth --cov 0.01 --attack pbs --flips $FL --label $LAB
    echo "--- $LAB codeword-aware, layout known ---"
    python -u lean_pbs_codec.py --dataset $DS --arch $A --imagenet-root $IM --ckpt $D/$F.pth --cov 0.01 --attack cwaware --attacker-knows-key 1 --label $LAB
    echo "--- $LAB codeword-aware, layout unknown (keyed interleave) ---"
    python -u lean_pbs_codec.py --dataset $DS --arch $A --imagenet-root $IM --ckpt $D/$F.pth --cov 0.01 --attack cwaware --attacker-knows-key 0 --label $LAB
  done
done
echo "DONE codec $(date)"
} >> $LOG 2>&1
