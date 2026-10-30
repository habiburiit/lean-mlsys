#!/usr/bin/env bash
# run_lean_fill_blanks.sh <GPU>  -  fills the blank table cells (new runs only; existing scripts unchanged):
#   1. real BCH codec, CIFAR-10 ResNet-18, codeword-aware attacker with the key, 5 and 20 overloaded codewords (65 and 260 errors)
#   2. all-to-one T-BFA, ImageNet ResNet-18 held-out, 1% code by loss, reported at 4, 8, 10, 12, 20, 30 flips (fills the 4/8/12 cells)
#   3. AttentionBreaker fixed-budget rows for Llama-3.2-1B (20, 50, 100 of the attacker's best flips), control and LEAN lam10
# log: sensreg_results/fill_blanks.log      tail -f sensreg_results/fill_blanks.log
export CUDA_VISIBLE_DEVICES=${1:-0}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=sensreg_results/fill_blanks.log; mkdir -p sensreg_results results_llm; : > $LOG
IM=${IMAGENET_ROOT:-./imagenet-val}
{
echo "=== 1. codec CIFAR-10 ResNet-18, 5 and 20 codewords  $(date) ==="
D=artifacts/models/cifar10/resnet18
for ck in model_float32:base model_float32_sensreg_lam40:lam40; do
  F=${ck%%:*}; LAB=cifar10_resnet18_${ck##*:}_fill
  python -u lean_pbs_codec.py --dataset cifar10 --arch resnet18 --ckpt $D/$F.pth --cov 0.01 --attack cwaware --attacker-knows-key 1 --cw-counts 5,20 --label $LAB
done
echo "=== 2. all-to-one T-BFA ImageNet ResNet-18, code by loss  $(date) ==="
D=artifacts/models/imagenet/resnet18
for CK in model_float32.pth model_float32_sensreg_lam30.pth; do
  python -u lean_tbfa_all.py --dataset imagenet --arch resnet18 --imagenet-root $IM --ckpt $D/$CK --cov 0.01 --rank ce \
      --targets 1,100,500 --n-flips 30 --report 4,8,10,12,20,30 --label resnet18_${CK%.pth}_fill
done
echo "=== 3. Llama-3.2-1B fixed-budget AttentionBreaker rows  $(date) ==="
for ck in lam0_1ep lam10_sum; do
  MP=ckpt_llm/llama32_1b/$ck; [ -f $MP/config.json ] || { echo "missing $MP"; continue; }
  for N in 20 50 100; do
    python -u lean_llm_showcase.py --model $MP --cov 0.001 --seed 0 --fixed $N --tag llama32_1b_${ck}_fixed$N --out results_llm/showcase
  done
done
echo "DONE fill_blanks $(date)"
} >> $LOG 2>&1
