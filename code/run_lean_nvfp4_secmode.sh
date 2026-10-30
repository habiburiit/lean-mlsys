#!/usr/bin/env bash
# run_lean_nvfp4_secmode.sh <GPU>  -  NVFP4 security mode = the paper's low-bit recipe applied to NVFP4:
#   block scales under the code (0.5 b/w) + long interleaved BCH over the sign plane (t=19, 0.19 b/w, a sign flip costs the
#   attacker 20 flips) + damage-ranked 1% code.  Control vs LEAN v3 (lam 100 and 200), ResNet-18 CIFAR-10, two attack seeds.
#   Uses lean_pbs_nvfp4s.py (derived copy with --sign-cost); existing scripts unchanged.
#   Add ImageNet with:  IMNET=1 bash run_lean_nvfp4_secmode.sh <GPU>
# usage:  nohup bash run_lean_nvfp4_secmode.sh 1 > /dev/null 2>&1 &    tail -f sensreg_results/nvfp4/secmode.log
export CUDA_VISIBLE_DEVICES=${1:-0}
O=sensreg_results/nvfp4; mkdir -p $O; LOG=$O/secmode.log; : > $LOG
COMMON="--flips 10,25,50,100,200,300,500 --topk 50 --attack-batches 2 --probe-batches 8 --protect selcode --cov 0.01 --protect-scales on --sign-cost 20"
{
echo "=== NVFP4 security mode (scales + sign plane t=19 + 1% code)  $(date) ==="
D=artifacts/models/cifar10/resnet18/nvfp4
for S in 0 1; do
  for ck in lam0 v3_lam100 v3_lam200; do
    [ -f $D/$ck.pth ] || { echo "missing $D/$ck.pth"; continue; }
    echo "--- cifar10 resnet18 $ck seed $S ---"
    python -u lean_pbs_nvfp4s.py --dataset cifar10 --arch resnet18 --ckpt $D/$ck.pth $COMMON --seed $S --label c10_${ck}_sec_s$S
  done
done
if [ "${IMNET:-0}" = 1 ]; then
  D=artifacts/models/imagenet/resnet50/nvfp4; IM=${IMAGENET_ROOT:-./imagenet-val}
  for ck in lam0 v3_lam200; do
    echo "--- imagenet resnet50 $ck seed 0 ---"
    python -u lean_pbs_nvfp4s.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpt $D/$ck.pth $COMMON --seed 0 --label imnet_${ck}_sec_s0
  done
fi
echo "DONE secmode $(date)"
} >> $LOG 2>&1
