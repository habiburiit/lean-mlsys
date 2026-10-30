#!/usr/bin/env bash
# run_lean_e6_logit.sh <GPU> <LOGIT_W>  -  E6: ImageNet ResNet-50 LEAN fine-tune with the logit-aware tail term (lean_train_logit.py),
#   then the held-out all-to-one T-BFA with the 1% code placed by loss sensitivity (paper protocol) and by logit sensitivity,
#   and the untargeted PBS with the 1% code.  Checkpoint: .../model_float32_sensreg_lam40_logitW<W>.pth
#   Log: sensreg_results/e6_logit_w<W>.log
# usage:  nohup bash run_lean_e6_logit.sh 2 1.0 > /dev/null 2>&1 &     nohup bash run_lean_e6_logit.sh 3 0.3 > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=$1; export LOGIT_W=$2; export LOGIT_K=${LOGIT_K:-4}; IM=${IMAGENET_ROOT:-./imagenet-val}
D=artifacts/models/imagenet/resnet50; CK=$D/model_float32_sensreg_lam40_logitW$2.pth; O=sensreg_results; LOG=$O/e6_logit_w$2.log; : > $LOG
{
echo "=== E6 logit-aware LEAN, ResNet-50 ImageNet, lam 40, W=$2 K=$LOGIT_K  $(date) ==="
[ -f $CK ] || python -u lean_train_logit.py --dataset imagenet --arch resnet50 --imagenet-root $IM --init $D/model_float32.pth --reg both --lam 40 --lr 0.0005 --bs 32 --epochs 10 --out $CK
[ -f $CK ] || { echo "training failed"; exit 1; }
echo "--- T-BFA all-to-one, loss-ranked code ---"
python -u lean_tbfa_all.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpt $CK --cov 0.01 --rank loss --targets 1,100,500 --n-flips 12 --report 2,4,6,8,10,12 --label r50_logitW$2_loss
echo "--- T-BFA all-to-one, logit-ranked code ---"
python -u lean_tbfa_all.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpt $CK --cov 0.01 --rank logit --targets 1,100,500 --n-flips 20 --report 4,6,8,10,12,15,20 --label r50_logitW$2_logit
echo "--- untargeted PBS, 1% code ---"
python -u lean_pbs_codec.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpt $CK --cov 0.01 --attack pbs --exclude-coded 1 --flips 100,200,300,500 --label r50_logitW$2_pbs
echo "DONE e6 W=$2 $(date)"
} >> $LOG 2>&1
