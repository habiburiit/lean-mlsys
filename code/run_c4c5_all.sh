#!/usr/bin/env bash
# run_c4c5_all.sh  -  launches everything for C4 and C5 on GPUs 0-3 of an empty server:
#   GPU0: ImageNet P_clip-only fine-tune + its per-channel PBS      GPU1: ImageNet P_tail-only fine-tune + its per-channel PBS
#   GPU2: ImageNet base and lam40 per-channel PBS, then true damage  GPU3: CIFAR-10 C4 (per-channel) then C5 (code-aware sweep)
#   status: grep -h "^RESULT\|^DONE\|^ep \|HIGH\|worst 100" sensreg_results/c4/*.log sensreg_results/c5/*.log
mkdir -p sensreg_results/c4 sensreg_results/c5
nohup bash run_c4_imnet_train.sh 0 clip > /dev/null 2>&1 &
nohup bash run_c4_imnet_train.sh 1 tail > /dev/null 2>&1 &
nohup bash run_c4_imnet_pbs.sh 2 > /dev/null 2>&1 &
nohup bash -c "bash run_c4_cifar_pc.sh 3; bash run_c5_codeaware.sh 3" > /dev/null 2>&1 &
echo "launched 4 jobs $(date)"
