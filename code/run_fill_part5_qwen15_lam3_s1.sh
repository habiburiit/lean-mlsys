#!/usr/bin/env bash
# run_fill_part5_qwen15_lam3_s1.sh <GPU>  -  Table 23 Qwen2-1.5B LEAN lam3 seed-1 bracket: AttentionBreaker search (seed 1, 0.5% code)
#   then the six-task harness before/after (same as run_lean_llm_full_s1.sh but for the lam3_sum checkpoint).
#   Log: sensreg_results/fill_part5_qwen15_lam3_s1.log      watch: tail -f sensreg_results/fill_part5_qwen15_lam3_s1.log
export CUDA_VISIBLE_DEVICES=${1:-0}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=sensreg_results/fill_part5_qwen15_lam3_s1.log; mkdir -p sensreg_results results_llm; : > $LOG
{
ck=lam3_sum; TAG=qwen2_1.5b
MP=$(python lean_llm_export.py --base Qwen/Qwen2-1.5B --ckpt ckpt_llm/$TAG/$ck) || { echo "export failed"; exit 1; }
echo "=== $TAG $ck seed 1 cov 0.005 $(date) ==="
[ -f results_llm/showcase_${TAG}_${ck}_s1.json ] || python -u lean_llm_showcase.py --model $MP --cov 0.005 --seed 1 --tag ${TAG}_${ck}_s1 --out results_llm/showcase
python -u lean_llm_harness.py --model $MP --flips results_llm/showcase_${TAG}_${ck}_s1.json --tag ${TAG}_${ck}_s1 --out results_llm/harness
echo "DONE part5 $(date)"
} >> $LOG 2>&1
