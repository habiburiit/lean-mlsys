#!/usr/bin/env bash
# run_lean_harness.sh <GPU>  -  LM-Eval-Harness tasks (ARC-E/C, BoolQ, HellaSwag, PIQA, OpenBookQA, SocialIQA) before/after the
#   saved AttentionBreaker flip lists, Qwen2-0.5B control (12 flips) and LEAN lam10 (1372 flips).
#   Log: sensreg_results/llm/harness_qwen2_0.5b.log
export CUDA_VISIBLE_DEVICES=$1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export HF_ALLOW_CODE_EVAL=1
O=sensreg_results/llm; mkdir -p $O results_llm; LOG=$O/harness_qwen2_0.5b.log; : > $LOG
{
M0=$(python lean_llm_export.py --base Qwen/Qwen2-0.5B --ckpt ckpt_llm/qwen2_0.5b/lam0_1ep)
M1=$(python lean_llm_export.py --base Qwen/Qwen2-0.5B --ckpt ckpt_llm/qwen2_0.5b/lam10_sum)
echo "--- control lam0_1ep, 12 flips ---"
python -u lean_llm_harness.py --model $M0 --flips results_llm/showcase_qwen2_0.5b_lam0_1ep.json --tag q05_control --out results_llm/harness
echo "--- LEAN lam10_sum, 1372 flips ---"
python -u lean_llm_harness.py --model $M1 --flips results_llm/showcase_qwen2_0.5b_lam10_sum_genbfa.json --tag q05_lean10 --out results_llm/harness
echo "DONE $(date)"
} >> $LOG 2>&1
