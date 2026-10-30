#!/usr/bin/env bash
# run_llama8b_s1.sh <GPU> <ckpt: lam0_1ep|lam10_sum> <placement: attacker|gradw>
#   Second attacker seed (s1) for the Llama-3.1-8B Table 11 rows.  Only change versus run_llama8b_attsel_attack.sh
#   and the original rows: --seed 1.  Same checkpoints, per-tensor INT8, 0.5% code, blind attacker, loss threshold 7.0,
#   same population / generations / patience (the search stops on its own budget, which is what gave 39,648 for s0).
#   placement attacker = code placed by the attacker's sign-flip-damage ranking; gradw = |g| x |w| (the original rows).
#   Then the six-task harness before and after (MMLU is listed; the harness skips it in this env, as in the s0 runs).
#   Log: sensreg_results/llm/full_llama31_8b_s1_<ckpt>_<placement>.log   watch: tail -f <log>
#   Result line: grep -h "^RESULT" sensreg_results/llm/full_llama31_8b_s1_*.log
export CUDA_VISIBLE_DEVICES=${1:-0}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$(command -v python 2>/dev/null); [ -z "$PY" ] && PY=python
TAG=llama31_8b; COV=0.005; CK=${2:?ckpt}; PL=${3:?placement}; SEED=1
MP=ckpt_llm/$TAG/$CK; [ -f $MP/config.json ] || { echo "missing $MP"; exit 1; }
O=sensreg_results/llm; mkdir -p $O results_llm; LOG=$O/full_${TAG}_s1_${CK}_${PL}.log; : > $LOG
RT=${TAG}_${CK}_${PL}_s1
{
echo "=== $TAG $CK  per-tensor INT8  cov $COV  placement $PL  attacker seed $SEED  $(date) ==="
T0=$(date +%s)
[ -f results_llm/showcase_${RT}.json ] || $PY -u lean_llm_bf16.py lean_llm_showcase_codesel.py --model $MP --cov $COV --seed $SEED --code-score $PL --tag $RT --out results_llm/showcase
T1=$(date +%s); echo "--- attack wall-clock: $((T1-T0)) s ---"
echo "--- harness: six tasks + MMLU, before and after  $(date) ---"
$PY -u lean_llm_bf16.py lean_llm_harness.py --model $MP --flips results_llm/showcase_${RT}.json --tag $RT --out results_llm/harness --bs 8 --tasks arc_easy,arc_challenge,boolq,hellaswag,piqa,openbookqa,mmlu
T2=$(date +%s); echo "--- harness wall-clock: $((T2-T1)) s; total $((T2-T0)) s ---"
echo "DONE $RT $(date)"
} >> $LOG 2>&1
