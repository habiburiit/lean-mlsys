#!/usr/bin/env bash
# run_lean_archs_dw_all.sh  -  launches the 8 depthwise-coded PBS runs (2 models x base/lam5 x per-tensor/per-channel) on GPUs 0-3,
#   two per GPU, one after the other on each GPU.  Progress: tail -f sensreg_results/archs/pbs_dw_all.log
mkdir -p sensreg_results/archs
run2() { bash run_lean_archs_dw_one.sh $1 $2 $3 0; bash run_lean_archs_dw_one.sh $1 $2 $3 1; }
{
run2 0 mobilenet_v2 base &
run2 1 mobilenet_v2 lam5 &
run2 2 efficientnet_b0 base &
run2 3 efficientnet_b0 lam5 &
wait
echo "ALL DONE $(date)"
} > sensreg_results/archs/pbs_dw_all.log 2>&1
