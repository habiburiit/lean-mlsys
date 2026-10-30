#!/bin/bash
# runs timeloop-mapper on every ResNet-50 layer shape; logs live to runs/all.log
cd "$(dirname "$0")"
for p in probs/*.yaml; do
  n=$(basename $p .yaml); [ "$n" = layers ] && continue
  [ -f runs/$n/timeloop-mapper.stats.txt ] && { echo "[skip] $n"; continue; }
  mkdir -p runs/$n; echo "[start] $n $(date +%T)"
  (cd runs/$n && timeloop-mapper ../../arch_int8.yaml ../../constraints_rs.yaml ../../probs/$n.yaml ../../mapper.yaml > run.log 2>&1)
  echo "[done] $n $(tail -1 runs/$n/run.log)"
done
echo ALL_DONE
