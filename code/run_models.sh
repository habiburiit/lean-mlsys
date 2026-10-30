#!/bin/bash
cd "$(dirname "$0")"
for m in "$@"; do echo "##### $m $(date +%T)"; (cd models/$m && ./run_all.sh); done
echo MODELS_DONE
