#!/bin/bash
# bash check_anonymity.sh   (run inside LEAN_repo). Rewrites text files with the anonymization substitutions and
# then greps for anything left. Prints "clean" when nothing identifying remains.
cd "$(dirname "$0")"
FILES=$(grep -rIl "" code hw energy README.md 2>/dev/null)
for f in $FILES; do
  sed -i -E \
    -e 's#/nelms/UFAD/[^ "'"'"'`)}]*All_final/?#./#g' \
    -e 's#/nelms/UFAD/[^ "'"'"'`)}]*imagenet-val#./imagenet-val#g' \
    -e 's#/home/UFAD/rahaman\.habibur/miniconda3/envs/dnl-imnet/bin/python#python#g' \
    -e 's#/home/UFAD/[^ "'"'"'`)}]*#$HOME#g' -e 's#/nelms/UFAD/[^ "'"'"'`)}]*#./#g' \
    -e 's/ece-nelms-x[0-9]/server/g' \
    -e 's/\bsanai_sensreg_llm\b/lean_sensreg_llm/g; s/\bsanai_sensreg\b/lean_sensreg/g; s/\bsanai_attnbreaker\b/attnbreaker_core/g' \
    -e 's/\bsanai_pbs\b/pbs_core/g; s/\bsanai_eval_mp\b/lean_eval_mp/g; s/\bsanai_eval\b/eval_core/g; s/\bsanai_vgg\b/lean_vgg/g' \
    -e "s/SANAI's/the selective code's/g; s/SANAI/the selective code/g; s/sanai/selcode/g" \
    -e 's/ufl\.edu/anon.org/g' -e 's/rahaman\.habibur|Habibur Rahaman|Habibur|habibur|Rahaman|rahaman|Bhunia|bhunia|Swarup|UFAD|nelms|University of Florida|AiSPY|aispy|dnl-imnet/ANON/g' "$f"
done
LEFT=$(grep -rIn -i "rahaman\|habibur\|bhunia\|swarup\|ufad\|nelms\|sanai\|aispy\|dnl-imnet\|florida\|ufl\.edu\|habiburiit\|gainesville" code hw energy README.md SUPPLEMENTARY.tex 2>/dev/null | grep -v "Piece-wise")
if [ -z "$LEFT" ]; then echo clean; else echo "$LEFT"; fi
