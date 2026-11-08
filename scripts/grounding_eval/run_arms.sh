#!/usr/bin/env bash
# Two-arm run (review ON vs OFF, separate extractions). Superseded by run_paired.sh:
# separate extractions differ run to run, so arm differences are confounded by nondeterminism.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.." || exit 1
Q1="结直肠癌血浆cfDNA甲基化，需要健康对照"
Q2="肝细胞癌血浆cfDNA甲基化数据，要有健康人对照"
Q3="结直肠癌组织和癌旁组织的DNA甲基化数据"
for q in "$Q1" "$Q2" "$Q3"; do
  echo "===== ARM=review-ON  QUERY=$q"
  "$HERE/run_grounding.sh" --query "$q" --top-n 15 --out-dir data/eval/arm_review || echo "FAILED: $q"
  sleep 20
done
for q in "$Q1" "$Q2" "$Q3"; do
  echo "===== ARM=review-OFF QUERY=$q"
  "$HERE/run_grounding.sh" --query "$q" --top-n 15 --no-review --out-dir data/eval/arm_noreview || echo "FAILED: $q"
  sleep 20
done
echo "ALL DONE"
