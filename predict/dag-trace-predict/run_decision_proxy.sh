#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints reports predictions

python3 predict/dag-trace-predict/train_dag_delta.py \
  data/openhands_minimax_10k_dag.jsonl \
  --feature-mode combined \
  --horizon 3 \
  --history-length 16 \
  --min-edge-confidence 0.6 \
  --iterations 250 \
  --depth 6 \
  --thread-count 4 \
  --verbose 50 \
  --model-out checkpoints/dag_combined_h3 \
  --report-out reports/dag_combined_h3_10k.json \
  --predictions-out predictions/dag_combined_h3_10k.jsonl

python3 predict/dag-trace-predict/evaluate_ilp_proxy.py \
  predictions/dag_combined_h3_10k.jsonl \
  --max-k 3 \
  --demand-error-threshold 0.25 \
  --report-out reports/dag_combined_h3_ilp_proxy_10k.json
