#!/usr/bin/env bash

set -euo pipefail

mkdir -p checkpoints reports

python3 predict/dag-trace-predict/train_dag_delta.py \
  data/openhands_minimax_10k_dag.jsonl \
  --feature-mode sequence \
  --horizon 3 \
  --history-length 16 \
  --min-edge-confidence 0.6 \
  --iterations 250 \
  --depth 6 \
  --thread-count 4 \
  --verbose 50 \
  --model-out checkpoints/dag_sequence_h3 \
  --report-out reports/dag_sequence_h3_10k.json

python3 predict/dag-trace-predict/train_dag_delta.py \
  data/openhands_minimax_10k_dag.jsonl \
  --feature-mode graph \
  --horizon 3 \
  --history-length 16 \
  --min-edge-confidence 0.6 \
  --iterations 250 \
  --depth 6 \
  --thread-count 4 \
  --verbose 50 \
  --model-out checkpoints/dag_graph_h3 \
  --report-out reports/dag_graph_h3_10k.json

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
  --report-out reports/dag_combined_h3_10k.json