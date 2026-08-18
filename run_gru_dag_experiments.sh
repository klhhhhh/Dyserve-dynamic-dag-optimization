#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_gru_dag_experiments.sh [normalized_jsonl] [report_directory] [checkpoint_directory]
#
# Example:
#   ./run_gru_dag_experiments.sh \
#     data/openhands_minimax_10k.jsonl \
#     reports/openhands_minimax_10k_gru \
#     checkpoints/openhands_minimax_10k_gru

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRACE_PATH="${1:-${SCRIPT_DIR}/data/openhands_minimax_10k.jsonl}"
REPORT_DIR="${2:-${SCRIPT_DIR}/reports/openhands_minimax_10k_gru}"
CHECKPOINT_DIR="${3:-${SCRIPT_DIR}/checkpoints/openhands_minimax_10k_gru}"
PREDICTOR="${SCRIPT_DIR}/predict_dag_gru.py"

if [[ ! -f "${PREDICTOR}" ]]; then
  echo "Error: predictor not found: ${PREDICTOR}" >&2
  exit 1
fi

if [[ ! -f "${TRACE_PATH}" ]]; then
  echo "Error: trace data not found: ${TRACE_PATH}" >&2
  echo "Run prepare_swe_trace.py first, or pass the JSONL path as argument 1." >&2
  exit 1
fi

mkdir -p "${REPORT_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

HORIZONS=(1 2 3)
HISTORY_LENGTHS=(1 2 3 5 8 16)

for horizon in "${HORIZONS[@]}"; do
  for history_length in "${HISTORY_LENGTHS[@]}"; do
    experiment_name="h${horizon}_hist${history_length}"
    report="${REPORT_DIR}/${experiment_name}.json"
    checkpoint="${CHECKPOINT_DIR}/${experiment_name}.pt"
    log="${REPORT_DIR}/${experiment_name}.log"

    echo "Running horizon=${horizon}, history_length=${history_length}"

    python3 "${PREDICTOR}" "${TRACE_PATH}" \
      --history-length "${history_length}" \
      --horizon "${horizon}" \
      --top-k 4 \
      --test-ratio 0.2 \
      --validation-ratio 0.15 \
      --confidence 0.6 \
      --seed 42 \
      --report-out "${report}" \
      --checkpoint-out "${checkpoint}" \
      > "${log}" 2>&1

    echo "Finished: ${experiment_name}"
    echo "  Report:     ${report}"
    echo "  Checkpoint: ${checkpoint}"
    echo "  Log:        ${log}"
  done
done

python3 - "${REPORT_DIR}" <<'PY'
import json
import re
import sys
from pathlib import Path

report_dir = Path(sys.argv[1])
rows = []

pattern = re.compile(r"^h(?P<horizon>\d+)_hist(?P<history_length>\d+)$")

for path in report_dir.glob("h*_hist*.json"):
    match = pattern.match(path.stem)
    if match is None:
        continue

    report = json.loads(path.read_text(encoding="utf-8"))
    config = report["config"]
    model = report["model"]
    baseline = report["context_free_baseline"]
    gate = model["compile_gate"]

    depth_metrics = model.get("depth_metrics", [])
    depth1 = depth_metrics[0] if depth_metrics else {}

    rows.append(
        {
            "horizon": config["horizon"],
            "history_length": config["history_length"],
            "samples": model["samples"],
            "next_accuracy": depth1.get("accuracy", 0.0),
            "next_macro_f1": depth1.get("macro_f1", 0.0),
            "transition_accuracy": model.get(
                "next_node_transition_accuracy", 0.0
            ),
            "exact_sequence": model.get("exact_sequence_at_1", 0.0),
            "mean_correct_prefix": model.get(
                "mean_correct_prefix_nodes", 0.0
            ),
            "top_k_coverage": model.get(
                "sequence_coverage_at_4", 0.0
            ),
            "gate_trigger_rate": gate.get("trigger_rate", 0.0),
            "gate_exact_precision": gate.get(
                "exact_precision_when_triggered", 0.0
            ),
            "gate_safe_prefix": gate.get(
                "mean_safe_prefix_when_triggered", 0.0
            ),
            "baseline_exact_sequence": baseline.get(
                "exact_sequence_at_1", 0.0
            ),
        }
    )

rows.sort(
    key=lambda row: (
        row["horizon"],
        row["history_length"],
    )
)

summary_path = report_dir / "summary.json"
summary_path.write_text(
    json.dumps(rows, indent=2) + "\n",
    encoding="utf-8",
)

header = (
    "h  hist  next_acc  macro_f1  transition  exact_seq  "
    "mean_prefix  topk_cov  gate_rate  gate_precision"
)

print("\n" + header)
print("-" * len(header))

for row in rows:
    print(
        f'{row["horizon"]:<2} '
        f'{row["history_length"]:<5} '
        f'{row["next_accuracy"]:>8.3f} '
        f'{row["next_macro_f1"]:>9.3f} '
        f'{row["transition_accuracy"]:>11.3f} '
        f'{row["exact_sequence"]:>10.3f} '
        f'{row["mean_correct_prefix"]:>11.3f} '
        f'{row["top_k_coverage"]:>9.3f} '
        f'{row["gate_trigger_rate"]:>10.3f} '
        f'{row["gate_exact_precision"]:>14.3f}'
    )

print(f"\nSummary saved to: {summary_path}")
PY

echo "All GRU experiment reports saved under: ${REPORT_DIR}"
echo "All GRU checkpoints saved under: ${CHECKPOINT_DIR}"