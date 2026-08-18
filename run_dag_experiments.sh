#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_dag_experiments.sh [normalized_jsonl] [report_directory]
#
# Example:
#   ./run_dag_experiments.sh \
#     data/openhands_minimax_1k.jsonl \
#     reports/openhands_minimax_1k

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_PATH="${1:-${SCRIPT_DIR}/data/openhands_minimax_10k.jsonl}"
REPORT_DIR="${2:-${SCRIPT_DIR}/reports/openhands_minimax_10k_ngrams}"
PREDICTOR="${SCRIPT_DIR}/predict_dag.py"

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

HORIZONS=(1 2 3)
WINDOWS=(0 1 2 3 5 8)

for horizon in "${HORIZONS[@]}"; do
  for window in "${WINDOWS[@]}"; do
    report="${REPORT_DIR}/h${horizon}_w${window}.json"
    echo "Running horizon=${horizon}, window=${window}"
    python3 "${PREDICTOR}" "${TRACE_PATH}" \
      --taxonomy semantic \
      --window "${window}" \
      --horizon "${horizon}" \
      --top-k 4 \
      --test-ratio 0.2 \
      --confidence 0.6 \
      --seed 42 \
      --report-out "${report}" \
      > "${REPORT_DIR}/h${horizon}_w${window}.log"
  done
done

python3 - "${REPORT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

report_dir = Path(sys.argv[1])
rows = []
for path in sorted(report_dir.glob("h*_w*.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    config = report["config"]
    model = report["model"]
    baseline = report["context_free_baseline"]
    depth1 = model["depth_metrics"][0]
    gate = model["compile_gate"]
    rows.append(
        {
            "horizon": config["horizon"],
            "window": config["window"],
            "next_accuracy": depth1["accuracy"],
            "next_macro_f1": depth1["macro_f1"],
            "transition_accuracy": model["next_node_transition_accuracy"],
            "exact_sequence": model["exact_sequence_at_1"],
            "top_k_coverage": model.get("sequence_coverage_at_4", 0.0),
            "gate_trigger_rate": gate["trigger_rate"],
            "gate_exact_precision": gate["exact_precision_when_triggered"],
            "baseline_exact_sequence": baseline["exact_sequence_at_1"],
        }
    )

rows.sort(key=lambda row: (row["horizon"], row["window"]))
summary_path = report_dir / "summary.json"
summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

header = (
    "h  w  next_acc  macro_f1  transition  exact_seq  topk_cov  "
    "gate_rate  gate_precision"
)
print("\n" + header)
print("-" * len(header))
for row in rows:
    print(
        f'{row["horizon"]:<2} {row["window"]:<2} '
        f'{row["next_accuracy"]:>9.3f} '
        f'{row["next_macro_f1"]:>9.3f} '
        f'{row["transition_accuracy"]:>11.3f} '
        f'{row["exact_sequence"]:>10.3f} '
        f'{row["top_k_coverage"]:>9.3f} '
        f'{row["gate_trigger_rate"]:>10.3f} '
        f'{row["gate_exact_precision"]:>14.3f}'
    )
print(f"\nSummary saved to: {summary_path}")
PY

echo "All experiment reports saved under: ${REPORT_DIR}"
