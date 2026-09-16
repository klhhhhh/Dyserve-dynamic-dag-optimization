#!/usr/bin/env python3
"""Evaluate selective triggering from exported DAG predictions.

This remains solver-independent. A trigger is considered correct only under the
existing decision-proxy definition, and is never called an actual ILP hit.
"""

import argparse
import json
from pathlib import Path

from runtime_analysis.proxy_metrics import DEFAULT_DEMAND_WEIGHTS, demand_error


def load(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def topology_confidence(row):
    probabilities = row.get("topology_probabilities", {})
    return max(probabilities.values(), default=0.0)


def demand_confidence(row):
    scores = list(row.get("presence_scores", {}).values())
    if not scores:
        return 0.0
    return sum(abs(score - 0.5) * 2 for score in scores) / len(scores)


def safe_div(a, b):
    return a / b if b else 0.0


def evaluate(rows, topology_threshold, demand_threshold, error_threshold, k):
    triggered = [row for row in rows
                 if topology_confidence(row) >= topology_threshold
                 and demand_confidence(row) >= demand_threshold]
    hits = [
        row["true_topology"] in row["topology_ranking"][:k]
        and demand_error(row, DEFAULT_DEMAND_WEIGHTS) <= error_threshold
        for row in triggered
    ]
    structural = {"branch", "join", "branch_join"}
    triggered_structural = [row for row in triggered
                            if row["true_topology"] in structural]
    structural_hits = [
        row["true_topology"] in row["topology_ranking"][:k]
        and demand_error(row, DEFAULT_DEMAND_WEIGHTS) <= error_threshold
        for row in triggered_structural
    ]
    return {
        "topology_confidence_threshold": topology_threshold,
        "demand_confidence_threshold": demand_threshold,
        "triggered": len(triggered),
        "trigger_coverage": safe_div(len(triggered), len(rows)),
        "decision_proxy_precision": safe_div(sum(hits), len(hits)),
        "false_speculation_rate": 1 - safe_div(sum(hits), len(hits)) if hits else 0,
        "structural_triggered": len(triggered_structural),
        "structural_decision_proxy_precision": safe_div(
            sum(structural_hits), len(structural_hits)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_jsonl")
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--topology-thresholds", default="0,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--demand-thresholds", default="0,0.25,0.5,0.75")
    parser.add_argument("--demand-error-threshold", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    rows = load(args.predictions_jsonl)
    topology_thresholds = [float(value) for value in args.topology_thresholds.split(",")]
    demand_thresholds = [float(value) for value in args.demand_thresholds.split(",")]
    grid = [evaluate(rows, t, d, args.demand_error_threshold, args.top_k)
            for t in topology_thresholds for d in demand_thresholds]
    report = {
        "metric_status": "selective_decision_proxy_not_actual_ilp_hit",
        "samples": len(rows),
        "top_k": args.top_k,
        "demand_error_threshold": args.demand_error_threshold,
        "grid": grid,
        "recommended_operating_points": sorted(
            (row for row in grid if row["triggered"]),
            key=lambda row: (row["decision_proxy_precision"], row["trigger_coverage"]),
            reverse=True,
        )[:5],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["recommended_operating_points"], indent=2))


if __name__ == "__main__":
    main()
