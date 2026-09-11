#!/usr/bin/env python3
"""Evaluate whether DAG predictions are useful enough to pre-run an ILP.

This is deliberately a proxy evaluator.  It does not claim that two ILP plans
are equal because the repository does not yet contain the Dyserve solver,
runtime resource snapshots, or per-node service profiles.  It combines a
Top-K topology hit with a demand-vector error so structural accuracy alone
cannot be reported as a system win.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DEMAND_WEIGHTS = {
    "inspect": 0.25,
    "edit": 1.0,
    "shell": 0.75,
    "test": 0.5,
    "reason": 1.25,
    "web": 0.5,
    "other": 0.5,
}
DEFAULT_TOPOLOGY_WEIGHTS = {
    "none": 0.5,
    "independent": 1.0,
    "chain": 1.25,
    "branch": 2.0,
    "join": 2.0,
    "branch_join": 2.5,
}
STRUCTURAL_CHANGE = {"branch", "join", "branch_join"}


def parse_weights(raw, defaults):
    result = dict(defaults)
    if not raw:
        return result
    for item in raw.split(","):
        name, value = item.split("=", 1)
        result[name.strip()] = float(value)
    return result


def load_rows(path):
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("Prediction JSONL is empty")
    required = {"true_topology", "topology_ranking", "true_counts",
                "predicted_counts"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing prediction fields: {sorted(missing)}")
    return rows


def demand_error(row, weights):
    """Weighted L1 error normalized by true future demand (lower is better)."""
    node_types = set(row["true_counts"]) | set(row["predicted_counts"])
    numerator = sum(
        weights.get(node_type, weights.get("other", 1.0))
        * abs(row["predicted_counts"].get(node_type, 0)
              - row["true_counts"].get(node_type, 0))
        for node_type in node_types
    )
    denominator = sum(
        weights.get(node_type, weights.get("other", 1.0))
        * row["true_counts"].get(node_type, 0)
        for node_type in node_types
    )
    return numerator / max(denominator, 1.0)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize(rows, errors, threshold, max_k, topology_weights):
    result = {"samples": len(rows)}
    if not rows:
        return result
    sample_weights = [topology_weights.get(row["true_topology"], 1.0)
                      for row in rows]
    total_weight = sum(sample_weights)
    result.update({
        "mean_weighted_demand_error": mean(errors),
        "mean_demand_similarity": mean(max(0.0, 1.0 - value)
                                       for value in errors),
        "demand_within_threshold": mean(value <= threshold for value in errors),
    })
    for k in range(1, max_k + 1):
        topology_hits = [
            row["true_topology"] in row["topology_ranking"][:k]
            for row in rows
        ]
        decision_hits = [hit and error <= threshold
                         for hit, error in zip(topology_hits, errors)]
        reuse_scores = [
            float(hit) * max(0.0, 1.0 - error)
            for hit, error in zip(topology_hits, errors)
        ]
        result[f"topology_hit_at_{k}"] = mean(topology_hits)
        result[f"decision_proxy_hit_at_{k}"] = mean(decision_hits)
        result[f"criticality_weighted_decision_proxy_hit_at_{k}"] = (
            sum(weight * hit for weight, hit in zip(sample_weights, decision_hits))
            / total_weight
        )
        result[f"proxy_plan_reuse_score_at_{k}"] = mean(reuse_scores)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_jsonl")
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--demand-error-threshold", type=float, default=0.25)
    parser.add_argument(
        "--demand-weights",
        help="Comma-separated overrides, e.g. reason=2,edit=1.5,test=1",
    )
    parser.add_argument(
        "--topology-weights",
        help="Comma-separated criticality overrides, e.g. branch=3,join=3",
    )
    args = parser.parse_args()
    if args.max_k < 1:
        parser.error("--max-k must be at least 1")
    if args.demand_error_threshold < 0:
        parser.error("--demand-error-threshold must be non-negative")

    rows = load_rows(args.predictions_jsonl)
    demand_weights = parse_weights(args.demand_weights, DEFAULT_DEMAND_WEIGHTS)
    topology_weights = parse_weights(
        args.topology_weights, DEFAULT_TOPOLOGY_WEIGHTS
    )
    errors = [demand_error(row, demand_weights) for row in rows]

    per_topology_indices = defaultdict(list)
    for i, row in enumerate(rows):
        per_topology_indices[row["true_topology"]].append(i)

    def selected(indices):
        return summarize([rows[i] for i in indices], [errors[i] for i in indices],
                         args.demand_error_threshold, args.max_k,
                         topology_weights)

    structural = [i for i, row in enumerate(rows)
                  if row["true_topology"] in STRUCTURAL_CHANGE]
    non_structural = [i for i, row in enumerate(rows)
                      if row["true_topology"] not in STRUCTURAL_CHANGE]
    full_horizon = [i for i, row in enumerate(rows)
                    if row.get("full_horizon", True)]

    report = {
        "metric_status": "proxy_not_actual_ilp_replay",
        "definition": {
            "weighted_demand_error": (
                "sum_t weight[t] * abs(pred_count[t] - true_count[t]) / "
                "max(sum_t weight[t] * true_count[t], 1)"
            ),
            "decision_proxy_hit_at_k": (
                "true topology is in Top-K AND weighted demand error is at "
                "most the configured threshold"
            ),
            "proxy_plan_reuse_score_at_k": (
                "Top-K topology hit * max(0, 1 - weighted demand error)"
            ),
            "warning": (
                "These metrics rank predictor candidates before a real solver "
                "is integrated; they are not latency, SLO, cost, objective "
                "regret, or actual ILP-plan equivalence."
            ),
        },
        "config": {
            **vars(args),
            "demand_weights_resolved": demand_weights,
            "topology_weights_resolved": topology_weights,
        },
        "topology_distribution": dict(Counter(
            row["true_topology"] for row in rows
        )),
        "overall": summarize(rows, errors, args.demand_error_threshold,
                             args.max_k, topology_weights),
        "structural_change_only": selected(structural),
        "non_structural": selected(non_structural),
        "full_horizon_only": selected(full_horizon),
        "per_topology": {
            label: selected(indices)
            for label, indices in sorted(per_topology_indices.items())
        },
        "next_real_ilp_metrics": [
            "candidate plan feasibility on the realized DAG",
            "objective regret versus oracle future-DAG plan",
            "fraction of replanning latency hidden by speculation",
            "plan reuse / serving-decision agreement",
            "E2E latency, SLO goodput, and monetary or GPU cost",
        ],
    }
    output = Path(args.report_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "metric_status": report["metric_status"],
        "overall": report["overall"],
        "structural_change_only": report["structural_change_only"],
    }, indent=2))


if __name__ == "__main__":
    main()
