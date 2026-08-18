#!/usr/bin/env python3
"""Stream Open-SWE-Traces from Hugging Face and create DAG node JSONL.

Example:
  python3 prepare_swe_trace.py \
    --config openhands --split minimax_m25 \
    --output openhands_minimax_semantic.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from predict_dag import nodes_from_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="nvidia/Open-SWE-Traces",
        help="Hugging Face dataset repository",
    )
    parser.add_argument(
        "--config", choices=("openhands", "sweagent"), default="openhands"
    )
    parser.add_argument(
        "--split", choices=("minimax_m25", "qwen35_122b"), default="minimax_m25"
    )
    parser.add_argument("--output", required=True, help="Output normalized JSONL")
    parser.add_argument(
        "--taxonomy", choices=("semantic", "tool"), default="semantic"
    )
    parser.add_argument("--max-rows", type=int, help="Optional smoke-test limit")
    parser.add_argument(
        "--resolved",
        choices=("all", "yes", "no", "known"),
        default="all",
        help="Filter by task outcome",
    )
    parser.add_argument("--language", help="Optional language filter, e.g. python")
    parser.add_argument(
        "--min-actions", type=int, default=1, help="Exclude shorter trajectories"
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install it with: python3 -m pip install datasets"
        ) from exc

    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")

    stream = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = seen = rejected = 0
    node_counts = Counter()
    instance_ids = set()
    with output.open("w", encoding="utf-8") as sink:
        for row in stream:
            seen += 1
            resolved = row.get("resolved", -1)
            if args.resolved == "yes" and resolved != 1:
                continue
            if args.resolved == "no" and resolved != 0:
                continue
            if args.resolved == "known" and resolved not in (0, 1):
                continue
            if args.language and str(row.get("language", "")).lower() != args.language.lower():
                continue

            nodes = nodes_from_trajectory(row.get("trajectory", []), args.taxonomy)
            if len(nodes) - 1 < args.min_actions:
                rejected += 1
                continue
            item = {
                "instance_id": str(row.get("instance_id", f"unknown-{seen}")),
                "run_id": str(row.get("trajectory_id", f"run-{seen}")),
                "nodes": nodes,
                "resolved": resolved,
                "repo": row.get("repo"),
                "language": row.get("language"),
                "category": (row.get("metadata") or {}).get("category")
                if isinstance(row.get("metadata"), dict)
                else None,
                "agent": args.config,
                "teacher": args.split,
            }
            sink.write(json.dumps(item, ensure_ascii=False) + "\n")
            node_counts.update(nodes)
            instance_ids.add(item["instance_id"])
            written += 1
            if args.max_rows is not None and written >= args.max_rows:
                break

    summary = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "taxonomy": args.taxonomy,
        "rows_seen": seen,
        "rows_written": written,
        "rows_rejected_as_short": rejected,
        "unique_instances": len(instance_ids),
        "node_counts": dict(node_counts.most_common()),
        "output": str(output.resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
