#!/usr/bin/env python3
"""Stream raw Open-SWE-Traces rows to JSONL without discarding trajectory data.

Example:
  python download_raw_swe_trace.py \
    --config openhands --split minimax_m25 --max-rows 10000 \
    --output data/raw_openhands_minimax_10k.jsonl
"""

import argparse
import json
from pathlib import Path


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nvidia/Open-SWE-Traces")
    parser.add_argument("--config", default="openhands",
                        help="Dataset configuration, normally openhands or sweagent")
    parser.add_argument("--split", default="minimax_m25",
                        help="Dataset split, for example minimax_m25")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int,
                        help="Maximum number of rows written after filtering")
    parser.add_argument("--resolved", choices=("all", "yes", "no", "known"),
                        default="all")
    parser.add_argument("--language", help="Optional language filter, e.g. python")
    parser.add_argument("--start-row", type=int, default=0,
                        help="Skip this many source rows before filtering")
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    if args.start_row < 0:
        parser.error("--start-row must be non-negative")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install the Hugging Face datasets package first: "
            "python3 -m pip install datasets"
        ) from exc

    stream = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = skipped = filtered = written = 0
    with output.open("w", encoding="utf-8") as sink:
        for row in stream:
            seen += 1
            if seen <= args.start_row:
                skipped += 1
                continue
            resolved = row.get("resolved", -1)
            if args.resolved == "yes" and resolved != 1:
                filtered += 1
                continue
            if args.resolved == "no" and resolved != 0:
                filtered += 1
                continue
            if args.resolved == "known" and resolved not in (0, 1):
                filtered += 1
                continue
            if (args.language and
                    str(row.get("language", "")).lower() != args.language.lower()):
                filtered += 1
                continue
            trajectory = row.get("trajectory")
            if not isinstance(trajectory, list) or not trajectory:
                filtered += 1
                continue

            # Keep fields needed for graph inference and later grouped splits.
            raw = {
                "instance_id": str(row.get("instance_id", f"unknown-{seen}")),
                "trajectory_id": str(row.get("trajectory_id", f"run-{seen}")),
                "repo": row.get("repo"),
                "language": row.get("language"),
                "license": row.get("license"),
                "resolved": resolved,
                "metadata": row.get("metadata"),
                "hf_dataset_name": row.get("hf_dataset_name"),
                "trajectory": trajectory,
            }
            sink.write(json.dumps(
                raw, ensure_ascii=False, default=json_default
            ) + "\n")
            written += 1
            if args.max_rows is not None and written >= args.max_rows:
                break

    print(json.dumps({
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "source_rows_seen": seen,
        "source_rows_skipped": skipped,
        "rows_filtered": filtered,
        "rows_written": written,
        "output": str(output.resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
