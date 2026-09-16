#!/usr/bin/env python3
"""Validate canonical runtime_event.jsonl and report field completeness."""

import argparse
import json
from pathlib import Path

from runtime_analysis.schema import read_events, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events_jsonl")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--require-valid-fraction", type=float, default=0.95)
    parser.add_argument("--require-ground-truth-ready", action="store_true")
    args = parser.parse_args()
    report = validate_dataset(read_events(args.events_jsonl))
    report["passes_requested_threshold"] = (
        report["valid_event_fraction"] >= args.require_valid_fraction
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["passes_requested_threshold"]:
        raise SystemExit(2)
    if args.require_ground_truth_ready and not report["runtime_ground_truth_ready"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
