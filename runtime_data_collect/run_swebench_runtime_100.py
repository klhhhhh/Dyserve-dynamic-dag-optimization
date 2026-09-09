#!/usr/bin/env python3
"""Run about 100 SWE-bench tasks with OpenHands and collect runtime traces.

This is a thin wrapper around the official OpenHands benchmarks V1
``swebench-infer`` command.  It keeps the original OpenHands output and writes
a second JSONL file with normalized event-level runtime records for analysis.

The LLM configuration (and any API key it references) is never read or copied
by this script; only its path is passed to ``swebench-infer``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        required=True,
        help="Checkout of https://github.com/OpenHands/benchmarks",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        required=True,
        help="OpenHands LLM JSON configuration file",
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Verified",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--workspace",
        choices=("docker", "remote", "apptainer"),
        default="docker",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime_data/swebench_100"),
        help="Destination for OpenHands output and normalized traces",
    )
    parser.add_argument(
        "--raw-output-jsonl",
        type=Path,
        help="Normalize an existing OpenHands output.jsonl instead of running inference",
    )
    parser.add_argument(
        "--tool-preset",
        choices=("default", "gemini", "planning"),
        default="default",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional single argument passed to swebench-infer; repeat as needed",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    return str(value)


def first_value(mapping: dict[str, Any], paths: Iterable[tuple[str, ...]]) -> Any:
    for path in paths:
        current: Any = mapping
        for key in path:
            if not isinstance(current, dict) or key not in current:
                break
            current = current[key]
        else:
            if current not in (None, ""):
                return current
    return None


def event_type(event: dict[str, Any]) -> str:
    value = first_value(
        event,
        (("event_type",), ("type",), ("kind",), ("action",), ("name",)),
    )
    if isinstance(value, dict):
        value = first_value(value, (("type",), ("kind",), ("name",)))
    return str(value or "unknown")


def tool_name(event: dict[str, Any]) -> str | None:
    value = first_value(
        event,
        (
            ("tool_name",),
            ("tool", "name"),
            ("tool_call", "name"),
            ("tool_call", "function", "name"),
            ("action", "tool_name"),
            ("action", "name"),
            ("observation", "tool_name"),
        ),
    )
    return str(value) if value not in (None, "") else None


def timestamp(event: dict[str, Any]) -> str | None:
    value = first_value(
        event,
        (
            ("timestamp",),
            ("created_at",),
            ("event_time",),
            ("time",),
        ),
    )
    return str(value) if value not in (None, "") else None


def status(event: dict[str, Any]) -> str:
    explicit = first_value(
        event,
        (("status",), ("observation", "status"), ("result", "status")),
    )
    if explicit is not None:
        return str(explicit)
    text = json.dumps(event, ensure_ascii=False).lower()
    if any(token in text for token in ("error", "failed", "exception", "traceback")):
        return "failed"
    if any(token in text for token in ("success", "completed", "finish")):
        return "success"
    return "unknown"


def event_phase(kind: str, event: dict[str, Any]) -> str:
    text = f"{kind} {json.dumps(event, ensure_ascii=False)[:1000]}".lower()
    if "tool" in text and any(word in text for word in ("result", "observation", "response")):
        return "tool_completed"
    if "tool" in text and any(word in text for word in ("call", "action", "request")):
        return "tool_started"
    if any(word in text for word in ("message", "chat", "assistant")):
        return "agent_message"
    return "other"


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    history = row.get("history") or []
    if not isinstance(history, list):
        history = []
    normalized_events = []
    for sequence, raw in enumerate(history):
        raw_event = jsonable(raw)
        if not isinstance(raw_event, dict):
            raw_event = {"value": raw_event}
        kind = event_type(raw_event)
        normalized_events.append(
            {
                "sequence": sequence,
                "timestamp": timestamp(raw_event),
                "event_type": kind,
                "phase": event_phase(kind, raw_event),
                "tool_name": tool_name(raw_event),
                "status": status(raw_event),
                "raw_event": raw_event,
            }
        )

    test_result = row.get("test_result") or {}
    if not isinstance(test_result, dict):
        test_result = {"value": test_result}
    return {
        "instance_id": str(row.get("instance_id", "unknown")),
        "attempt": row.get("attempt"),
        "instruction": row.get("instruction"),
        "error": jsonable(row.get("error")),
        "metrics": jsonable(row.get("metrics") or {}),
        "event_count": len(normalized_events),
        "tool_event_count": sum(
            event["phase"].startswith("tool_") for event in normalized_events
        ),
        "failed_event_count": sum(
            event["status"] == "failed" for event in normalized_events
        ),
        "patch_chars": len(str(test_result.get("git_patch", ""))),
        "events": normalized_events,
    }


def normalize_jsonl(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = events = failed_rows = 0
    with source.open(encoding="utf-8") as input_stream, destination.open(
        "w", encoding="utf-8"
    ) as output_stream:
        for line_number, line in enumerate(input_stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {source}:{line_number}: {error}") from error
            normalized = normalize_row(row)
            output_stream.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            rows += 1
            events += normalized["event_count"]
            failed_rows += int(normalized["error"] is not None)
    return {
        "instances": rows,
        "events": events,
        "instances_with_runner_error": failed_rows,
        "source": str(source.resolve()),
        "normalized_output": str(destination.resolve()),
    }


def find_output_jsonl(root: Path, started_at: float) -> Path:
    candidates = [
        path
        for path in root.rglob("*.jsonl")
        if path.is_file()
        and path.stat().st_mtime >= started_at - 2
        and path.name not in {"runtime_events.jsonl"}
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No new OpenHands JSONL output found below {root}. "
            "Check the swebench-infer log for its output_json path."
        )
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def run_inference(args: argparse.Namespace) -> Path:
    benchmarks_dir = args.benchmarks_dir.expanduser().resolve()
    llm_config = args.llm_config.expanduser().resolve()
    if not (benchmarks_dir / "pyproject.toml").is_file():
        raise FileNotFoundError(
            f"{benchmarks_dir} is not an OpenHands benchmarks checkout"
        )
    if not llm_config.is_file():
        raise FileNotFoundError(f"LLM config not found: {llm_config}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "uv",
        "run",
        "swebench-infer",
        str(llm_config),
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--workspace",
        args.workspace,
        "--num-workers",
        str(args.workers),
        "--max-iterations",
        str(args.max_iterations),
        "--n-limit",
        str(args.limit),
        "--tool-preset",
        args.tool_preset,
        "--output-dir",
        str(output_dir),
        *args.extra_arg,
    ]
    started_at = datetime.now().timestamp()
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=benchmarks_dir, check=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return find_output_jsonl(output_dir, started_at)


def main() -> None:
    args = parse_args()
    if args.raw_output_jsonl:
        raw_output = args.raw_output_jsonl.expanduser().resolve()
        if not raw_output.is_file():
            raise FileNotFoundError(f"Raw output not found: {raw_output}")
    else:
        raw_output = run_inference(args)

    destination = args.output_dir.expanduser().resolve() / "runtime_events.jsonl"
    summary = normalize_jsonl(raw_output, destination)
    summary.update(
        {
            "dataset": args.dataset,
            "split": args.split,
            "requested_limit": args.limit,
            "workspace": args.workspace,
            "max_iterations": args.max_iterations,
        }
    )
    summary_path = destination.with_name("runtime_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
