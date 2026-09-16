#!/usr/bin/env python3
"""Convert OpenHands output JSONL into flat canonical runtime events.

The converter is deliberately conservative: missing runtime DAG parents remain
empty instead of being invented. Raw records are retained in metadata so a new
OpenHands schema can be audited without re-running the model.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from runtime_analysis.schema import make_event, validate_dataset


def nested_first(obj: Any, names: set[str]) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in names and value not in (None, ""):
                return value
        for value in obj.values():
            found = nested_first(value, names)
            if found not in (None, ""):
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = nested_first(value, names)
            if found not in (None, ""):
                return found
    return None


def timestamp_ms(raw: Any) -> float | None:
    value = nested_first(raw, {"timestamp", "created_at", "event_time", "time"})
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value * 1000 if value < 10_000_000_000 else value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000
    except ValueError:
        return None


def text_of(raw: Any) -> str:
    return json.dumps(raw, ensure_ascii=False, default=str).lower()


def canonical_type(raw: dict[str, Any]) -> str:
    text = text_of(raw)
    kind = str(nested_first(raw, {"event_type", "type", "kind"}) or "").lower()
    if "retry" in text:
        return "retry"
    if any(word in text for word in ("exception", "traceback", "failed")):
        return "failure"
    if "tool" in text and any(word in text for word in ("observation", "result", "response")):
        return "node_finished"
    if "tool" in text and any(word in text for word in ("action", "call", "request")):
        return "node_started"
    if any(word in kind for word in ("message", "assistant", "chat")):
        return "agent_message"
    if "token" in text or "llm" in text or "model" in text:
        return "llm_call"
    return "unknown"


def semantic_type(raw: dict[str, Any]) -> str | None:
    text = text_of(raw)
    tool = str(nested_first(raw, {"tool_name", "name"}) or "").lower()
    joined = f"{tool} {text[:3000]}"
    if any(word in joined for word in ("pytest", "test", "unittest")):
        return "test"
    if any(word in joined for word in ("edit", "patch", "str_replace", "write_file")):
        return "edit"
    if any(word in joined for word in ("bash", "shell", "terminal", "execute")):
        return "shell"
    if any(word in joined for word in ("browser", "web", "search")):
        return "web"
    if any(word in joined for word in ("read", "view", "open_file", "grep", "find")):
        return "inspect"
    if "assistant" in joined or "reason" in joined:
        return "reason"
    return None


def as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def convert_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    workflow_id = str(row.get("instance_id") or row.get("workflow_id") or "unknown")
    history = row.get("history") or row.get("events") or []
    if isinstance(history, dict):
        history = history.get("events", [])
    events = []
    for sequence, raw in enumerate(history if isinstance(history, list) else []):
        if not isinstance(raw, dict):
            raw = {"value": raw}
        event_type = canonical_type(raw)
        failed = event_type == "failure"
        tool = nested_first(raw, {"tool_name", "function_name"})
        node_id = nested_first(raw, {"node_id", "tool_call_id", "call_id"})
        if event_type in {"node_started", "node_finished", "failure", "retry", "recovery"}:
            node_id = str(node_id or f"{workflow_id}:node:{sequence}")
        parents = nested_first(raw, {"parent_ids", "parents"}) or []
        if not isinstance(parents, list):
            parents = [str(parents)]
        events.append(make_event(
            workflow_id=workflow_id,
            event_id=f"{workflow_id}:event:{sequence}",
            sequence=sequence,
            event_type=event_type,
            timestamp_ms=timestamp_ms(raw),
            node_id=node_id,
            parent_ids=[str(value) for value in parents],
            semantic_type=semantic_type(raw),
            tool_name=str(tool) if tool else None,
            status="failed" if failed else (
                "success" if event_type == "node_finished" else "unknown"
            ),
            model=(str(nested_first(raw, {"model", "model_name"}))
                   if nested_first(raw, {"model", "model_name"}) else None),
            input_tokens=as_int(nested_first(raw, {"input_tokens", "prompt_tokens"})),
            output_tokens=as_int(nested_first(raw, {"output_tokens", "completion_tokens"})),
            latency_ms=as_float(nested_first(raw, {"latency_ms", "duration_ms"})),
            cost_usd=as_float(nested_first(raw, {"cost_usd", "cost", "proxy_cost"})),
            metadata={"source_format": "openhands", "raw_event": raw},
        ))
    starts = {}
    for event in events:
        node_id = event.get("node_id")
        event_time = event.get("timestamp_ms")
        if event["event_type"] == "node_started" and node_id and event_time is not None:
            starts[node_id] = event_time
        if (event["event_type"] in {"node_finished", "failure"}
                and node_id in starts and event_time is not None
                and event.get("latency_ms") is None):
            event["latency_ms"] = max(0.0, event_time - starts[node_id])
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("--events-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    all_events = []
    with args.input_jsonl.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} is not an object")
            all_events.extend(convert_row(row))
    args.events_out.parent.mkdir(parents=True, exist_ok=True)
    with args.events_out.open("w", encoding="utf-8") as stream:
        for event in all_events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    report = validate_dataset(all_events)
    report["source"] = str(args.input_jsonl)
    report["warning"] = (
        "Empty parent_ids mean the source did not expose runtime DAG edges; "
        "the converter does not infer them."
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
