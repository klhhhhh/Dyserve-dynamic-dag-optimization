#!/usr/bin/env python3
"""Canonical runtime-event schema and validation helpers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "1.0"
EVENT_TYPES = {
    "workflow_started", "workflow_finished", "node_created", "edge_added",
    "node_started", "node_finished", "branch_resolved", "failure", "retry",
    "recovery", "llm_call", "resource_snapshot", "ilp_started", "ilp_finished",
    "agent_message", "unknown",
}
NODE_EVENTS = {"node_created", "node_started", "node_finished", "failure",
               "retry", "recovery"}
STATUSES = {"pending", "running", "success", "failed", "cancelled", "unknown"}


def make_event(*, workflow_id: str, event_id: str, sequence: int,
               event_type: str, timestamp_ms: float | None = None,
               node_id: str | None = None, parent_ids: Iterable[str] = (),
               semantic_type: str | None = None, tool_name: str | None = None,
               status: str = "unknown", model: str | None = None,
               input_tokens: int | None = None,
               output_tokens: int | None = None,
               latency_ms: float | None = None, cost_usd: float | None = None,
               resource_snapshot: dict[str, Any] | None = None,
               metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": str(workflow_id),
        "event_id": str(event_id),
        "sequence": int(sequence),
        "event_type": event_type,
        "timestamp_ms": timestamp_ms,
        "node_id": node_id,
        "parent_ids": list(parent_ids),
        "semantic_type": semantic_type,
        "tool_name": tool_name,
        "status": status,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "resource_snapshot": resource_snapshot or {},
        "metadata": metadata or {},
    }


def validate_event(event: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("schema_version", "workflow_id", "event_id", "sequence",
                  "event_type", "parent_ids", "status"):
        if field not in event:
            errors.append(f"missing:{field}")
    if event.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported:schema_version")
    if event.get("event_type") not in EVENT_TYPES:
        errors.append("invalid:event_type")
    if not isinstance(event.get("sequence"), int) or event.get("sequence", -1) < 0:
        errors.append("invalid:sequence")
    if not isinstance(event.get("parent_ids"), list):
        errors.append("invalid:parent_ids")
    if event.get("status") not in STATUSES:
        errors.append("invalid:status")
    if event.get("event_type") in NODE_EVENTS and not event.get("node_id"):
        errors.append("missing:node_id")
    for field in ("timestamp_ms", "latency_ms", "cost_usd"):
        value = event.get(field)
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            errors.append(f"invalid:{field}")
    for field in ("input_tokens", "output_tokens"):
        value = event.get(field)
        if value is not None and (not isinstance(value, int) or value < 0):
            errors.append(f"invalid:{field}")
    return errors


def read_events(path: str | Path) -> list[dict[str, Any]]:
    events = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            events.append(event)
    return events


def validate_dataset(events: list[dict[str, Any]]) -> dict[str, Any]:
    errors = Counter()
    workflows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_ids = set()
    duplicate_ids = 0
    for event in events:
        for error in validate_event(event):
            errors[error] += 1
        event_id = (event.get("workflow_id"), event.get("event_id"))
        duplicate_ids += int(event_id in event_ids)
        event_ids.add(event_id)
        workflows[str(event.get("workflow_id", ""))].append(event)

    nonmonotonic_sequence = nonmonotonic_time = 0
    for rows in workflows.values():
        sequences = [row.get("sequence") for row in rows]
        nonmonotonic_sequence += int(sequences != sorted(sequences))
        times = [row["timestamp_ms"] for row in rows
                 if isinstance(row.get("timestamp_ms"), (int, float))]
        nonmonotonic_time += int(times != sorted(times))

    total = len(events)
    coverage_fields = ["timestamp_ms", "node_id", "semantic_type", "tool_name",
                       "model", "input_tokens", "output_tokens", "latency_ms",
                       "cost_usd"]
    field_coverage = {
        field: (sum(event.get(field) is not None for event in events) / total
                if total else 0)
        for field in coverage_fields
    }
    explicit_edges = sum(event.get("event_type") == "edge_added" for event in events)
    schema_valid = bool(total) and not errors and not duplicate_ids
    readiness_reasons = []
    if not schema_valid:
        readiness_reasons.append("schema_errors_or_duplicate_ids")
    if field_coverage["timestamp_ms"] < 0.95:
        readiness_reasons.append("timestamp_coverage_below_95_percent")
    if field_coverage["node_id"] < 0.80:
        readiness_reasons.append("node_id_coverage_below_80_percent")
    if field_coverage["semantic_type"] < 0.80:
        readiness_reasons.append("semantic_type_coverage_below_80_percent")
    if explicit_edges == 0:
        readiness_reasons.append("no_explicit_runtime_edges")
    return {
        "schema_version": SCHEMA_VERSION,
        "events": total,
        "workflows": len(workflows),
        "valid_event_fraction": (
            sum(not validate_event(event) for event in events) / total if total else 0
        ),
        "errors": dict(errors),
        "duplicate_event_ids": duplicate_ids,
        "workflows_with_nonmonotonic_sequence": nonmonotonic_sequence,
        "workflows_with_nonmonotonic_time": nonmonotonic_time,
        "event_type_distribution": dict(Counter(
            event.get("event_type", "missing") for event in events
        )),
        "field_coverage": field_coverage,
        "explicit_edge_events": explicit_edges,
        "schema_valid": schema_valid,
        "runtime_ground_truth_ready": not readiness_reasons,
        "runtime_ground_truth_readiness_reasons": readiness_reasons,
    }
