#!/usr/bin/env python3
"""Replay runtime events through a solver adapter.

The bundled mock solver validates integration only. It cannot produce oracle
gap or optimization claims about Dyserve.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from runtime_analysis.schema import read_events
from runtime_analysis.solver_adapter import build_solver


def replay(events, solver, horizon):
    grouped = defaultdict(list)
    for event in events:
        grouped[event["workflow_id"]].append(event)
    positions = []
    for workflow_id, rows in grouped.items():
        rows.sort(key=lambda row: row["sequence"])
        nodes, edges = [], []
        node_ids = set()
        node_events = []
        seen_for_replay = set()
        for row in rows:
            node_id = row.get("node_id")
            if not node_id or node_id in seen_for_replay:
                continue
            seen_for_replay.add(node_id)
            node_events.append(row)
        for index, event in enumerate(node_events):
            node_id = str(event["node_id"])
            if node_id not in node_ids:
                node_ids.add(node_id)
                nodes.append({
                    "id": node_id,
                    "semantic_type": event.get("semantic_type") or "other",
                })
                edges.extend(
                    {"source": parent, "target": node_id}
                    for parent in event.get("parent_ids", []) if parent in node_ids
                )
            future_events = node_events[index + 1:index + 1 + horizon]
            oracle_nodes = nodes + [
                {
                    "id": f"oracle:{workflow_id}:{index}:{j}",
                    "semantic_type": row.get("semantic_type") or "other",
                }
                for j, row in enumerate(future_events)
            ]
            snapshot = event.get("resource_snapshot") or {"capacity": 10_000}
            static = solver.solve(nodes, edges, snapshot)
            oracle = solver.solve(oracle_nodes, edges, snapshot)
            positions.append({
                "workflow_id": workflow_id,
                "position": index + 1,
                "static_objective": static.objective,
                "oracle_objective": oracle.objective,
                "static_solver_time_ms": static.solver_time_ms,
                "oracle_solver_time_ms": oracle.solver_time_ms,
                "oracle_future_nodes": len(future_events),
            })
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events_jsonl")
    parser.add_argument("--solver", default="mock", choices=("mock",))
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--report-out", required=True, type=Path)
    args = parser.parse_args()
    solver = build_solver(args.solver)
    positions = replay(read_events(args.events_jsonl), solver, args.horizon)
    report = {
        "solver": solver.name,
        "scientific_validity": solver.scientific_validity,
        "warning": (
            "Mock results validate replay plumbing only. They are not oracle "
            "gap, plan hit, latency improvement, or Dyserve measurements."
        ),
        "positions": len(positions),
        "workflows": len({row["workflow_id"] for row in positions}),
        "mean_static_solver_time_ms": (
            sum(row["static_solver_time_ms"] for row in positions) / len(positions)
            if positions else 0
        ),
        "records": positions,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
