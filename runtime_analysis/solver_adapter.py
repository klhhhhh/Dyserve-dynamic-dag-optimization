"""Solver interface plus an explicitly non-scientific mock implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class SolverResult:
    feasible: bool
    objective: float
    solver_time_ms: float
    assignments: dict[str, str]
    metadata: dict[str, Any]


class SolverAdapter(Protocol):
    name: str
    scientific_validity: str

    def solve(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
              resource_snapshot: dict[str, Any]) -> SolverResult: ...


class MockCapacitySolver:
    """Deterministic plumbing test; never use its results as Dyserve evidence."""

    name = "mock_capacity_solver"
    scientific_validity = "integration_test_only_not_dyserve_ilp"
    MODEL_BY_TYPE = {
        "reason": "large", "edit": "medium", "shell": "medium",
        "test": "small", "inspect": "small", "web": "small", "other": "small",
    }
    COST = {"large": 4.0, "medium": 2.0, "small": 1.0}

    def solve(self, nodes, edges, resource_snapshot):
        assignments = {
            str(node["id"]): self.MODEL_BY_TYPE.get(
                node.get("semantic_type", "other"), "small"
            )
            for node in nodes
        }
        capacity = int(resource_snapshot.get("capacity", 10_000))
        demand = sum(self.COST[value] for value in assignments.values())
        return SolverResult(
            feasible=demand <= capacity,
            objective=demand,
            solver_time_ms=2.0 + 0.3 * len(nodes) + 0.05 * len(edges),
            assignments=assignments,
            metadata={"capacity": capacity, "warning": self.scientific_validity},
        )


def build_solver(name: str) -> SolverAdapter:
    if name == "mock":
        return MockCapacitySolver()
    raise ValueError(
        f"Unknown solver {name!r}. Add a real adapter implementing SolverAdapter."
    )

