"""Shared decision-proxy formulas."""

DEFAULT_DEMAND_WEIGHTS = {
    "inspect": 0.25, "edit": 1.0, "shell": 0.75, "test": 0.5,
    "reason": 1.25, "web": 0.5, "other": 0.5,
}


def demand_error(row, weights):
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

