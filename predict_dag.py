#!/usr/bin/env python3
"""Predict a bounded future workflow suffix from historical SWE trajectories.

The input is JSONL in either of two forms:
  1. Normalized: {"instance_id": ..., "run_id": ..., "nodes": [...]}
  2. Open-SWE-Traces/Hugging Face rows containing a ``trajectory`` field.

Only the observable tool-call sequence is used.  Splitting is always by
instance_id so rollouts for the same issue cannot leak across train and test.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


STOP = "STOP"


def as_json(value, fallback):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return value if value is not None else fallback


def classify_tool(name, arguments, taxonomy="semantic"):
    """Map heterogeneous agent tools to a small, ILP-friendly node taxonomy."""
    name = (name or "unknown").lower()
    args = as_json(arguments, {})
    args_text = json.dumps(args, ensure_ascii=False).lower()
    if taxonomy == "tool":
        return name

    if name in {"finish", "submit", "final", "done"}:
        return STOP
    if name in {"think", "reason", "plan"}:
        return "reason"
    if any(word in name for word in ("browser", "web", "search_web")):
        return "web"

    if any(word in name for word in ("edit", "patch", "write", "replace")):
        command = str(args.get("command", "")).lower()
        if command in {"view", "read"}:
            return "inspect"
        return "edit"

    command = str(
        args.get("command", args.get("cmd", args.get("query", args_text)))
    ).lower()
    if re.search(
        r"(^|[;&| ]+)(pytest|py\.test|tox|go test|cargo test|npm test|"
        r"npm run test|pnpm test|yarn test|mvn test|gradle test|make test)",
        command,
    ):
        return "test"
    if re.search(
        r"(^|[;&| ]+)(ls|find|grep|rg|sed|cat|head|tail|pwd|tree|"
        r"git (status|diff|log|show)|python[^ ]* -m pip show)([ ;&|]|$)",
        command,
    ):
        return "inspect"
    if any(word in name for word in ("search", "view", "read", "inspect")):
        return "inspect"
    if any(word in name for word in ("test", "lint", "check")):
        return "test"
    if any(word in name for word in ("bash", "shell", "terminal", "execute")):
        return "shell"
    return name


def nodes_from_trajectory(trajectory, taxonomy="semantic"):
    trajectory = as_json(trajectory, [])
    nodes = []
    for message in trajectory if isinstance(trajectory, list) else []:
        if not isinstance(message, dict):
            continue
        calls = as_json(message.get("tool_calls"), [])
        for call in calls if isinstance(calls, list) else []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            node = classify_tool(
                function.get("name"), function.get("arguments", {}), taxonomy
            )
            nodes.append(node)
    # STOP is a boundary marker, not an action after termination.
    nodes = nodes[: nodes.index(STOP) + 1] if STOP in nodes else nodes + [STOP]
    return nodes


def normalize_nodes(nodes):
    result = [str(node).strip() for node in nodes if str(node).strip()]
    if STOP in result:
        result = result[: result.index(STOP) + 1]
    else:
        result.append(STOP)
    return result


def load_sequences(path, taxonomy="semantic", converted_out=None):
    sequences, rejected = [], 0
    output = open(converted_out, "w", encoding="utf-8") if converted_out else None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    rejected += 1
                    continue
                if isinstance(item.get("nodes"), list):
                    nodes = normalize_nodes(item["nodes"])
                elif "trajectory" in item:
                    nodes = nodes_from_trajectory(item["trajectory"], taxonomy)
                else:
                    rejected += 1
                    continue
                if len(nodes) < 2:  # At least one action and STOP.
                    rejected += 1
                    continue
                sequence = {
                    "instance_id": str(item.get("instance_id", f"unknown-{line_number}")),
                    "run_id": str(
                        item.get("run_id", item.get("trajectory_id", f"run-{line_number}"))
                    ),
                    "nodes": nodes,
                    "resolved": item.get("resolved"),
                    "agent": item.get("agent", item.get("hf_dataset_name")),
                }
                sequences.append(sequence)
                if output:
                    output.write(json.dumps(sequence, ensure_ascii=False) + "\n")
    finally:
        if output:
            output.close()
    print(f"Loaded {len(sequences)} sequences ({rejected} rejected) from {path}")
    return sequences


def split_by_instance(sequences, test_ratio=0.2, seed=42):
    instance_ids = sorted({item["instance_id"] for item in sequences})
    if len(instance_ids) < 2:
        raise ValueError("Need at least two distinct instance_id values for evaluation")
    rng = random.Random(seed)
    rng.shuffle(instance_ids)
    test_count = min(len(instance_ids) - 1, max(1, round(len(instance_ids) * test_ratio)))
    test_ids = set(instance_ids[:test_count])
    train = [x for x in sequences if x["instance_id"] not in test_ids]
    test = [x for x in sequences if x["instance_id"] in test_ids]
    return train, test


def future_label(nodes, position, horizon):
    future = list(nodes[position : position + horizon])
    return tuple(future + [STOP] * (horizon - len(future)))


class NGramDAGPredictor:
    def __init__(self, window=3, horizon=3, alpha=0.25):
        self.window, self.horizon, self.alpha = window, horizon, alpha
        self.counts = defaultdict(Counter)

    def fit(self, sequences):
        for item in sequences:
            nodes = item["nodes"]
            for position in range(1, len(nodes)):
                future = future_label(nodes, position, self.horizon)
                for size in range(self.window + 1):
                    context = tuple(nodes[max(0, position - size) : position])
                    self.counts[context][future] += 1
        return self

    def distribution(self, history):
        max_size = min(self.window, len(history))
        for size in range(max_size, -1, -1):
            context = tuple(history[-size:]) if size else ()
            counter = self.counts.get(context)
            if counter:
                total = sum(counter.values()) + self.alpha * len(counter)
                return [
                    {
                        "future": future,
                        "probability": (count + self.alpha) / total,
                        "support": count,
                        "context": context,
                    }
                    for future, count in counter.most_common()
                ]
        return []

    def predict_top_k(self, history, k=4):
        return self.distribution(history)[:k]


def macro_f1(pairs):
    labels = {x for pair in pairs for x in pair}
    scores = []
    for label in labels:
        tp = sum(p == label and y == label for p, y in pairs)
        fp = sum(p == label and y != label for p, y in pairs)
        fn = sum(p != label and y == label for p, y in pairs)
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(model, sequences, top_k=4, confidence=0.6):
    rows = []
    for item in sequences:
        for position in range(1, len(item["nodes"])):
            history = item["nodes"][:position]
            truth = future_label(item["nodes"], position, model.horizon)
            predictions = model.predict_top_k(history, top_k)
            if not predictions:
                continue
            top = predictions[0]
            predicted = top["future"]
            prefix_len = 0
            for p, y in zip(predicted, truth):
                if p != y:
                    break
                prefix_len += 1
            rows.append(
                {
                    "instance_id": item["instance_id"],
                    "history": history,
                    "truth": truth,
                    "predicted": predicted,
                    "top_k": [x["future"] for x in predictions],
                    "confidence": top["probability"],
                    "prefix_len": prefix_len,
                    "context": top["context"],
                }
            )

    n = len(rows)
    exact = sum(r["predicted"] == r["truth"] for r in rows)
    covered = sum(r["truth"] in r["top_k"] for r in rows)
    triggered = [r for r in rows if r["confidence"] >= confidence]
    transitions = [r for r in rows if r["truth"][0] != r["history"][-1]]
    depth_metrics = []
    for depth in range(model.horizon):
        pairs = [(r["predicted"][depth], r["truth"][depth]) for r in rows]
        depth_metrics.append(
            {
                "depth": depth + 1,
                "accuracy": sum(p == y for p, y in pairs) / n if n else 0.0,
                "macro_f1": macro_f1(pairs),
                "prefix_accuracy": sum(r["prefix_len"] >= depth + 1 for r in rows) / n
                if n
                else 0.0,
            }
        )
    metrics = {
        "samples": n,
        "exact_sequence_at_1": exact / n if n else 0.0,
        f"sequence_coverage_at_{top_k}": covered / n if n else 0.0,
        "mean_correct_prefix_nodes": sum(r["prefix_len"] for r in rows) / n if n else 0.0,
        "next_node_transition_accuracy": (
            sum(r["predicted"][0] == r["truth"][0] for r in transitions) / len(transitions)
            if transitions
            else 0.0
        ),
        "transition_samples": len(transitions),
        "depth_metrics": depth_metrics,
        "compile_gate": {
            "confidence_threshold": confidence,
            "trigger_rate": len(triggered) / n if n else 0.0,
            "exact_precision_when_triggered": (
                sum(r["predicted"] == r["truth"] for r in triggered) / len(triggered)
                if triggered
                else 0.0
            ),
            "mean_safe_prefix_when_triggered": (
                sum(r["prefix_len"] for r in triggered) / len(triggered) if triggered else 0.0
            ),
        },
    }
    examples = sorted(rows, key=lambda r: (-r["confidence"], r["instance_id"]))[:10]
    return metrics, examples


def sequence_stats(sequences):
    actions = Counter(node for item in sequences for node in item["nodes"])
    lengths = sorted(len(item["nodes"]) - 1 for item in sequences)
    return {
        "runs": len(sequences),
        "instances": len({x["instance_id"] for x in sequences}),
        "node_counts": dict(actions.most_common()),
        "mean_actions_per_run": sum(lengths) / len(lengths) if lengths else 0.0,
        "median_actions_per_run": lengths[len(lengths) // 2] if lengths else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", help="Normalized or Open-SWE-Traces JSONL")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--taxonomy", choices=("semantic", "tool"), default="semantic")
    parser.add_argument("--converted-out", help="Optionally save normalized JSONL")
    parser.add_argument("--report-out", help="Optionally save all metrics as JSON")
    args = parser.parse_args()
    if args.horizon < 1 or args.window < 0 or args.top_k < 1:
        parser.error("horizon/top-k must be positive and window non-negative")

    sequences = load_sequences(args.trace_path, args.taxonomy, args.converted_out)
    train, test = split_by_instance(sequences, args.test_ratio, args.seed)
    model = NGramDAGPredictor(args.window, args.horizon, args.alpha).fit(train)
    metrics, examples = evaluate(model, test, args.top_k, args.confidence)

    # A context-free model is the required majority-pattern baseline.
    baseline = NGramDAGPredictor(0, args.horizon, args.alpha).fit(train)
    baseline_metrics, _ = evaluate(baseline, test, args.top_k, args.confidence)
    report = {
        "config": vars(args),
        "all_data": sequence_stats(sequences),
        "train": sequence_stats(train),
        "test": sequence_stats(test),
        "model": metrics,
        "context_free_baseline": baseline_metrics,
        "examples": examples,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False, default=list)
    print(rendered)
    if args.report_out:
        Path(args.report_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
