#!/usr/bin/env python3
"""Train and evaluate a partial-DAG delta-signature predictor with CatBoost.

At every discovery position the model sees only the current partial graph and
predicts a bounded future delta signature such as:
  types=edit+test|topology=chain
  types=inspect+edit|topology=branch

This is intentionally a finite, ILP-friendly abstraction rather than exact
generation of arbitrary future node IDs and edges.
"""

import argparse
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path


PAD = "<PAD>"


def load_graphs(path):
    graphs = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                graph = json.loads(line)
                if graph.get("nodes"):
                    graphs.append(graph)
    return graphs


def split_by_instance(graphs, ratio, seed):
    ids = sorted({g["instance_id"] for g in graphs})
    rng = random.Random(seed)
    rng.shuffle(ids)
    count = min(len(ids) - 1, max(1, round(len(ids) * ratio)))
    held = set(ids[:count])
    return ([g for g in graphs if g["instance_id"] not in held],
            [g for g in graphs if g["instance_id"] in held])


def visible_graph(graph, position, min_confidence):
    nodes = graph["nodes"][:position]
    ids = {n["id"] for n in nodes}
    edges = [e for e in graph.get("edges", [])
             if e["source"] in ids and e["target"] in ids
             and e.get("confidence", 0) >= min_confidence]
    return nodes, edges


def graph_statistics(nodes, edges):
    indegree, outdegree, successors = Counter(), Counter(), defaultdict(list)
    for edge in edges:
        indegree[edge["target"]] += 1
        outdegree[edge["source"]] += 1
        successors[edge["source"]].append(edge["target"])
    depth = {n["id"]: 0 for n in nodes}
    queue = deque([n["id"] for n in nodes if indegree[n["id"]] == 0])
    while queue:
        source = queue.popleft()
        for target in successors[source]:
            depth[target] = max(depth[target], depth[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    semantic = Counter(n["semantic_type"] for n in nodes)
    statuses = Counter(n.get("status", "unknown") for n in nodes)
    return [
        len(nodes), len(edges), max(depth.values(), default=0),
        sum(outdegree[n["id"]] > 1 for n in nodes),
        sum(Counter(e["target"] for e in edges)[n["id"]] > 1 for n in nodes),
        max(outdegree.values(), default=0),
        max(Counter(e["target"] for e in edges).values(), default=0),
        sum(n.get("result_chars", 0) for n in nodes[-8:]),
        statuses["failed"], semantic["inspect"], semantic["edit"],
        semantic["shell"], semantic["test"], semantic["reason"],
    ]


def delta_signature(graph, position, horizon, min_confidence):
    future = graph["nodes"][position : position + horizon]
    if not future:
        return "types=STOP|topology=none"
    future_ids = {n["id"] for n in future}
    visible_ids = {n["id"] for n in graph["nodes"][:position]}
    incoming, internal = Counter(), []
    parent_fanout = Counter()
    for edge in graph.get("edges", []):
        if edge.get("confidence", 0) < min_confidence:
            continue
        if edge["target"] in future_ids:
            incoming[edge["target"]] += 1
            if edge["source"] in visible_ids:
                parent_fanout[edge["source"]] += 1
            if edge["source"] in future_ids:
                internal.append(edge)
    has_branch = any(v > 1 for v in parent_fanout.values())
    has_join = any(v > 1 for v in incoming.values())
    if has_branch and has_join:
        topology = "branch_join"
    elif has_branch:
        topology = "branch"
    elif has_join:
        topology = "join"
    elif internal:
        topology = "chain"
    else:
        topology = "independent"
    counts = Counter(n["semantic_type"] for n in future)
    types = "+".join(f"{key}:{counts[key]}" for key in sorted(counts))
    return f"types={types}|topology={topology}"


class Encoder:
    def __init__(self, history_length, min_confidence, feature_mode="combined"):
        self.history_length = history_length
        self.min_confidence = min_confidence
        self.feature_mode = feature_mode

    @property
    def categorical_indices(self):
        if self.feature_mode == "sequence":
            return list(range(self.history_length + 2))
        if self.feature_mode == "graph":
            return [0]  # language
        return list(range(self.history_length + 3))

    def one(self, graph, position):
        nodes, edges = visible_graph(graph, position, self.min_confidence)
        recent = [n["semantic_type"] for n in nodes[-self.history_length:]]
        recent = [PAD] * (self.history_length - len(recent)) + recent
        last = nodes[-1]
        sequence_features = recent + [
            str(last.get("tool_name", "unknown")),
            str(last.get("status", "unknown")),
        ]
        graph_features = [str(graph.get("language", "unknown"))]
        graph_features += graph_statistics(nodes, edges)
        if self.feature_mode == "sequence":
            return sequence_features
        if self.feature_mode == "graph":
            return graph_features
        return sequence_features + graph_features


def make_dataset(graphs, encoder, horizon, min_confidence):
    x, y = [], []
    for graph in graphs:
        # At least one observed node; exclude the point after the final node.
        for position in range(1, len(graph["nodes"])):
            x.append(encoder.one(graph, position))
            y.append(delta_signature(graph, position, horizon, min_confidence))
    return x, y


def metrics(labels, predictions, topk):
    n = len(labels)
    exact = sum(y == p[0] for y, p in zip(labels, predictions)) / n if n else 0
    covered = sum(y in p for y, p in zip(labels, predictions)) / n if n else 0
    return {"samples": n, "classes": len(set(labels)),
            "exact_accuracy": exact, f"top_{topk}_coverage": covered}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dag_jsonl")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--history-length", type=int, default=16)
    parser.add_argument(
        "--feature-mode",
        choices=("sequence", "graph", "combined"),
        default="combined",
        help=("sequence: recent execution history only; graph: partial-DAG "
              "statistics only; combined: both feature groups"),
    )
    parser.add_argument("--min-edge-confidence", type=float, default=0.6)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-class-count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise SystemExit("Install CatBoost: python3 -m pip install catboost") from exc

    graphs = load_graphs(args.dag_jsonl)
    pool, test = split_by_instance(graphs, args.test_ratio, args.seed)
    train, validation = split_by_instance(pool, args.validation_ratio, args.seed + 1)
    encoder = Encoder(
        args.history_length, args.min_edge_confidence, args.feature_mode
    )
    x_train, y_train = make_dataset(train, encoder, args.horizon, args.min_edge_confidence)
    x_val, y_val = make_dataset(validation, encoder, args.horizon, args.min_edge_confidence)
    x_test, y_test = make_dataset(test, encoder, args.horizon, args.min_edge_confidence)

    # Collapse signatures too rare to learn reliably; mapping is train-only.
    counts = Counter(y_train)
    known = {label for label, count in counts.items() if count >= args.min_class_count}
    remap = lambda labels: [label if label in known else "<RARE>" for label in labels]
    y_train, y_val, y_test = remap(y_train), remap(y_val), remap(y_test)
    model = CatBoostClassifier(
        loss_function="MultiClass", eval_metric="MultiClass",
        iterations=200, depth=6, learning_rate=0.06,
        random_seed=args.seed, verbose=False, allow_writing_files=False,
    )
    model.fit(x_train, y_train, cat_features=encoder.categorical_indices,
              eval_set=(x_val, y_val), early_stopping_rounds=50, verbose=False)

    probabilities = model.predict_proba(x_test)
    classes = list(model.classes_)
    top_predictions = []
    for row in probabilities:
        order = sorted(range(len(row)), key=lambda i: row[i], reverse=True)[:args.top_k]
        top_predictions.append([classes[i] for i in order])
    report = {
        "config": vars(args),
        "graphs": {"train": len(train), "validation": len(validation), "test": len(test)},
        "train_class_counts": Counter(y_train),
        "test": metrics(y_test, top_predictions, args.top_k),
        "examples": [
            {"truth": y_test[i], "predictions": top_predictions[i],
             "probabilities": [float(probabilities[i][classes.index(p)])
                               for p in top_predictions[i]]}
            for i in range(min(20, len(y_test)))
        ],
    }
    model_path = Path(args.model_out)
    report_path = Path(args.report_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                      default=dict) + "\n")
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
