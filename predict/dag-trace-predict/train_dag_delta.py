#!/usr/bin/env python3
"""Train a multi-task partial-DAG delta predictor with CatBoost.

The model no longer treats every (node counts + topology) combination as one
growing class.  It trains one fixed 0..H count classifier per semantic node
type and one separate topology classifier.
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

PAD = "<PAD>"
DEFAULT_NODE_TYPES = "inspect,edit,shell,test,reason,web"


def load_graphs(path):
    with open(path, encoding="utf-8") as stream:
        return [g for line in stream if line.strip()
                if (g := json.loads(line)).get("nodes")]


def split_by_instance(graphs, ratio, seed):
    ids = sorted({g["instance_id"] for g in graphs})
    if len(ids) < 2:
        raise ValueError("At least two instance IDs are required")
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
    original_indegree = indegree.copy()
    depth = {node["id"]: 0 for node in nodes}
    queue = deque(node["id"] for node in nodes if indegree[node["id"]] == 0)
    while queue:
        source = queue.popleft()
        for target in successors[source]:
            depth[target] = max(depth[target], depth[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    semantic = Counter(node["semantic_type"] for node in nodes)
    statuses = Counter(node.get("status", "unknown") for node in nodes)
    return [
        len(nodes), len(edges), max(depth.values(), default=0),
        sum(outdegree[node["id"]] > 1 for node in nodes),
        sum(original_indegree[node["id"]] > 1 for node in nodes),
        max(outdegree.values(), default=0),
        max(original_indegree.values(), default=0),
        sum(node.get("result_chars", 0) for node in nodes[-8:]),
        statuses["failed"], semantic["inspect"], semantic["edit"],
        semantic["shell"], semantic["test"], semantic["reason"],
    ]


def future_targets(graph, position, horizon, min_confidence, node_types):
    future = graph["nodes"][position:position + horizon]
    future_ids = {node["id"] for node in future}
    visible_ids = {node["id"] for node in graph["nodes"][:position]}
    incoming, parent_fanout = Counter(), Counter()
    internal_edges = 0
    for edge in graph.get("edges", []):
        if edge.get("confidence", 0) < min_confidence:
            continue
        if edge["target"] not in future_ids:
            continue
        incoming[edge["target"]] += 1
        if edge["source"] in visible_ids:
            parent_fanout[edge["source"]] += 1
        if edge["source"] in future_ids:
            internal_edges += 1
    branch = any(value > 1 for value in parent_fanout.values())
    join = any(value > 1 for value in incoming.values())
    if not future:
        topology = "none"
    elif branch and join:
        topology = "branch_join"
    elif branch:
        topology = "branch"
    elif join:
        topology = "join"
    elif internal_edges:
        topology = "chain"
    else:
        topology = "independent"
    raw_counts = Counter(node["semantic_type"] for node in future)
    counts = {node_type: raw_counts[node_type] for node_type in node_types}
    counts["other"] = sum(count for node_type, count in raw_counts.items()
                          if node_type not in node_types)
    return topology, counts


class Encoder:
    def __init__(self, history_length, min_confidence, feature_mode):
        self.history_length = history_length
        self.min_confidence = min_confidence
        self.feature_mode = feature_mode

    @property
    def categorical_indices(self):
        if self.feature_mode == "sequence":
            # language + history + tool + status
            return list(range(self.history_length + 3))

        if self.feature_mode == "graph":
            # language
            return [0]

        # language + history + tool + status
        return list(range(self.history_length + 3))

    def one(self, graph, position):
        nodes, edges = visible_graph(graph, position, self.min_confidence)

        static = [
            str(graph.get("language", "unknown")),
        ]

        recent = [
            node["semantic_type"]
            for node in nodes[-self.history_length:]
        ]
        recent = [PAD] * (self.history_length - len(recent)) + recent

        last = nodes[-1]
        sequence = recent + [
            str(last.get("tool_name", "unknown")),
            str(last.get("status", "unknown")),
        ]

        graph_features = graph_statistics(nodes, edges)

        if self.feature_mode == "sequence":
            return static + sequence

        if self.feature_mode == "graph":
            return static + graph_features

        return static + sequence + graph_features


def make_dataset(graphs, encoder, horizon, min_confidence, node_types):
    features, topologies, metadata = [], [], []
    count_labels = {node_type: [] for node_type in node_types + ["other"]}
    for graph in graphs:
        total_nodes = len(graph["nodes"])
        for position in range(1, total_nodes):
            topology, counts = future_targets(
                graph, position, horizon, min_confidence, node_types
            )
            features.append(encoder.one(graph, position))
            topologies.append(topology)
            remaining_nodes = total_nodes - position
            metadata.append({
                "instance_id": graph["instance_id"],
                "position": position,
                "visible_nodes": position,
                "total_nodes": total_nodes,
                "remaining_nodes": remaining_nodes,
                "progress_ratio": position / total_nodes,
                "full_horizon": remaining_nodes >= horizon,
                "last_status": graph["nodes"][position - 1].get(
                    "status", "unknown"
                ),
            })
            for node_type in count_labels:
                count_labels[node_type].append(counts[node_type])
    return features, topologies, count_labels, metadata


def macro_f1(truth, predicted):
    labels = set(truth) | set(predicted)
    values = []
    for label in labels:
        tp = sum(y == label and p == label for y, p in zip(truth, predicted))
        fp = sum(y != label and p == label for y, p in zip(truth, predicted))
        fn = sum(y == label and p != label for y, p in zip(truth, predicted))
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return sum(values) / len(values) if values else 0


def presence_metrics(truth_counts, predicted_counts):
    truth = [value > 0 for value in truth_counts]
    predicted = [value > 0 for value in predicted_counts]
    tp = sum(y and p for y, p in zip(truth, predicted))
    fp = sum(not y and p for y, p in zip(truth, predicted))
    fn = sum(y and not p for y, p in zip(truth, predicted))
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall else 0,
    }


def classification_report(truth, predicted):
    labels = sorted(set(truth) | set(predicted))
    report, confusion = {}, {label: Counter() for label in labels}
    for y, p in zip(truth, predicted):
        confusion[y][p] += 1
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_as_label = sum(row[label] for row in confusion.values())
        fp = predicted_as_label - tp
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall else 0,
            "support": support,
        }
    return report, {label: dict(confusion[label]) for label in labels}


def average_precision(binary_truth, scores):
    positives = sum(binary_truth)
    if not positives:
        return 0.0
    ranked = sorted(zip(scores, binary_truth), reverse=True)
    true_so_far = 0
    total = 0.0
    for rank, (_, truth) in enumerate(ranked, 1):
        if truth:
            true_so_far += 1
            total += true_so_far / rank
    return total / positives


def expected_calibration_error(binary_truth, scores, bins=10):
    total = len(binary_truth)
    error = 0.0
    for bin_index in range(bins):
        lower, upper = bin_index / bins, (bin_index + 1) / bins
        indices = [i for i, score in enumerate(scores)
                   if lower <= score < upper or (bin_index == bins - 1 and score == 1)]
        if not indices:
            continue
        confidence = sum(scores[i] for i in indices) / len(indices)
        accuracy = sum(binary_truth[i] for i in indices) / len(indices)
        error += len(indices) / total * abs(accuracy - confidence)
    return error


def topology_calibration(truth, probabilities, classes, bins=10):
    class_to_id = {label: index for index, label in enumerate(classes)}
    confidences, correct = [], []
    nll = brier = 0.0
    for y, row in zip(truth, probabilities):
        best = max(range(len(row)), key=lambda i: row[i])
        confidences.append(float(row[best]))
        correct.append(int(classes[best] == y))
        if y in class_to_id:
            probability = max(float(row[class_to_id[y]]), 1e-12)
            nll -= math.log(probability)
            brier += sum((float(row[i]) - int(classes[i] == y)) ** 2
                         for i in range(len(row)))
    return {
        "nll": nll / len(truth),
        "multiclass_brier": brier / len(truth),
        "ece": expected_calibration_error(correct, confidences, bins),
    }


def flattened_prediction(model, features, cast=str):
    values = model.predict(features)
    return [cast(value[0] if hasattr(value, "__len__") and not isinstance(value, str)
                 else value) for value in values]


def top_k_coverage(truth, rankings, max_k=3):
    """Return coverage@1..K for one ranked candidate list per sample."""
    if not truth:
        return {f"top_{k}_coverage": 0.0 for k in range(1, max_k + 1)}
    return {
        f"top_{k}_coverage": sum(
            label in ranking[:k] for label, ranking in zip(truth, rankings)
        ) / len(truth)
        for k in range(1, max_k + 1)
    }


def frequency_ranking(labels, fallback_order):
    """Rank classes by training frequency with deterministic tie breaking."""
    counts = Counter(labels)
    tie_order = {label: i for i, label in enumerate(fallback_order)}
    return sorted(
        fallback_order,
        key=lambda label: (-counts[label], tie_order[label]),
    )


def conditioned_frequency_baseline(train_labels, train_metadata, test_metadata,
                                   bucket_fn, global_ranking, max_k=3):
    """Build bucket-specific rankings from train only and evaluate on test."""
    bucket_labels = defaultdict(list)
    for label, sample in zip(train_labels, train_metadata):
        bucket_labels[bucket_fn(sample)].append(label)

    rankings_by_bucket = {
        bucket: frequency_ranking(labels, global_ranking)
        for bucket, labels in bucket_labels.items()
    }
    test_rankings = [
        rankings_by_bucket.get(bucket_fn(sample), global_ranking)
        for sample in test_metadata
    ]
    return rankings_by_bucket, test_rankings


def prefix_bucket(position):
    if position == 1:
        return "after_1_node"
    if position == 2:
        return "after_2_nodes"
    if position == 3:
        return "after_3_nodes"
    if position <= 7:
        return "after_4_7_nodes"
    if position <= 15:
        return "after_8_15_nodes"
    return "after_16_plus_nodes"


def progress_bucket(ratio):
    if ratio <= 0.2:
        return "progress_0_20"
    if ratio <= 0.4:
        return "progress_20_40"
    if ratio <= 0.6:
        return "progress_40_60"
    if ratio <= 0.8:
        return "progress_60_80"
    return "progress_80_100"


def evaluate_subset(indices, metadata, topology_truth, topology_pred, top3,
                    counts_truth, counts_pred):
    """Evaluate a selected set of replay positions without retraining."""
    if not indices:
        return {"samples": 0, "workflows": 0}

    subset_topology_truth = [topology_truth[i] for i in indices]
    subset_topology_pred = [topology_pred[i] for i in indices]
    flat_presence_truth, flat_presence_pred = [], []
    type_presence_f1 = []

    for node_type in counts_truth:
        truth = [counts_truth[node_type][i] for i in indices]
        predicted = [counts_pred[node_type][i] for i in indices]
        flat_presence_truth.extend(int(value > 0) for value in truth)
        flat_presence_pred.extend(int(value > 0) for value in predicted)
        type_presence_f1.append(presence_metrics(truth, predicted)["f1"])

    exact_counts = sum(
        all(counts_truth[node_type][i] == counts_pred[node_type][i]
            for node_type in counts_truth)
        for i in indices
    ) / len(indices)

    return {
        "samples": len(indices),
        "workflows": len({metadata[i]["instance_id"] for i in indices}),
        "topology_accuracy": sum(
            topology_truth[i] == topology_pred[i] for i in indices
        ) / len(indices),
        "topology_macro_f1": macro_f1(
            subset_topology_truth, subset_topology_pred
        ),
        "top_3_coverage": sum(
            topology_truth[i] in top3[i] for i in indices
        ) / len(indices),
        "presence_micro": presence_metrics(
            flat_presence_truth, flat_presence_pred
        ),
        "presence_macro_f1": sum(type_presence_f1) / len(type_presence_f1),
        "exact_node_count_vector": exact_counts,
        "topology_distribution": dict(Counter(subset_topology_truth)),
    }


def grouped_position_report(metadata, grouping, topology_truth, topology_pred,
                            top3, counts_truth, counts_pred):
    groups = defaultdict(list)
    for index, sample in enumerate(metadata):
        groups[grouping(sample)].append(index)
    return {
        name: evaluate_subset(
            indices, metadata, topology_truth, topology_pred, top3,
            counts_truth, counts_pred
        )
        for name, indices in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dag_jsonl")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--history-length", type=int, default=16)
    parser.add_argument("--feature-mode", choices=("sequence", "graph", "combined"),
                        default="combined")
    parser.add_argument("--node-types", default=DEFAULT_NODE_TYPES,
                        help="Fixed comma-separated output vocabulary")
    parser.add_argument("--min-edge-confidence", type=float, default=0.6)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", required=True,
                        help="Output directory for topology and count models")
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()
    node_types = [value.strip() for value in args.node_types.split(",") if value.strip()]
    if not node_types:
        parser.error("--node-types must not be empty")
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise SystemExit("Install CatBoost: python3 -m pip install catboost") from exc

    graphs = load_graphs(args.dag_jsonl)
    pool, test = split_by_instance(graphs, args.test_ratio, args.seed)
    train, validation = split_by_instance(pool, args.validation_ratio, args.seed + 1)
    encoder = Encoder(args.history_length, args.min_edge_confidence, args.feature_mode)
    train_data = make_dataset(train, encoder, args.horizon,
                              args.min_edge_confidence, node_types)
    validation_data = make_dataset(validation, encoder, args.horizon,
                                   args.min_edge_confidence, node_types)
    test_data = make_dataset(test, encoder, args.horizon,
                             args.min_edge_confidence, node_types)
    x_train, topology_train, counts_train, train_metadata = train_data
    x_val, topology_val, counts_val, val_metadata = validation_data
    x_test, topology_test, counts_test, test_metadata = test_data

    model_dir = Path(args.model_out)
    report_path = Path(args.report_out)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    common = dict(iterations=args.iterations, depth=args.depth,
                  learning_rate=args.learning_rate, random_seed=args.seed,
                  thread_count=args.thread_count, allow_writing_files=False,
                  verbose=args.verbose)
    topology_model = CatBoostClassifier(loss_function="MultiClass", **common)
    topology_model.fit(x_train, topology_train,
                       cat_features=encoder.categorical_indices)
    topology_model.save_model(str(model_dir / "topology.cbm"))
    topology_pred = flattened_prediction(topology_model, x_test, str)
    topology_probabilities = topology_model.predict_proba(x_test)
    topology_classes = [str(value) for value in topology_model.classes_]
    topology_rankings = []
    for row in topology_probabilities:
        order = sorted(range(len(row)), key=lambda i: row[i], reverse=True)
        topology_rankings.append([topology_classes[i] for i in order])
    top3 = [ranking[:3] for ranking in topology_rankings]

    count_predictions = {}
    count_presence_scores = {}
    count_reports = {}
    constant_counts = {}
    for node_type, labels in counts_train.items():
        unique = sorted(set(labels))
        if len(unique) == 1:
            constant_counts[node_type] = unique[0]
            predicted = [unique[0]] * len(x_test)
            presence_scores = [float(unique[0] > 0)] * len(x_test)
        else:
            model = CatBoostClassifier(loss_function="MultiClass", **common)
            model.fit(x_train, [str(value) for value in labels],
                      cat_features=encoder.categorical_indices)
            model.save_model(str(model_dir / f"count_{node_type}.cbm"))
            predicted = flattened_prediction(model, x_test, int)
            probabilities = model.predict_proba(x_test)
            model_classes = [int(value) for value in model.classes_]
            zero_index = model_classes.index(0) if 0 in model_classes else None
            presence_scores = [
                1.0 - float(row[zero_index]) if zero_index is not None else 1.0
                for row in probabilities
            ]
        count_predictions[node_type] = predicted
        count_presence_scores[node_type] = presence_scores
        truth = counts_test[node_type]
        positive_indices = [i for i, value in enumerate(truth) if value > 0]
        majority_count = Counter(counts_train[node_type]).most_common(1)[0][0]
        binary_truth = [int(value > 0) for value in truth]
        count_reports[node_type] = {
            "count_accuracy": sum(y == p for y, p in zip(truth, predicted)) / len(truth),
            "count_mae": sum(abs(y - p) for y, p in zip(truth, predicted)) / len(truth),
            "presence": presence_metrics(truth, predicted),
            "presence_auprc": average_precision(binary_truth, presence_scores),
            "presence_ece": expected_calibration_error(binary_truth, presence_scores),
            "positive_count_samples": len(positive_indices),
            "positive_count_accuracy": (
                sum(truth[i] == predicted[i] for i in positive_indices)
                / len(positive_indices) if positive_indices else 0
            ),
            "positive_count_mae": (
                sum(abs(truth[i] - predicted[i]) for i in positive_indices)
                / len(positive_indices) if positive_indices else 0
            ),
            "majority_count_from_train": majority_count,
            "majority_count_accuracy": sum(value == majority_count for value in truth)
            / len(truth),
            "test_distribution": dict(Counter(truth)),
        }

    count_types = list(counts_test)
    exact_counts = [all(counts_test[node_type][i] == count_predictions[node_type][i]
                        for node_type in count_types) for i in range(len(x_test))]
    complete = [exact_counts[i] and topology_test[i] == topology_pred[i]
                for i in range(len(x_test))]
    topology_per_class, topology_confusion = classification_report(
        topology_test, topology_pred
    )
    topology_majority = Counter(topology_train).most_common(1)[0][0]
    flat_presence_truth = []
    flat_presence_predicted = []
    type_presence_f1 = []
    for node_type in count_types:
        truth_binary = [value > 0 for value in counts_test[node_type]]
        predicted_binary = [value > 0 for value in count_predictions[node_type]]
        flat_presence_truth.extend(truth_binary)
        flat_presence_predicted.extend(predicted_binary)
        type_presence_f1.append(
            presence_metrics(counts_test[node_type], count_predictions[node_type])["f1"]
        )
    presence_micro = presence_metrics(
        [int(value) for value in flat_presence_truth],
        [int(value) for value in flat_presence_predicted],
    )
    majority_count_vector = {
        node_type: Counter(counts_train[node_type]).most_common(1)[0][0]
        for node_type in count_types
    }
    majority_exact_counts = [
        all(counts_test[node_type][i] == majority_count_vector[node_type]
            for node_type in count_types) for i in range(len(x_test))
    ]
    by_visible_prefix = grouped_position_report(
        test_metadata,
        lambda sample: prefix_bucket(sample["position"]),
        topology_test, topology_pred, top3, counts_test, count_predictions,
    )
    by_progress_ratio = grouped_position_report(
        test_metadata,
        lambda sample: progress_bucket(sample["progress_ratio"]),
        topology_test, topology_pred, top3, counts_test, count_predictions,
    )
    full_horizon_indices = [
        i for i, sample in enumerate(test_metadata) if sample["full_horizon"]
    ]
    truncated_tail_indices = [
        i for i, sample in enumerate(test_metadata) if not sample["full_horizon"]
    ]
    active_topology_classes = sorted(set(topology_train))
    global_ranking = frequency_ranking(
        topology_train, active_topology_classes
    )
    global_test_rankings = [global_ranking] * len(topology_test)
    position_candidates, position_test_rankings = conditioned_frequency_baseline(
        topology_train,
        train_metadata,
        test_metadata,
        lambda sample: prefix_bucket(sample["position"]),
        global_ranking,
    )
    progress_candidates, progress_test_rankings = conditioned_frequency_baseline(
        topology_train,
        train_metadata,
        test_metadata,
        lambda sample: progress_bucket(sample["progress_ratio"]),
        global_ranking,
    )
    random_expected = {
        f"top_{k}_coverage": min(k, len(active_topology_classes))
        / len(active_topology_classes)
        for k in range(1, 4)
    }
    model_top_k = top_k_coverage(topology_test, topology_rankings)
    global_top_k = top_k_coverage(topology_test, global_test_rankings)
    position_top_k = top_k_coverage(topology_test, position_test_rankings)
    progress_top_k = top_k_coverage(topology_test, progress_test_rankings)
    report = {
        "config": vars(args),
        "node_types": node_types + ["other"],
        "graphs": {"train": len(train), "validation": len(validation), "test": len(test)},
        "samples": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "topology": {
            "classes_seen_in_train": sorted(set(topology_train)),
            "accuracy": sum(y == p for y, p in zip(topology_test, topology_pred)) / len(x_test),
            "macro_f1": macro_f1(topology_test, topology_pred),
            "top_3_coverage": sum(y in candidates for y, candidates
                                  in zip(topology_test, top3)) / len(x_test),
            "per_class": topology_per_class,
            "confusion_matrix": topology_confusion,
            "calibration": topology_calibration(
                topology_test, topology_probabilities, topology_classes
            ),
            "majority_class_from_train": topology_majority,
            "majority_accuracy": sum(y == topology_majority for y in topology_test)
            / len(topology_test),
            "test_distribution": dict(Counter(topology_test)),
        },
        "topology_top_k_baselines": {
            "note": (
                "All frequency candidates are selected from the training split "
                "only. Progress-conditioned results are diagnostic/oracle-style "
                "when final workflow length is unavailable at runtime."
            ),
            "active_classes": active_topology_classes,
            "number_of_active_classes": len(active_topology_classes),
            "model": {
                **model_top_k,
                "top_3_absolute_lift_vs_global": (
                    model_top_k["top_3_coverage"]
                    - global_top_k["top_3_coverage"]
                ),
                "top_3_absolute_lift_vs_position_conditioned": (
                    model_top_k["top_3_coverage"]
                    - position_top_k["top_3_coverage"]
                ),
                "top_3_absolute_lift_vs_progress_conditioned": (
                    model_top_k["top_3_coverage"]
                    - progress_top_k["top_3_coverage"]
                ),
            },
            "random_uniform_expected": random_expected,
            "global_frequency": {
                "ranked_candidates": global_ranking,
                **global_top_k,
            },
            "position_conditioned_frequency": {
                "runtime_available": True,
                "condition": "visible node-count bucket",
                "ranked_candidates_by_bucket": position_candidates,
                **position_top_k,
            },
            "progress_conditioned_frequency": {
                "runtime_available": False,
                "oracle_diagnostic": True,
                "condition": "visible_nodes / final_total_nodes",
                "ranked_candidates_by_bucket": progress_candidates,
                **progress_top_k,
            },
        },
        "node_counts": count_reports,
        "summary": {
            "exact_node_count_vector": sum(exact_counts) / len(exact_counts),
            "complete_exact_counts_and_topology": sum(complete) / len(complete),
            "mean_count_accuracy": sum(value["count_accuracy"]
                                       for value in count_reports.values()) / len(count_reports),
            "mean_presence_f1": sum(value["presence"]["f1"]
                                    for value in count_reports.values()) / len(count_reports),
            "presence_micro": presence_micro,
            "presence_macro_f1": sum(type_presence_f1) / len(type_presence_f1),
            "majority_exact_node_count_vector": sum(majority_exact_counts)
            / len(majority_exact_counts),
            "majority_complete_counts_and_topology": sum(
                majority_exact_counts[i] and topology_test[i] == topology_majority
                for i in range(len(x_test))
            ) / len(x_test),
        },
        "execution_position_analysis": {
            "by_visible_prefix": by_visible_prefix,
            "by_progress_ratio": by_progress_ratio,
        },
        "horizon_analysis": {
            "full_horizon_only": evaluate_subset(
                full_horizon_indices, test_metadata, topology_test,
                topology_pred, top3, counts_test, count_predictions,
            ),
            "truncated_workflow_tail": evaluate_subset(
                truncated_tail_indices, test_metadata, topology_test,
                topology_pred, top3, counts_test, count_predictions,
            ),
        },
        "constant_count_models": constant_counts,
    }
    (model_dir / "manifest.json").write_text(json.dumps({
        "config": vars(args), "node_types": node_types + ["other"],
        "constant_count_models": constant_counts,
    }, indent=2) + "\n")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"topology": report["topology"],
                      "topology_top_k_baselines": report["topology_top_k_baselines"],
                      "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
