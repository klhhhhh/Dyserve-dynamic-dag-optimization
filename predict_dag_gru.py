#!/usr/bin/env python3
"""Small multi-head GRU for bounded workflow-suffix prediction.

Input is the normalized JSONL produced by prepare_swe_trace.py.  The model
encodes up to ``history_length`` observed nodes and uses one classification
head for each future depth 1..horizon.
"""

import argparse
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

from predict_dag import (
    NGramDAGPredictor,
    STOP,
    evaluate as evaluate_ngram,
    future_label,
    load_sequences,
    macro_f1,
    sequence_stats,
    split_by_instance,
)

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Install it with: python3 -m pip install torch"
    ) from exc


PAD = "<PAD>"
UNK = "<UNK>"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_vocabulary(sequences):
    counts = Counter(node for item in sequences for node in item["nodes"])
    labels = sorted(counts)
    tokens = [PAD, UNK] + labels
    token_to_id = {token: index for index, token in enumerate(tokens)}
    label_to_id = {label: index for index, label in enumerate(labels)}
    return tokens, labels, token_to_id, label_to_id, counts


class TracePositionDataset(Dataset):
    """Stores sequence/position references instead of copying every prefix."""

    def __init__(
        self, sequences, token_to_id, label_to_id, history_length, horizon
    ):
        self.sequences = sequences
        self.token_to_id = token_to_id
        self.label_to_id = label_to_id
        self.history_length = history_length
        self.horizon = horizon
        self.positions = [
            (sequence_index, position)
            for sequence_index, item in enumerate(sequences)
            for position in range(1, len(item["nodes"]))
        ]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        sequence_index, position = self.positions[index]
        item = self.sequences[sequence_index]
        history_nodes = item["nodes"][max(0, position - self.history_length) : position]
        history = [self.token_to_id.get(node, self.token_to_id[UNK]) for node in history_nodes]
        truth_nodes = future_label(item["nodes"], position, self.horizon)
        truth = [self.label_to_id.get(node, self.label_to_id[STOP]) for node in truth_nodes]
        return {
            "history": history,
            "truth": truth,
            "instance_id": item["instance_id"],
            "history_nodes": history_nodes,
            "truth_nodes": truth_nodes,
        }


def make_collate(pad_id):
    def collate(batch):
        lengths = torch.tensor([len(x["history"]) for x in batch], dtype=torch.long)
        max_length = int(lengths.max())
        histories = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
        for row, item in enumerate(batch):
            histories[row, : len(item["history"])] = torch.tensor(item["history"])
        truths = torch.tensor([x["truth"] for x in batch], dtype=torch.long)
        return {
            "history": histories,
            "lengths": lengths,
            "truth": truths,
            "instance_id": [x["instance_id"] for x in batch],
            "history_nodes": [x["history_nodes"] for x in batch],
            "truth_nodes": [x["truth_nodes"] for x in batch],
        }

    return collate


class MultiHeadGRU(nn.Module):
    def __init__(self, vocab_size, label_count, embedding_dim, hidden_size, horizon, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(
            nn.Linear(hidden_size, label_count) for _ in range(horizon)
        )

    def forward(self, history, lengths):
        embedded = self.embedding(history)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        state = self.dropout(hidden[-1])
        return torch.stack([head(state) for head in self.heads], dim=1)


def class_weights(sequences, labels, label_to_id):
    counts = Counter()
    for item in sequences:
        nodes = item["nodes"]
        for position in range(1, len(nodes)):
            counts.update(future_label(nodes, position, 1))
    weights = torch.ones(len(labels), dtype=torch.float)
    total = sum(counts.values())
    for label in labels:
        count = max(1, counts[label])
        weights[label_to_id[label]] = math.sqrt(total / count)
    return weights / weights.mean()


def weighted_loss(logits, truth, criterion, depth_decay):
    losses = []
    for depth in range(logits.shape[1]):
        losses.append((depth_decay**depth) * criterion(logits[:, depth], truth[:, depth]))
    return sum(losses) / sum(depth_decay**d for d in range(logits.shape[1]))


@torch.no_grad()
def mean_loss(model, loader, criterion, depth_decay, device):
    model.eval()
    total_loss = total_rows = 0
    for batch in loader:
        logits = model(batch["history"].to(device), batch["lengths"])
        loss = weighted_loss(logits, batch["truth"].to(device), criterion, depth_decay)
        total_loss += loss.item() * len(batch["truth"])
        total_rows += len(batch["truth"])
    return total_loss / total_rows if total_rows else float("inf")


def train_model(model, train_loader, validation_loader, criterion, args, device):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = None
    best_validation = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = rows = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["history"].to(device), batch["lengths"])
            loss = weighted_loss(
                logits, batch["truth"].to(device), criterion, args.depth_decay
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            running_loss += loss.item() * len(batch["truth"])
            rows += len(batch["truth"])
        validation_loss = mean_loss(
            model, validation_loader, criterion, args.depth_decay, device
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / rows,
            "validation_loss": validation_loss,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result))
        if validation_loss < best_validation - args.min_delta:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break
    model.load_state_dict(best_state)
    model.to(device)
    return history


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, truths = [], []
    metadata = []
    for batch in loader:
        logits.append(model(batch["history"].to(device), batch["lengths"]).cpu())
        truths.append(batch["truth"])
        metadata.extend(
            zip(batch["instance_id"], batch["history_nodes"], batch["truth_nodes"])
        )
    return torch.cat(logits), torch.cat(truths), metadata


def fit_temperature(logits, truths):
    """Simple validation-set grid search; robust on CPU/MPS and dependency-free."""
    best_temperature, best_nll = 1.0, float("inf")
    flat_truth = truths.reshape(-1)
    for step in range(10, 61):
        temperature = step / 20  # 0.5 .. 3.0
        flat_logits = (logits / temperature).reshape(-1, logits.shape[-1])
        nll = nn.functional.cross_entropy(flat_logits, flat_truth).item()
        if nll < best_nll:
            best_temperature, best_nll = temperature, nll
    return best_temperature


def top_k_sequences(probabilities, labels, k):
    """Return the k highest-product suffixes from independent depth heads."""
    choices = []
    for depth_probs in probabilities:
        depth_choices = [(labels[i], float(p)) for i, p in enumerate(depth_probs)]
        choices.append(depth_choices)
    candidates = []
    for combination in itertools.product(*choices):
        sequence = tuple(item[0] for item in combination)
        score = math.prod(item[1] for item in combination)
        candidates.append((score, sequence))
    candidates.sort(reverse=True)
    return [sequence for _, sequence in candidates[:k]]


def evaluate_gru(logits, truths, metadata, labels, temperature, top_k, confidence):
    probabilities = torch.softmax(logits / temperature, dim=-1)
    predicted_ids = probabilities.argmax(dim=-1)
    predicted_confidences = probabilities.max(dim=-1).values
    rows = []
    for index in range(len(truths)):
        instance_id, history, truth_nodes = metadata[index]
        truth = tuple(truth_nodes)
        predicted = tuple(labels[i] for i in predicted_ids[index].tolist())
        top_sequences = top_k_sequences(probabilities[index].tolist(), labels, top_k)
        prefix_len = 0
        for prediction, target in zip(predicted, truth):
            if prediction != target:
                break
            prefix_len += 1
        # All predicted nodes must be sufficiently confident before compiling.
        sequence_confidence = float(predicted_confidences[index].min())
        rows.append(
            {
                "instance_id": instance_id,
                "history": history,
                "truth": truth,
                "predicted": predicted,
                "top_k": top_sequences,
                "confidence": sequence_confidence,
                "per_depth_confidence": predicted_confidences[index].tolist(),
                "prefix_len": prefix_len,
            }
        )

    sample_count = len(rows)
    transitions = [r for r in rows if r["truth"][0] != r["history"][-1]]
    triggered = [r for r in rows if r["confidence"] >= confidence]
    depth_metrics = []
    for depth in range(len(truths[0])):
        pairs = [(r["predicted"][depth], r["truth"][depth]) for r in rows]
        depth_metrics.append(
            {
                "depth": depth + 1,
                "accuracy": sum(p == y for p, y in pairs) / sample_count,
                "macro_f1": macro_f1(pairs),
                "prefix_accuracy": sum(r["prefix_len"] >= depth + 1 for r in rows)
                / sample_count,
            }
        )
    metrics = {
        "samples": sample_count,
        "temperature": temperature,
        "exact_sequence_at_1": sum(r["predicted"] == r["truth"] for r in rows)
        / sample_count,
        f"sequence_coverage_at_{top_k}": sum(r["truth"] in r["top_k"] for r in rows)
        / sample_count,
        "mean_correct_prefix_nodes": sum(r["prefix_len"] for r in rows) / sample_count,
        "next_node_transition_accuracy": sum(
            r["predicted"][0] == r["truth"][0] for r in transitions
        )
        / len(transitions),
        "transition_samples": len(transitions),
        "depth_metrics": depth_metrics,
        "compile_gate": {
            "confidence_threshold": confidence,
            "trigger_rate": len(triggered) / sample_count,
            "exact_precision_when_triggered": sum(
                r["predicted"] == r["truth"] for r in triggered
            )
            / len(triggered)
            if triggered
            else 0.0,
            "mean_safe_prefix_when_triggered": sum(r["prefix_len"] for r in triggered)
            / len(triggered)
            if triggered
            else 0.0,
        },
    }
    examples = sorted(rows, key=lambda row: -row["confidence"])[:10]
    return metrics, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path")
    parser.add_argument("--history-length", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--depth-decay", type=float, default=0.7)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--checkpoint-out")
    args = parser.parse_args()
    if args.history_length < 1 or args.horizon < 1:
        parser.error("history-length and horizon must be positive")

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    sequences = load_sequences(args.trace_path)
    training_pool, test = split_by_instance(sequences, args.test_ratio, args.seed)
    train, validation = split_by_instance(
        training_pool, args.validation_ratio, args.seed + 1
    )
    tokens, labels, token_to_id, label_to_id, counts = build_vocabulary(train)
    print(f"Labels: {labels}")

    train_dataset = TracePositionDataset(
        train, token_to_id, label_to_id, args.history_length, args.horizon
    )
    validation_dataset = TracePositionDataset(
        validation, token_to_id, label_to_id, args.history_length, args.horizon
    )
    test_dataset = TracePositionDataset(
        test, token_to_id, label_to_id, args.history_length, args.horizon
    )
    collate = make_collate(token_to_id[PAD])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, collate_fn=collate
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate)

    model = MultiHeadGRU(
        len(tokens), len(labels), args.embedding_dim, args.hidden_size,
        args.horizon, args.dropout
    ).to(device)
    weights = None if args.no_class_weights else class_weights(train, labels, label_to_id)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    training_history = train_model(
        model, train_loader, validation_loader, criterion, args, device
    )

    validation_logits, validation_truths, _ = collect_logits(
        model, validation_loader, device
    )
    temperature = fit_temperature(validation_logits, validation_truths)
    test_logits, test_truths, test_metadata = collect_logits(model, test_loader, device)
    metrics, examples = evaluate_gru(
        test_logits, test_truths, test_metadata, labels, temperature,
        args.top_k, args.confidence
    )

    baseline = NGramDAGPredictor(0, args.horizon).fit(train)
    baseline_metrics, _ = evaluate_ngram(baseline, test, args.top_k, args.confidence)
    ngram = NGramDAGPredictor(3, args.horizon).fit(train)
    ngram_metrics, _ = evaluate_ngram(ngram, test, args.top_k, args.confidence)
    report = {
        "config": vars(args),
        "device": str(device),
        "vocabulary": {"tokens": tokens, "labels": labels, "train_counts": counts},
        "all_data": sequence_stats(sequences),
        "train": sequence_stats(train),
        "validation": sequence_stats(validation),
        "test": sequence_stats(test),
        "training_history": training_history,
        "model": metrics,
        "ngram_window_3": ngram_metrics,
        "context_free_baseline": baseline_metrics,
        "examples": examples,
    }
    output = Path(args.report_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=list) + "\n",
        encoding="utf-8",
    )
    if args.checkpoint_out:
        checkpoint = Path(args.checkpoint_out)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "tokens": tokens,
                "labels": labels,
                "temperature": temperature,
                "config": vars(args),
            },
            checkpoint,
        )
    print(json.dumps(metrics, indent=2))
    print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()
