#!/usr/bin/env python3
"""Convert raw Open-SWE-Traces trajectories into weakly-labelled DAGs.

The source traces do not contain authoritative dependency edges.  Every edge
therefore records a reason and confidence.  The output keeps the original
tool-call discovery order so partial DAGs can later be replayed online.
"""

import argparse
import json
import re
import shlex
from collections import defaultdict
from pathlib import Path, PurePosixPath

from predict_dag import as_json, classify_tool


PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:\.?\.?/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.@+-]+)+|"
    r"(?<![A-Za-z0-9_])[A-Za-z0-9_.-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|c|h|"
    r"json|ya?ml|toml|md|txt|sh|sql)(?![A-Za-z0-9_])"
)


def normalize_path(value):
    value = str(value).strip().strip("'\"`:,;()[]{}")
    if not value or value.startswith(("http://", "https://")):
        return None
    value = value.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    try:
        return str(PurePosixPath(value))
    except Exception:
        return value


def paths_in_text(text):
    return {p for match in PATH_RE.findall(str(text or ""))
            if (p := normalize_path(match))}


def command_tokens(command):
    try:
        return shlex.split(str(command))
    except ValueError:
        return str(command).split()


def resources_for_call(name, arguments, semantic_type):
    """Infer resource reads/writes from tool arguments; deliberately conservative."""
    args = as_json(arguments, {})
    if not isinstance(args, dict):
        args = {}
    name = str(name or "unknown").lower()
    command = str(args.get("command", args.get("cmd", "")))
    explicit = set()
    for key in ("path", "file_path", "filepath", "target_file", "source_file"):
        value = args.get(key)
        if isinstance(value, str):
            explicit |= paths_in_text(value)
    mentioned = explicit | paths_in_text(command) | paths_in_text(json.dumps(args))
    reads, writes = set(), set()

    editor_command = str(args.get("command", "")).lower()
    if any(x in name for x in ("edit", "patch", "write", "replace")):
        if editor_command in {"view", "read"}:
            reads |= mentioned
        else:
            writes |= mentioned
    elif semantic_type == "inspect":
        reads |= mentioned
    elif semantic_type == "edit":
        writes |= mentioned
    elif semantic_type == "test":
        reads |= mentioned
    elif semantic_type == "shell":
        tokens = command_tokens(command)
        mutating = any(token in {"rm", "mv", "cp", "touch", "mkdir", "tee"}
                       for token in tokens)
        (writes if mutating else reads).update(mentioned)
    return sorted(reads), sorted(writes)


def tool_result(messages, message_index, call_id):
    """Find a nearby tool response and return compact status metadata."""
    for message in messages[message_index + 1 : message_index + 8]:
        if not isinstance(message, dict):
            continue
        if message.get("role") not in {"tool", "function"}:
            if message.get("role") == "assistant":
                break
            continue
        if call_id and message.get("tool_call_id") not in {None, call_id}:
            continue
        content = str(message.get("content", ""))
        failed = bool(re.search(
            r"\b(error|failed|failure|traceback|exception)\b|exit code[^0-9]*[1-9]",
            content, re.I
        ))
        return {"status": "failed" if failed else "success",
                "result_chars": len(content)}
    return {"status": "unknown", "result_chars": 0}


def add_edge(edges, seen, source, target, reason, confidence):
    if source == target:
        return
    key = (source, target, reason)
    if key not in seen:
        edges.append({"source": source, "target": target,
                      "reason": reason, "confidence": confidence})
        seen.add(key)


def trajectory_to_dag(row, taxonomy="semantic"):
    messages = as_json(row.get("trajectory", []), [])
    if not isinstance(messages, list):
        messages = []
    nodes, edges, edge_seen = [], [], set()
    last_writer = {}
    last_readers = defaultdict(list)
    previous_group = []
    last_failed = None

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        calls = as_json(message.get("tool_calls", []), [])
        if not isinstance(calls, list) or not calls:
            continue
        current_group = []
        for call_index, call in enumerate(calls):
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = function.get("name", "unknown")
            arguments = as_json(function.get("arguments", {}), {})
            semantic = classify_tool(name, arguments, taxonomy)
            node_id = f"n{len(nodes):04d}"
            reads, writes = resources_for_call(name, arguments, semantic)
            result = tool_result(messages, message_index,
                                 call.get("id") if isinstance(call, dict) else None)
            node = {
                "id": node_id,
                "discovery_index": len(nodes),
                "message_index": message_index,
                "parallel_group": message_index,
                "tool_name": str(name),
                "semantic_type": semantic,
                "reads": reads,
                "writes": writes,
                **result,
            }
            nodes.append(node)
            current_group.append(node_id)

            # Calls in the previous assistant turn are possible control parents.
            for parent in previous_group:
                add_edge(edges, edge_seen, parent, node_id,
                         "previous_tool_group", 0.35)
            if last_failed:
                add_edge(edges, edge_seen, last_failed, node_id,
                         "failure_recovery", 0.70)

            for resource in reads:
                if resource in last_writer:
                    add_edge(edges, edge_seen, last_writer[resource], node_id,
                             f"write_read:{resource}", 0.95)
                last_readers[resource].append(node_id)
            for resource in writes:
                if resource in last_writer:
                    add_edge(edges, edge_seen, last_writer[resource], node_id,
                             f"write_write:{resource}", 0.90)
                for reader in last_readers.get(resource, [])[-4:]:
                    add_edge(edges, edge_seen, reader, node_id,
                             f"read_write:{resource}", 0.80)
                last_writer[resource] = node_id
                last_readers[resource] = []
            if semantic == "test":
                for writer in list(dict.fromkeys(last_writer.values()))[-12:]:
                    add_edge(edges, edge_seen, writer, node_id,
                             "recent_write_before_test", 0.65)
            last_failed = node_id if result["status"] == "failed" else None
        previous_group = current_group

    return {
        "instance_id": str(row.get("instance_id", "unknown")),
        "run_id": str(row.get("trajectory_id", row.get("run_id", "unknown"))),
        "repo": row.get("repo"),
        "language": row.get("language"),
        "resolved": row.get("resolved"),
        "graph_source": "inferred_from_open_swe_trace",
        "nodes": nodes,
        "edges": edges,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", help="Raw JSONL with a trajectory field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--taxonomy", choices=("semantic", "tool"), default="semantic")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = nodes = edges = 0
    with open(args.input_jsonl, encoding="utf-8") as source, output.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if "trajectory" not in row:
                raise ValueError("Input must be raw Open-SWE-Traces JSONL with trajectory; normalized nodes are insufficient")
            graph = trajectory_to_dag(row, args.taxonomy)
            if graph["nodes"]:
                sink.write(json.dumps(graph, ensure_ascii=False) + "\n")
                rows += 1
                nodes += len(graph["nodes"])
                edges += len(graph["edges"])
            if args.max_rows and rows >= args.max_rows:
                break
    print(json.dumps({"graphs": rows, "nodes": nodes, "edges": edges,
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
