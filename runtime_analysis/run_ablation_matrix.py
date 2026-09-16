#!/usr/bin/env python3
"""Generate or execute a reproducible CatBoost ablation matrix."""

import argparse
import json
import subprocess
from itertools import product
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dag_jsonl")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/ablation"))
    parser.add_argument("--feature-modes", default="sequence,graph,combined")
    parser.add_argument("--horizons", default="1,2,3")
    parser.add_argument("--history-lengths", default="4,8,16")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--thread-count", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    modes = [value.strip() for value in args.feature_modes.split(",")]
    horizons = [int(value) for value in args.horizons.split(",")]
    histories = [int(value) for value in args.history_lengths.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    runs = []
    for mode, horizon, history, seed in product(modes, horizons, histories, seeds):
        name = f"{mode}_h{horizon}_hist{history}_seed{seed}"
        command = [
            "python3", "predict/dag-trace-predict/train_dag_delta.py",
            args.dag_jsonl, "--feature-mode", mode, "--horizon", str(horizon),
            "--history-length", str(history), "--seed", str(seed),
            "--iterations", str(args.iterations),
            "--thread-count", str(args.thread_count),
            "--model-out", str(args.output_root / "models" / name),
            "--report-out", str(args.output_root / "reports" / f"{name}.json"),
            "--predictions-out",
            str(args.output_root / "predictions" / f"{name}.jsonl"),
        ]
        runs.append({"name": name, "command": command})
    if args.max_runs is not None:
        runs = runs[:args.max_runs]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "manifest.json"
    manifest.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    print(f"Prepared {len(runs)} runs in {manifest}")
    if args.execute:
        for index, run in enumerate(runs, 1):
            print(f"[{index}/{len(runs)}] {run['name']}", flush=True)
            subprocess.run(run["command"], check=True)


if __name__ == "__main__":
    main()

