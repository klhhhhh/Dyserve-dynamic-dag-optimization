# SWE-bench Runtime Data Collection Guide

## 1. Purpose

This guide explains how to run approximately 100 SWE-bench tasks with an
OpenHands agent and collect the resulting runtime event histories.

The generated data can be used to analyze:

- Tool-call execution order
- Tool start and completion events
- Success, failure, retry, and recovery behavior
- Runtime timestamps and execution duration
- File inspection, editing, shell, and testing operations
- Future node and resource demand

The collector preserves the original OpenHands events so that additional
features can be extracted later without rerunning the agent.

> Important: These records are real agent and tool runtime events, but they are
> not automatically ground-truth workflow DAG edges. Ground-truth events such
> as `node_created`, `edge_added`, and `branch_resolved` require instrumentation
> inside the workflow orchestrator.

## 2. Files

The runtime collection script is:

```text
run_swebench_runtime_100.py
```

Place it in the root of the Dyserve prediction repository, or adjust its path
in the commands below.

## 3. Prerequisites

The local workflow uses the official OpenHands benchmark runner and Docker.
Recommended resources are:

- Docker installed and running
- At least 16 GB RAM
- At least 120 GB available disk space
- Four or more CPU cores
- An LLM API key with enough quota for approximately 100 agent runs
- Git and `uv`

On Apple Silicon, SWE-bench Docker support may be slower or less reliable than
on an x86-64 Linux machine. A Linux server is recommended for the complete run.

Check the environment before starting:

```bash
docker --version
docker run --rm hello-world
uv --version
df -h
```

## 4. Install OpenHands Benchmarks

Clone and initialize the official OpenHands benchmark repository:

```bash
git clone https://github.com/OpenHands/benchmarks.git openhands-benchmarks
cd openhands-benchmarks
make build
cd ..
```

The benchmark repository includes the `swebench-infer` command used by the
collection script.

## 5. Configure the LLM

Create an OpenHands LLM configuration file, for example:

```text
openhands-benchmarks/.llm_config/my_model.json
```

Example configuration:

```json
{
  "model": "your-provider/your-model",
  "api_key": "YOUR_API_KEY"
}
```

Depending on the provider, the configuration may also need a `base_url`.
Follow the current OpenHands configuration schema for the selected provider.

Do not commit a configuration containing an API key to Git.

## 6. Start With a Small Smoke Test

Before running 100 tasks, test one or two instances:

```bash
python3 run_swebench_runtime_100.py \
  --benchmarks-dir ./openhands-benchmarks \
  --llm-config ./openhands-benchmarks/.llm_config/my_model.json \
  --dataset princeton-nlp/SWE-bench_Verified \
  --workspace docker \
  --limit 2 \
  --workers 1 \
  --max-iterations 20 \
  --output-dir runtime_data/swebench_smoke_test
```

Confirm that:

- Docker environments start correctly
- The LLM configuration is valid
- OpenHands writes an `output.jsonl`
- `runtime_events.jsonl` is generated
- The normalized records contain non-empty `events`

Example check:

```bash
wc -l runtime_data/swebench_smoke_test/runtime_events.jsonl
head -n 1 runtime_data/swebench_smoke_test/runtime_events.jsonl
```

## 7. Run Approximately 100 Tasks

After the smoke test succeeds, run the main collection:

```bash
python3 run_swebench_runtime_100.py \
  --benchmarks-dir ./openhands-benchmarks \
  --llm-config ./openhands-benchmarks/.llm_config/my_model.json \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --workspace docker \
  --limit 100 \
  --workers 4 \
  --max-iterations 100 \
  --tool-preset default \
  --output-dir runtime_data/swebench_100
```

The script uses the official OpenHands `--n-limit 100` option. Therefore, the
run normally selects the first 100 instances returned by the benchmark loader,
not a random sample.

Reduce `--workers` if Docker runs out of memory. Increasing the worker count
beyond the available CPU, memory, or API quota can make the run slower or less
stable.

## 8. Command-Line Options

| Option | Default | Meaning |
|---|---:|---|
| `--benchmarks-dir` | Required | Local OpenHands benchmarks checkout |
| `--llm-config` | Required | OpenHands LLM configuration path |
| `--dataset` | `princeton-nlp/SWE-bench_Verified` | SWE-bench dataset |
| `--split` | `test` | Dataset split |
| `--limit` | `100` | Number of instances |
| `--max-iterations` | `100` | Maximum agent iterations per instance |
| `--workers` | `4` | Number of concurrent instances |
| `--workspace` | `docker` | `docker`, `remote`, or `apptainer` |
| `--tool-preset` | `default` | OpenHands tool preset |
| `--output-dir` | `runtime_data/swebench_100` | Output directory |
| `--raw-output-jsonl` | None | Normalize an existing run without inference |
| `--extra-arg` | None | Extra argument passed to `swebench-infer` |

## 9. Output Files

The output directory will contain the original structured OpenHands run
directory and the normalized runtime file:

```text
runtime_data/swebench_100/
├── ... OpenHands structured output directories
├── ... original output.jsonl
└── runtime_events.jsonl
```

Each line in `runtime_events.jsonl` represents one SWE-bench instance:

```json
{
  "instance_id": "django__django-12345",
  "attempt": 1,
  "instruction": "...",
  "error": null,
  "metrics": {},
  "event_count": 54,
  "tool_event_count": 21,
  "failed_event_count": 2,
  "patch_chars": 1840,
  "events": []
}
```

Each normalized event contains:

```json
{
  "sequence": 8,
  "timestamp": "...",
  "event_type": "...",
  "phase": "tool_started",
  "tool_name": "terminal",
  "status": "unknown",
  "raw_event": {}
}
```

The complete `raw_event` is retained because OpenHands event schemas may vary
between versions and tools. It should be treated as the authoritative event
record; normalized fields are convenience fields for initial analysis.

## 10. Normalize an Existing OpenHands Run

If inference has already completed, the same script can normalize an existing
OpenHands JSONL without making new model calls:

```bash
python3 run_swebench_runtime_100.py \
  --benchmarks-dir ./openhands-benchmarks \
  --llm-config ./openhands-benchmarks/.llm_config/my_model.json \
  --raw-output-jsonl /path/to/output.jsonl \
  --output-dir runtime_data/swebench_100_existing
```

In this mode, `--benchmarks-dir` and `--llm-config` are accepted for a
consistent interface but are not used to launch inference.

## 11. Resume and Failure Handling

OpenHands can resume a run when the same structured output directory is reused;
completed instances are skipped by the official runner. Preserve the original
output directory if the run is interrupted.

Before retrying, inspect:

- OpenHands console output
- Per-instance errors in the raw `output.jsonl`
- The normalized `error` field
- `failed_event_count`
- Docker disk and memory usage

Do not change the output directory if the intention is to resume the same run.
Use a new output directory when changing the model or important inference
settings.

## 12. Recommended Initial Analysis

First compute the following descriptive statistics:

- Number of successful and failed instances
- Events and tool calls per workflow
- Runtime duration per workflow
- Frequency of inspect, edit, shell, and test operations
- Failure and retry frequency
- Distance from a failed event to the next recovery action
- Number and duration of repeated test-edit-test patterns
- Future tool/LLM demand over horizons 1, 2, and 3

For dynamic-DAG research, distinguish three levels of information:

1. **Observed runtime events:** actual tool and agent events from OpenHands.
2. **Inferred dependencies:** edges reconstructed from file reads, writes,
   tests, failures, and timing.
3. **Ground-truth control flow:** explicit orchestrator events such as node
   creation, edge creation, branch resolution, and subgraph expansion.

The first level is collected by this script. The second level can be produced
by an updated trace-to-DAG converter. The third level requires adding event
instrumentation to the orchestrator.

## 13. Suggested Event Instrumentation

If OpenHands or another workflow orchestrator is modified later, log events in
the following form:

```json
{"event": "node_created", "node_id": "n5", "timestamp": 123.4}
{"event": "edge_added", "source": "n2", "target": "n5", "timestamp": 123.5}
{"event": "branch_resolved", "branch": "recovery", "timestamp": 123.6}
{"event": "node_started", "node_id": "n5", "timestamp": 124.0}
{"event": "node_completed", "node_id": "n5", "timestamp": 130.2}
```

These events would allow direct evaluation of runtime DAG expansion, branch
prediction, prediction lead time, and speculative ILP overlap without relying
only on post-hoc dependency inference.

## 14. Security and Cost Notes

- Never print or commit the LLM API key.
- Start with two instances before launching 100.
- Set provider-side spending and rate limits.
- Keep raw traces private if prompts, source code, or model outputs are
  sensitive.
- Monitor Docker disk usage throughout the run.
- Avoid destructive Docker cleanup commands while other projects are using the
  same Docker installation.

