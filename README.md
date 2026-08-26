# Partial DAG Prediction for Dyserve

## Overview

This project studies whether historical agent traces can be used to predict
near-future workflow changes and prepare Dyserve's ILP computation before the
new DAG is fully available.

The earlier experiments treated every workflow as a linear sequence:

```text
inspect -> inspect -> edit -> test
```

N-gram and GRU models predicted the next one to three actions. This captures
local action patterns, but it does not represent dependencies, parallel
branches, joins, or critical-path changes.

The current version converts raw SWE-Trace trajectories into approximate DAGs.
It then uses the currently visible partial DAG to predict a bounded description
of the next three nodes:

- The number of future nodes of each semantic type
- The coarse topology of the future graph delta

The goal is not to generate an exact arbitrary DAG. The goal is to predict a
small amount of information that may help Dyserve prepare ILP templates,
constraints, and solver warm starts.

## Data Pipeline

```text
Raw Open-SWE-Traces trajectories
        |
        v
Extract tool calls, file accesses, commands, and results
        |
        v
Infer DAG nodes and dependency edges
        |
        v
Replay the partial DAG at each execution position
        |
        v
Predict future node counts and topology
```

The current experiment uses 1,000 OpenHands trajectories from Open-SWE-Traces.

## DAG Extraction

Each tool call becomes one graph node. A node records:

- Semantic node type
- Original tool name
- Resources read and written
- Success or failure status
- Discovery position
- Possible parallel-call group

An example trace is:

| Step | Action  | File or resource    | Result  |
| ---: | ------- | ------------------- | ------- |
|    1 | Inspect | `src/app.py`        | Success |
|    2 | Inspect | `tests/test_app.py` | Success |
|    3 | Edit    | `src/app.py`        | Success |
|    4 | Test    | Test suite          | Failed  |
|    5 | Inspect | `src/app.py`        | Success |
|    6 | Edit    | `src/app.py`        | Success |
|    7 | Test    | Test suite          | Success |

Open-SWE-Traces does not provide ground-truth workflow edges. Dependencies are
therefore inferred using resource flow and execution events.

| Relationship        | Meaning                                           | Confidence |
| ------------------- | ------------------------------------------------- | ---------: |
| Write-to-read       | A later operation reads a previously written file |       0.95 |
| Write-to-write      | Two operations modify the same file               |       0.90 |
| Read-to-write       | A file is inspected before it is modified         |       0.80 |
| Failure-to-recovery | A failed operation is followed by recovery work   |       0.70 |
| Write-to-test       | A test may validate recent code changes           |       0.65 |
| Temporal control    | Operations occur in consecutive agent turns       |       0.35 |

Low-confidence edges can be removed during training. The resulting graph is an
inferred tool and data-dependency DAG, not a ground-truth Dyserve orchestrator
DAG.

## Training Samples

Each workflow DAG is replayed in node-discovery order. At execution position
`t`, the model sees only nodes and edges that have appeared by `t`. Future data
is used only to create the target.

With a prediction horizon of three, one sample may contain:

```text
Observed partial DAG:
  two inspect nodes, one edit node, two edges, depth two

Future node counts:
  inspect = 1
  edit = 1
  test = 1
  all other types = 0

Future topology:
  chain
```

The 1,000 trajectories produce:

| Split      | Workflows | Prediction positions |
| ---------- | --------: | -------------------: |
| Train      |       680 |               39,629 |
| Validation |       120 |                7,174 |
| Test       |       200 |               11,858 |

The split is performed by workflow instance. Prediction positions from the same
task cannot appear in both training and testing.

## Bounded Multi-Task Prediction

The current model does not combine every possible future graph into one growing
class vocabulary. It uses fixed, separate prediction tasks.

### Future Node Counts

One classifier is trained for each semantic node type. It predicts how many
times the type appears in the next three positions:

```text
0 / 1 / 2 / 3
```

The current node types are:

```text
inspect, edit, shell, test, reason, web, other
```

### Future Topology

A separate classifier predicts one coarse topology class:

```text
none
independent
chain
branch
join
branch_join
```

This bounded output space remains stable when the dataset grows from 100 to
1,000 or 10,000 traces.

## Model Features

CatBoost is currently used because the partial DAG is summarized into mixed
categorical and numerical features:

- Recent semantic action history
- Latest tool and execution status
- Number of visible nodes and edges
- Maximum graph depth
- Branch and join counts
- Maximum fan-in and fan-out
- Counts of different node types
- Failure count and recent result size
- Programming language

CatBoost is used as a stable and interpretable baseline. A GNN or Graph
Transformer can be tested later when more reliable or ground-truth workflow
graphs are available.

## Feature Ablation

Three input modes are evaluated using the same split and prediction targets:

- **Sequence:** recent execution history only
- **Graph:** partial-DAG statistics only
- **Combined:** sequence and partial-DAG features together

This comparison tests whether DAG information adds value beyond the linear
execution sequence.

## Results on 1,000 Traces

### Topology Prediction

| Input             |   Accuracy |   Macro-F1 | Top-3 coverage |
| ----------------- | ---------: | ---------: | -------------: |
| Majority baseline |     30.31% |          - |              - |
| Sequence          |     45.59% |     31.19% |         88.26% |
| Graph             |     45.27% |     31.08% |         88.72% |
| Combined          | **47.74%** | **34.51%** |     **89.43%** |

Combined improves topology accuracy by 2.15 percentage points over Sequence and
17.43 points over the majority baseline.

The combined model performs best on `branch_join` and `independent`. The rare
`branch` class is not correctly detected. Many `branch` and `join` examples are
currently classified as `branch_join`.

### Node Presence Prediction

| Input    | Presence Micro-F1 | Presence Macro-F1 | Exact count vector |
| -------- | ----------------: | ----------------: | -----------------: |
| Sequence |            61.11% |            29.91% |             20.35% |
| Graph    |            58.36% |            26.33% |             18.23% |
| Combined |        **63.21%** |        **31.68%** |         **20.68%** |

Combined again performs best, showing that graph features provide some
additional information when combined with execution history.

For the Combined model:

| Node type | Precision | Recall |     F1 |  AUPRC |
| --------- | --------: | -----: | -----: | -----: |
| Inspect   |    83.07% | 80.59% | 81.81% | 90.76% |
| Edit      |    72.54% | 40.29% | 51.81% | 66.16% |
| Shell     |    70.48% | 44.60% | 54.63% | 68.58% |
| Test      |    67.70% | 17.32% | 27.58% | 58.02% |
| Reason    |        0% |     0% |     0% | 14.05% |

Inspect is relatively easy to predict. Edit and shell have reasonable precision
but still miss many events. Test recall is low. Reason is rare, and the default
decision rule always selects a zero count even though its probability ranking is
better than random.

### Strict Complete Match

| Metric                               | Majority baseline | Combined |
| ------------------------------------ | ----------------: | -------: |
| All node counts correct              |        **26.96%** |   20.68% |
| All node counts and topology correct |        **14.62%** |   12.88% |

The majority baseline is stronger on strict complete matching because it always
predicts the most common count pattern. The learned model attempts to detect
more edit, shell, and test events. This improves event F1 and topology prediction
but introduces more small count errors.

For the dynamic DAG task, the current main metrics are:

- Topology Macro-F1
- Node Presence Micro/Macro-F1
- Per-node Precision, Recall, F1, and AUPRC

Complete exact matching is retained as a strict secondary metric.

## What the Results Show

The current results suggest:

1. Recent execution history is still the strongest single source of information.
2. Partial-DAG statistics are not strong enough by themselves.
3. Combining sequence and graph features produces the best overall results.
4. The improvement from DAG information is positive but still limited.
5. Rare branches and rare node types remain difficult to predict.

More random seeds are needed to confirm that the Combined improvement is
stable.

## How This May Help Dyserve

The current model is not accurate enough to generate and directly execute a
complete future DAG. A predicted solution should not replace the real ILP plan
before the actual DAG is known.

The model may still support low-risk preparation:

- Preload model and verifier profiles
- Prepare candidate variables and shared constraints
- Build a small number of topology templates
- Copy the current solution as a possible MIP warm start
- Prepare an incremental or residual ILP problem
- Start speculative preparation only for high-confidence predictions

A possible integration is:

```text
Current partial DAG
        |
        v
Combined predictor
        |
        v
Top candidate structures and node probabilities
        |
        v
Prepare ILP templates and warm starts
        |
        v
Real DAG becomes available
        |
        v
Verify prediction and reuse matching work
```

The real DAG must always be checked before a plan is committed. Incorrect
candidates are discarded, and Dyserve follows the normal compilation path.

This approach aims to hide part of the ILP setup and solve time rather than
remove the entire compilation cost.

## Current Limitations

- The DAG edges are inferred rather than exported from a real orchestrator.
- True logical dependencies and parallel execution cannot always be recovered.
- The `branch` topology is rare and is currently never selected.
- Count classifiers prefer zero for rare node types.
- Some termination or unknown nodes may still be grouped under `other`.
- The current experiment does not measure real ILP reuse or latency reduction.

## Next Steps

1. Add class weights for rare topology classes.
2. Select a separate presence threshold for each node type using validation data.
3. Separate node presence prediction from positive count prediction.
4. Inspect and separate termination and unknown node types.
5. Repeat experiments with multiple random seeds.
6. Complete the 10,000-trace experiment.
7. Export ground-truth DAG events from a workflow orchestrator.
8. Measure prediction lead time, ILP reuse rate, hidden solver time, discarded
   speculation, and end-to-end Dyserve latency.

## Summary

The project now predicts a bounded description of future partial-DAG changes
instead of only the next action in a linear sequence. On 1,000 traces, combining
sequence and graph features gives the best topology and node-presence results.
The graph features provide a small positive improvement, but rare events and
exact node counts remain difficult. The current model is most suitable for
preparing ILP templates and warm starts, not for directly committing a predicted
runtime plan.