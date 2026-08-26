# Dynamic DAG Prediction for Dyserve

## Overview

This project studies how to predict changes in an agent workflow DAG so that
Dyserve can prepare or run the ILP solver before the new workflow structure is
fully available.

The original experiments represented each workflow as a linear sequence of
semantic actions:

```text
inspect -> inspect -> edit -> test
```

N-gram and GRU models were used to predict the next one to three actions. This
captures local execution patterns, but it does not show dependencies, parallel
branches, joins, or changes to the critical path. Predicting the next action is
therefore not enough to determine whether the Dyserve ILP solution needs to be
updated.

The current version extends the prediction setup from a linear sequence to an
inferred partial DAG. It uses raw SWE-Trace tool calls, file accesses, test
results, and recovery actions to build an approximate dependency graph. The
model then predicts what kinds of nodes and structures may appear in the next
few steps.

## Current Workflow

```text
Raw Open-SWE-Traces trajectories
        |
        v
Extract tool calls, files, commands, and results
        |
        v
Infer DAG nodes and dependency edges
        |
        v
Replay the partial DAG at each execution position
        |
        v
Train and evaluate the future DAG-delta predictor
```

## Data

The experiments use OpenHands trajectories from Open-SWE-Traces. The raw data
contains:

- Agent and tool messages
- Tool names and arguments
- File paths and shell commands
- Tool outputs and test results
- Programming language and repository metadata
- Task outcome

The old normalized dataset only kept semantic node labels such as `inspect`,
`edit`, `shell`, and `test`. The new pipeline uses the raw trajectories because
file paths and tool results are needed to infer dependencies.

An example trajectory is:

| Step | Action  | File or resource    | Result  |
| ---: | ------- | ------------------- | ------- |
|    1 | Inspect | `src/app.py`        | Success |
|    2 | Inspect | `tests/test_app.py` | Success |
|    3 | Edit    | `src/app.py`        | Success |
|    4 | Test    | Test suite          | Failed  |
|    5 | Inspect | `src/app.py`        | Success |
|    6 | Edit    | `src/app.py`        | Success |
|    7 | Test    | Test suite          | Success |

## DAG Construction

Each tool call becomes one graph node. A node records:

- Semantic node type
- Original tool name
- Resources read and written
- Execution status
- Discovery position
- Possible parallel-call group

Since Open-SWE-Traces does not contain ground-truth workflow edges, dependencies
are inferred using simple rules.

| Relationship        | Meaning                                           | Confidence |
| ------------------- | ------------------------------------------------- | ---------: |
| Write-to-read       | A later operation reads a previously written file |       0.95 |
| Write-to-write      | Two operations modify the same file               |       0.90 |
| Read-to-write       | A file is inspected before it is modified         |       0.80 |
| Failure-to-recovery | A failed operation is followed by recovery work   |       0.70 |
| Write-to-test       | A test may validate recent code changes           |       0.65 |
| Temporal control    | Operations occur in consecutive agent turns       |       0.35 |

For example:

```text
Inspect file -> Edit file -> Test
                              |
                              v failure
                    Inspect again -> Edit again -> Test again
```

Low-confidence edges can be excluded during training. The resulting graph is an
inferred tool and data-dependency DAG, not a ground-truth Dyserve orchestrator
DAG.

## Prediction Task

Each workflow DAG is replayed one node at a time. At every execution position,
the model only sees the currently visible nodes and edges. Future graph data is
used only to create the prediction label.

The current prediction horizon is three nodes. The target includes:

- Future semantic node types
- Number of each future node type
- Coarse future topology

The topology classes are:

- `none`
- `independent`
- `chain`
- `branch`
- `join`
- `branch_join`

An example target is:

```text
types=edit:1+inspect:1+test:1|topology=chain
```

This means that the next three nodes contain one `edit`, one `inspect`, and one
`test`, and that the inferred future structure is a chain.

## Model

CatBoost is currently used as the prediction model. The raw graph is not passed
directly to CatBoost. Instead, the current partial DAG is summarized into
categorical and numerical features.

The input features include:

- Recent semantic action history
- Latest tool and execution status
- Number of visible nodes and edges
- Maximum graph depth
- Branch and join counts
- Maximum fan-in and fan-out
- Counts of `inspect`, `edit`, `shell`, `test`, and `reason` nodes
- Number of failed calls
- Recent tool-result size
- Programming language

CatBoost is used as an initial baseline because these inputs are mixed tabular
features. It is easier to train and inspect than a GNN, especially while the DAG
edges are still inferred and may contain noise. A GNN or Graph Transformer can
be tested later when ground-truth workflow graphs become available.

## Feature Ablation

The experiments compare three feature modes using the same data split and
prediction target:

- **Sequence:** recent execution history only
- **Graph:** partial-DAG statistics only
- **Combined:** sequence and partial-DAG features together

This comparison tests whether graph information adds useful information beyond
the original linear sequence.

## Preliminary Results

The first pipeline test used 100 trajectories. The test split contained 1,207
partial-DAG prediction positions and 38 future-delta classes.

| Feature mode | Exact accuracy | Top-3 coverage |
| ------------ | -------------: | -------------: |
| Sequence     |         28.00% |         51.37% |
| Graph        |         27.17% |         47.06% |
| Combined     |         27.75% |         50.95% |

Exact accuracy requires both the future node-type counts and topology to be
correct. It is therefore much stricter than next-node accuracy. With 38 classes,
uniform random accuracy is approximately 2.63%.

The three modes currently have similar results. Since this experiment only uses
100 trajectories, it is intended to confirm that the full data and training
pipeline works. It is too small to determine whether DAG features provide a
consistent improvement.

## Running Experiments

The 1,000- and 10,000-trace experiments are still running. They take longer
because:

- Each trace produces many partial-DAG training samples
- Graph statistics are calculated at every execution position
- The model predicts 38 future-DAG classes
- Sequence, graph, and combined modes are trained separately

The 1,000-trace experiment will provide a more stable comparison among the three
feature modes. The 10,000-trace experiment will be used for the larger final
evaluation.

## Current Limitations

- The DAG is inferred from file operations and tool results rather than exported
  from the real workflow orchestrator.
- Logical dependencies, true parallelism, and branch conditions cannot always
  be recovered from SWE-Trace.
- The current target combines node types, counts, and topology into 38 classes.
  A small count or topology error makes the entire prediction incorrect.
- The current evaluation does not yet measure whether the predicted DAG change
  actually changes the ILP solution.
- End-to-end Dyserve latency and speculative ILP reuse have not yet been
  evaluated.

## Next Steps

1. Complete the 1,000- and 10,000-trace experiments.
2. Cache partial-DAG statistics to reduce repeated feature computation.
3. Evaluate multiple random seeds.
4. Add majority-class and random baselines.
5. Split the current target into three tasks:
   - Future node-type multi-label prediction
   - Topology classification
   - Node-count prediction
6. Compare different edge-confidence thresholds.
7. Collect ground-truth DAG events from a workflow orchestrator.
8. Add system-level labels:
   - Whether the ILP solution changes
   - Whether the previous solution is reusable
   - Available prediction lead time
   - Solver latency hidden by speculative execution
9. Compare sequence prediction, DAG prediction, always-speculative execution,
   and an oracle policy in Dyserve.

## Current Summary

The project has moved from predicting the next action in a linear sequence to
predicting changes in an inferred partial DAG. Raw SWE-Trace information is used
to build dependency graphs, and CatBoost models compare sequence features, graph
features, and both together. The 100-trace experiment confirms that the pipeline
works. Larger experiments are currently running to determine whether DAG
features improve prediction and can eventually support speculative ILP execution
in Dyserve.