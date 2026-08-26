# Dynamic DAG Prediction for Proactive ILP Execution

This project investigates whether the near-future structure of a dynamic agent
workflow can be predicted during execution and used to start Integer Linear
Programming (ILP) tasks early.

Agent workflows can be represented as dynamically evolving directed acyclic
graphs (DAGs). The complete DAG is often unavailable at the beginning because
new nodes and dependencies are generated from agent decisions, tool outputs, and
intermediate workflow state. An ILP task would normally start only after its node
and required inputs become available.

The goal of this work is to predict upcoming workflow nodes within a limited
depth. If an ILP-related node is predicted with sufficient confidence, the
runtime may start the ILP solver speculatively and overlap its execution with
earlier workflow operations.

## Proposed Workflow

1. Observe the currently available portion of the workflow DAG.
2. Predict the node types likely to appear within the next few steps.
3. Check whether the predicted future contains an ILP-related node.
4. Start or prepare the ILP computation when prediction confidence is high.
5. Reuse the result if the prediction is correct.
6. Cancel or discard the speculative computation if the workflow changes.

The current experiments focus on the prediction component. End-to-end ILP
integration and latency evaluation are planned as the next stage.

## Current Prediction Formulation

The current implementation represents a workflow as the sequence in which its
nodes become visible or execute. Given an observed history

```text
x_1, x_2, ..., x_t
```

the model predicts the next `H` semantic node types:

```text
x_(t+1), x_(t+2), ..., x_(t+H)
```

The evaluated prediction horizons are:

- `H = 1`: predict the next node
- `H = 2`: predict the next two nodes
- `H = 3`: predict the next three nodes

The semantic node taxonomy contains:

- `inspect`
- `edit`
- `shell`
- `test`
- `reason`
- `fetch`
- `STOP`

This is currently a sequence-based approximation of dynamic DAG evolution. It
does not yet explicitly model branches, joins, parallel nodes, or graph edges.

## Dataset

The experiments use approximately 10,000 SWE-Trace/OpenHands workflow traces.

| Property | Value |
|---|---:|
| Workflow runs | 10,000 |
| Total nodes | 596,745 |
| Mean actions per workflow | 58.7 |
| Median actions per workflow | 54 |
| Test workflows | 2,000 |
| Test prediction samples | 116,677 |

The dataset is imbalanced. The `inspect` type occurs 318,352 times and accounts
for approximately 53% of all nodes. Overall accuracy is therefore strongly
influenced by frequent node types, so Macro-F1 is also used to evaluate balanced
performance across classes.

## Prediction Methods

### N-gram

The N-gram predictor estimates future nodes using transition statistics from the
most recent node types. The evaluated context windows are:

```text
0, 1, 2, 3, 5, 8
```

Window `0` is a context-free baseline. The N-gram method is inexpensive,
interpretable, and suitable for online prediction, but large context windows can
suffer from sparse observations.

### GRU

The GRU predictor learns a recurrent representation of the observed workflow
history. The evaluated history lengths are:

```text
1, 2, 3, 5, 8, 16
```

The primary training configuration is:

| Parameter | Value |
|---|---:|
| Embedding dimension | 32 |
| Hidden size | 64 |
| Dropout | 0.1 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Maximum epochs | 20 |
| Early-stopping patience | 3 |
| Top-k | 4 |
| Gate confidence threshold | 0.6 |

Class weights are enabled to reduce the effect of the imbalanced node
distribution.

## Evaluation Metrics

- **Next-node accuracy:** whether the first predicted node is correct.
- **Macro-F1:** the mean of the per-class F1 scores, with equal weight assigned
  to each node type.
- **Exact-sequence accuracy:** whether every node in the predicted future
  sequence is correct and in the correct order.
- **Mean correct-prefix length:** the number of consecutive correct nodes from
  the beginning of the predicted sequence.
- **Top-4 coverage:** whether the true future sequence appears among four
  prediction candidates.
- **Gate trigger rate:** the fraction of predictions whose confidence reaches
  the speculative-execution threshold.
- **Gate precision:** exact-sequence accuracy among predictions that trigger the
  gate.

Gate precision and trigger rate are particularly important for ILP execution. A
high trigger rate creates more opportunities to hide latency, while high
precision limits wasted solver computation.

## Main Results

The following table compares representative results from N-gram with context
window 5 and GRU with history length 16.

| Horizon | Model | Configuration | Next accuracy | Macro-F1 | Exact sequence | Top-4 coverage | Gate rate | Gate precision |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | N-gram | Window 5 | **58.41%** | 29.56% | **58.41%** | 97.66% | 38.73% | **80.81%** |
| 1 | GRU | History 16 | 56.35% | **37.44%** | 56.35% | **97.83%** | 38.96% | 79.02% |
| 2 | N-gram | Window 5 | **57.88%** | 28.32% | **39.42%** | **66.03%** | 24.66% | 76.17% |
| 2 | GRU | History 16 | 56.38% | **37.48%** | 35.04% | 60.05% | 24.29% | **79.97%** |
| 3 | N-gram | Window 5 | **57.42%** | 28.99% | **29.44%** | **44.75%** | 22.93% | 65.09% |
| 3 | GRU | History 16 | 56.13% | **37.39%** | 25.54% | 37.51% | 18.93% | **77.84%** |

The context-free exact-sequence baselines are:

| Horizon | Exact-sequence baseline |
|---:|---:|
| 1 | 52.14% |
| 2 | 35.96% |
| 3 | 27.32% |

## Findings

### Local workflow patterns are predictive

Both models improve next-node prediction over the context-free baseline. This
indicates that recent workflow history contains information that can support
proactive execution.

### N-gram currently provides the best sequence prediction

N-gram with window 5 achieves the highest exact-sequence accuracy for all three
horizons:

- Horizon 1: 58.41%, an improvement of 6.28 percentage points over baseline
- Horizon 2: 39.42%, an improvement of 3.46 percentage points
- Horizon 3: 29.44%, an improvement of 2.13 percentage points

Performance decreases at window 8, likely because long exact contexts are sparse
even with 10,000 training traces. A five-node window currently provides the best
balance between contextual information and statistical reliability.

### GRU better handles imbalanced classes

GRU with history length 16 achieves a Macro-F1 of approximately 37.4%, compared
with roughly 29--30% for N-gram. Longer histories improve Macro-F1 more than
overall accuracy, suggesting that the GRU uses additional context mainly to
recognize less frequent node types.

This property may be important if ILP-related nodes are uncommon. Per-class
precision and recall are still required to confirm performance on the specific
node types used to trigger ILP.

### GRU multi-step prediction requires improvement

The current GRU does not outperform the context-free exact-sequence baseline for
horizons 2 and 3:

| Horizon | GRU exact sequence | Baseline |
|---:|---:|---:|
| 2 | 35.04% | 35.96% |
| 3 | 25.54% | 27.32% |

Possible causes include autoregressive error propagation, class-weighted
training that does not directly optimize sequence accuracy, and the limited input
representation. The model currently observes node types but not DAG edges, task
content, tool arguments, or intermediate results.

### Confidence filtering is promising

Despite weaker multi-step sequence accuracy, GRU with history length 16 maintains
approximately 78--80% gate precision across the three horizons. Its trigger rate
decreases from 38.96% at horizon 1 to 18.93% at horizon 3.

This suggests that the GRU can identify a smaller subset of relatively safe
predictions. N-gram remains the stronger general predictor, while GRU may be
useful as a conservative gate or as a complementary predictor for rare classes.

## Implications for ILP Execution

Prediction accuracy alone does not determine whether speculative ILP execution
is beneficial. The runtime should consider the expected benefit:

```text
expected benefit
  = P(correct trigger) * saved latency
  - P(incorrect trigger) * wasted computation
  - prediction overhead
```

If an incorrect ILP launch is expensive, the runtime should use a conservative,
high-precision threshold. If spare compute resources are available, a lower
threshold may provide more opportunities to overlap ILP execution with the
workflow.

It may also be unnecessary to predict the complete future sequence. The final
system primarily needs to determine whether an ILP-relevant node will appear and
whether enough information is available to start its computation. A
target-specific event prediction objective may therefore be more suitable than
requiring every future node to be correct.

## Current Recommendation

- Use **N-gram with window 5** as the initial online predictor.
- Prioritize **horizon 1** for the first ILP integration.
- Evaluate **horizon 2** with confidence filtering.
- Treat **horizon 3** as experimental until prediction quality improves.
- Continue investigating **GRU with history length 16** for rare-node detection
  and high-confidence gating.

## Limitations

- The current model predicts node-type sequences rather than explicit graph
  structure.
- The dataset is highly imbalanced.
- Overall metrics may not reflect performance on ILP-specific nodes.
- A fixed confidence threshold of 0.6 may not be equally appropriate for every
  prediction horizon.
- End-to-end ILP latency and wasted speculative computation have not yet been
  measured.

## Next Steps

1. Identify the workflow nodes and states that correspond to ILP tasks.
2. Report per-class precision, recall, F1, and confusion matrices.
3. Predict whether an ILP-related node appears within the next `H` steps.
4. Calibrate confidence thresholds separately for each horizon.
5. Produce precision-versus-coverage and cost-versus-benefit curves.
6. Measure how much earlier ILP execution can begin.
7. Measure wasted computation from incorrect predictions.
8. Compare end-to-end workflow latency with and without proactive ILP execution.
9. Add DAG topology, task state, and node metadata to the learned predictor.
10. Investigate a hybrid N-gram and GRU prediction gate.

