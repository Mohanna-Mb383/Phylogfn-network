# DS1 kNN result report

## Completed training and MLL

| Item | Result | Meaning |
|---|---:|---|
| Completed epochs | 300/300 | Historical code used zero-based epochs 0–299 |
| Accepted updates | **60,000** | `history.pkl` totals supplied from the completed run |
| Rejected updates | **0** | No rejected optimizer updates registered |
| Final MLL estimate | **−7153.974** | Mean across 3 independent estimates |
| Final MLL s.d. | **4.074** | Standard deviation of the 3 estimates |
| Trajectories per estimate | 1,024 | 3 × 1,024 in final evaluation |
| Epoch 299 MLL | −7154.572 | Separate epoch-level evaluation statistic |
| Epoch 299 Pearson R | 0.783 | On that epoch's evaluated set |
| Epoch 299 R=0,1,2 | 831, 183, 10 | Distribution among 1,024 samples |

## Historical comparison: contextual only

| Model | DS1 MLL | Caveat |
|---|---:|---|
| Tree baseline | about −7108.47 | Different model family; historical figure |
| Network without KNN | about −7043.06 | Historical network reference; not matched to new run |
| Old network KNN=5 | −6723.93 ± 5.20 | Invalid for performance-gain claims: optimizer froze after about epoch 200 |
| Repaired/vectorized network KNN=5 | **−7153.974 ± 4.074** | Numerically stable complete training; evaluation matching still needed |

The difference between the **historical** network baseline and this run is `−110.914` nats (`−7153.974 − (−7043.06)`). This is **not** a defensible paired comparison of posterior quality without matching model settings, target density, evaluation samples, seeds, and estimator diagnostics such as effective sample size.

## Late-epoch variability (data shown here are only epochs 269–299)

See [`data/epochs_269_299.csv`](data/epochs_269_299.csv) and [`figures/`](figures/).

- MLL varies considerably in the displayed tail; a few epochs near −7084 to −7091 differ from many others near −7150.
- Pearson R also varies and is **not** uniformly close to 1. Its evaluation state set may change across epochs in this code lineage.
- The count of reticulation-free samples changes sharply at some epochs. For example, R=0 is 4 at epoch 280 but 831 at epoch 299 (each out of 1,024).
- The final training epoch samples are primarily R=0; this warrants examination of the prior, exploration schedule and learned output distribution.

Do not claim full posterior convergence solely from accepted updates or these metrics.

## Performance

A previous three-step pilot benchmark on a GPU with a competing `peakminer` process gave roughly 66.38 seconds per step without KNN and 67.28 seconds with KNN. These are **contended GPU pilot measurements**, not evidence of a KNN speedup or slowdown on an idle device. The server was later restarted and the new training completed on the available GPU, but no matched no-KNN speed comparison from that period has been supplied.

## Needed to complete the paper-quality evaluation

- Re-evaluate the new KNN and no-KNN checkpoints using the **same** target, number of trajectories, seeds and evaluation procedure, reporting MLL uncertainty and effective sample size (ESS).
- Benchmark per-step rollout, loss/backward, total step time and epoch-level evaluation on an **idle** L40S with a matched batch and configuration.
- Audit candidate-only model scoring to establish whether KNN reduces high-cost network compute or merely masks full pair logits.
- Compare topology/reticulation distributions, sampled network diversity, and other fidelity metrics with the no-KNN model.
- Preserve the full 300-epoch logs, training-history summaries and exact code version alongside these results.
