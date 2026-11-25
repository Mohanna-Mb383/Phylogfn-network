# Validation and repair history

## Old failure

The old 300-epoch DS1 KNN run appeared to complete but stopped changing weights after approximately epoch 200. Archived checks reported all 137 model-state tensors unchanged from checkpoint 200 through 299, and Adam step counters ceased advancing at 40,056. The logged loss stayed finite while gradient-related optimizer checks rejected further updates. The historical MLL (`−6723.93 ± 5.20`) is therefore not suitable as evidence for a KNN accuracy gain.

## Repaired code

The run used a separate repaired copy of the historical KNN fork, not an assumed replacement of the public repository. Changes described in diagnostic records:

1. Clamp the raw scale sent to exponential transformations in continuous Gaussian heads, avoiding extremely large scales and non-finite gradients. The observed code included `clamp(max=20.0)` in both relevant files.
2. Add `UpdateMonitor` with per-epoch accepted/rejected counts and fail-fast thresholds (five consecutive rejections or >10% rejected in an epoch for the smoke/full configuration supplied).
3. Add a guard against an incompatible combination of action temperature and KNN sampling/log-probability treatment.
4. Implement a vectorized/batched feature path, enabled by `PHYLOGFN_FAST_FEATURES=on`.

A non-finite-gradient mechanism was reproduced in published-code lineage and repaired; the exact original failure source inside the historical KNN fork was not exhaustively proven by the standalone reproduced mechanism. The real full-run result below is the strongest operational validation of avoiding the freeze.

## Diagnostic results from vm03

| Diagnostic | Observed outcome |
|---|---|
| 50 Adam updates after repair | 50/50 accepted, 0 rejected; all parameters finite |
| Six-epoch smoke test | 6/6 epochs; 60/60 updates accepted, 0 rejected |
| Repaired vs vectorized CPU fingerprints | `EQUIVALENT` with zero differences in tested actions, probabilities, loss and gradients |
| Number of gradient entries compared | 3,861,571 |
| KNN nearest-neighbour selection unit check | PASSED; exact pairs agreed with reference sorting on 18 synthetic normalized embeddings |
| New DS1 full training | **300/300 epochs; 60,000/60,000 accepted; 0 rejected** |

The fingerprint test covered 3 batches × 16 trajectories = 48 trajectories, including R=1, R=2, all four merge cases and replay. STOP, empty-candidate and some other branches were not covered by that particular comparison.

## Why the distinction matters

**Optimizer stability** is demonstrated by zero rejected updates and advancing training through the full 300 epochs. **Inference accuracy, convergence and KNN efficiency** are distinct scientific questions and remain open pending matched evaluation.
