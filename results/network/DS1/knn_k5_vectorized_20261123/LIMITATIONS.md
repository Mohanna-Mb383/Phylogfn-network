# Known limitations, scientific integrity, and pending work

1. **No matched no-KNN performance test.** The older timed comparison was taken under GPU contention; it cannot establish a speedup on the L40S.
2. **No matched accuracy evaluation.** Tree and network reference MLLs came from historical different runs. Different evaluation procedures or seeds can alter the estimate.
3. **Final MLL uncertainty is based on three repeats.** Report it as the supplied standard deviation; not a confidence interval and not necessarily the posterior accuracy uncertainty.
4. **Sampling distribution drift is visible.** Late-epoch R histograms change strongly. A zero rejected update count proves neither convergence nor calibration.
5. **Pearson R may use a moving evaluation set** in the relevant code family; it should not be interpreted as a directly comparable fixed-set curve without checking the evaluator.
6. **KNN does not guarantee exact posterior preservation** and may change the proposal; epsilon exploration keeps legal non-neighbour actions reachable, but this does not ensure zero bias or fast mixing.
7. **Exact KNN still computes all distances** per state. KNN-only masking may not eliminate expensive full pair scoring. Verify on exact source and GPU.
8. **Exact deployed source tree is not yet part of this package.** Add hashes and optionally a snapshot on vm03 via `tools/collect_vm03.py`.
9. **Only epochs 269–299 are initially included in tabular form.** The export tool fills in all epochs from the local `train.log` on vm03.
10. **The old failure mechanism was reproduced on related public code**, but the original fork's exact first non-finite operation was not definitively proven in the evidence available here.
11. **No statements about DS2–DS8, DB or WDB** follow from this DS1 TB/KNN run.
