# DS1 — KNN k=5, 300 epochs, repaired/vectorized PhyloGFN-Net

**Experiment status:** training completed; all optimizer updates accepted. **Inference benefit and speedup not yet established.**

> **Evidence boundary:** This folder was prepared from the recorded `vm03` terminal outputs supplied by the researcher. The full `history.pkl`, `train.log`, `config.yaml` and model checkpoints were not accessible while creating this initial package. Run [`tools/collect_vm03.py`](tools/collect_vm03.py) on `vm03` before publication to add the exact run artifacts. Do **not** describe this folder as an independently reproduced evaluation.

## 1. Executive results

| Quantity | Verified reported value |
|---|---:|
| Dataset | DS1 (27 taxa; 1,949 sites) |
| Model | Level-1 phylogenetic network GFlowNet |
| Objective | Trajectory Balance (TB) |
| kNN | Enabled, `k=5`, `EXACT`, `COSINE`, 4-mer embeddings |
| Ancestor embedding propagation | `DIRICHLET` (branch-length-weighted) |
| Training | 300 epochs × 200 optimizer steps |
| Accepted optimizer updates | **60,000** |
| Rejected optimizer updates | **0** |
| Final marginal log-likelihood estimate | **−7153.974 ± 4.074** |
| Final MLL evaluation | 3 independent estimates, 1,024 trajectories each |
| Last training epoch (299) MLL / Pearson R | −7154.572 / 0.783 |
| Epoch 299 R=0 / R=1 / R=2 | 831 / 183 / 10 (out of 1,024) |

**Do not confuse** the final repeated MLL summary (`−7153.974 ± 4.074`) with the separate epoch-299 logging value (`−7154.572`). They come from different reporting operations.

## 2. What changed compared with the old code?

- **Without KNN:** all legal lineage merges are candidates for the policy. With 27 open lineages, there are `27 × 26 / 2 = 351` potential merge pairs before legality restrictions.
- **With KNN:** for each open lineage, find its 5 nearest embedding neighbours using cosine distance. Form the undirected *union* of neighbour pairs. The main candidate policy is restricted to this subset, while an epsilon-uniform legal-action component keeps other legal actions reachable. Reticulation candidates and applicable STOP actions are treated separately.
- **After a merge:** the ancestor's embedding is a branch-length-weighted combination of the children, normalized for cosine distance. In `EXACT` mode pairwise distances are recomputed at each construction step.
- **Limitations:** the cost of distance computation is still quadratic in the number of open lineages; and whether the model only scores KNN candidates (rather than scoring all and then masking) must be verified in the exact run source. KNN acceleration is a hypothesis, not a measured success.

More: [METHOD.md](METHOD.md).

## 3. Why the original KNN run was not valid

The historical 300-epoch run continued to produce logs but **stopped updating model parameters after approximately epoch 200**. Model checkpoints 200–299 and optimizer step counters showed frozen parameters. A plausible continuous-Gaussian scale overflow mechanism was reproduced and the code was repaired with scale caps and a fail-fast update monitor. See [VALIDATION.md](VALIDATION.md).

The historical KNN MLL of `−6723.93 ± 5.20` **must not be presented as a demonstrated KNN gain**, because the run was frozen and the evaluation was not a controlled comparison.

## 4. Files in this folder

| Path | Meaning |
|---|---|
| [`data/summary.json`](data/summary.json) | Confirmed headline metrics |
| [`data/epochs_269_299.csv`](data/epochs_269_299.csv) | Exact values transcribed from terminal logs, last 31 epochs only |
| [`data/historical_comparison.csv`](data/historical_comparison.csv) | Historical reference numbers with caveats |
| [`figures/`](figures/) | Plots of the **last 31** training epochs; not full 300-epoch curves |
| [`RESULTS.md`](RESULTS.md) | Interpretation and remaining comparisons |
| [`VALIDATION.md`](VALIDATION.md) | Repair, numerical diagnostics, and acceptance counts |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Run paths, command, and artifact collection |
| [`LIMITATIONS.md`](LIMITATIONS.md) | Explicit unresolved claims and risks |
| [`raw/`](raw/) | Console excerpts given in chat; extended by the server export |
| [`tools/collect_vm03.py`](tools/collect_vm03.py) | Read-only export of the real server run into this folder |

## 5. Interpretation

The new run completed 300 epochs with **60,000 accepted optimizer updates and zero rejected updates**. This directly addresses the old optimizer freeze. It does **not** prove convergence, correct posterior calibration, or superior MLL/speed relative to the no-KNN baseline.

At epoch 299, 831/1,024 sampled objects (`81.15%`) had no reticulation; 183 (`17.87%`) had one; 10 (`0.98%`) had two. The final training segment contains visible fluctuations in MLL, Pearson R, and reticulation counts: see [RESULTS.md](RESULTS.md).

## 6. External references

- [Project repository](https://github.com/Mohanna-Mb383/Phylogfn-network)
- [Original PhyloGFN reference implementation](https://github.com/zmy1116/phylogfn)

**Status date:** 2026-11-25. **No official GitHub upload has occurred simply by preparing this folder.**
