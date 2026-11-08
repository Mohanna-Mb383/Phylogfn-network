# N=4 Gate Experiment: Distributional Correctness of PhyloGFN-Net

## 1. Purpose

This experiment tests whether balance-aware GFlowNet objectives improve the **distributional correctness** of PhyloGFN-Net on a small phylogenetic-network problem for which a numerical reference posterior can be constructed.

The central question is:

> **Do Detailed Balance (DB) or Weighted Detailed Balance (WDB) make PhyloGFN-Net sample closer to the intended posterior than the existing Trajectory Balance (TB) baseline?**

This N=4 experiment is a **gate experiment**. It is deliberately small enough that the learned sampler can be compared directly with a numerical reference posterior over the complete enumerated network class.

---

## 2. Methods Compared

Four training objectives were evaluated.

| Method | Description |
|---|---|
| **TB** | Original Trajectory Balance baseline used by PhyloGFN-Net. |
| **DB** | Detailed Balance with a learned state-flow head \(F(s)\). |
| **WDB-logcount** | Weighted Detailed Balance with a tempered topological-descendant weight based on \(1/(1+\log \#D)\). |
| **WDB-count** | Weighted Detailed Balance using the direct topological-descendant proxy \(1/\#D\). |

For WDB, the trajectory loss is normalized by the sum of transition weights:

\[
L_{\mathrm{WDB}}(\tau)
=
\frac{
\sum_t
\gamma(s_t,s_{t+1})
\left[
\log
\frac{
F(s_t)P_F(s_{t+1}\mid s_t)
}{
F(s_{t+1})P_B(s_t\mid s_{t+1})
}
\right]^2
}{
\sum_t \gamma(s_t,s_{t+1})
}.
\]

The implementation uses a **network-specific topological descendant proxy**. It should not be described as the literal finite-descendant count from the original paper for the mixed discrete-continuous phylogenetic-network state space.

---

## 3. Experimental Setup

### Problem

- Number of taxa: **N = 4**
- Alignment length: **100 sites**
- Maximum reticulations: **R_MAX = 2**
- Enumerated graph topologies: **603**
- Variants: **4**
- Seeds per variant: **5**
- Total training runs: **20**
- Training epochs per run: **40**
- Updates per epoch: **50**

The four variants were run with seeds 1–5:

```text
TB               × 5 seeds
DB               × 5 seeds
WDB-logcount     × 5 seeds
WDB-count        × 5 seeds
```

### Numerical reference posterior

For the N=4 gate, all 603 allowed graph topologies are enumerated. Continuous branch-length and reticulation parameters are integrated numerically.

Reference construction used:

```text
4000 integration draws per graph
20000 samples for reference/sampler TV evaluation
```

The cached reference reported:

```text
integration floor:
TV(bootstrap reference, reference) = 0.1176 ± 0.0063

finite-sample floor:
TV(20000 draws from reference, reference) = 0.0549 ± 0.0022
```

These floors are important: the reference is a high-accuracy **numerical posterior**, not an analytically exact posterior.

### Hardware

The completed experiment was run on:

```text
GPU: NVIDIA L40S
GPU memory: approximately 46 GB
CUDA runtime reported by nvidia-smi: 12.4
```

Training and GPU-compatible evaluation used CUDA.

---

## 4. Evaluation Metrics

The experiment does not use MLL alone as evidence of distributional correctness.

### TV_lab

Total-variation distance between sampler frequencies and the **labelled target** used by the sampler.

Lower is better:

\[
TV_{\mathrm{lab}} \rightarrow 0.
\]

### TV_graph

Total-variation distance after correcting the slot-labelling multiplicity with \(2^{-R}\), compared with the posterior over graph topologies.

Lower is better:

\[
TV_{\mathrm{graph}} \rightarrow 0.
\]

### FCS

Flow Consistency Score, estimated from backward paths.

Lower is better:

\[
FCS \rightarrow 0.
\]

### Balance regression

For sampled terminal objects:

\[
\log p_T(x) = a \log R(x) + b.
\]

A balanced sampler should approach:

```text
slope a       -> 1
residual SD   -> 0
FCS           -> 0
```

### Time-averaged metric

Checkpoint-wise metrics are integrated over wall-clock training time. For fair comparison, all methods use the common horizon

```text
H = 600 s
```

which is the smallest final training time among the 20 runs.

The reported time-averaged metric is

\[
\frac{\mathrm{AUC}}{H}.
\]

Lower is better.

---

## 5. Main N=4 Results

### 5.1 Aggregate results across 5 seeds

Mean ± sample standard deviation:

| Method | Time-avg TV_lab ↓ | Time-avg TV_graph ↓ | Time-avg FCS ↓ | Final TV_lab ↓ | Final TV_graph ↓ | Final FCS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **TB** | **0.422 ± 0.036** | **0.437 ± 0.034** | **0.447 ± 0.037** | **0.280 ± 0.032** | **0.306 ± 0.040** | **0.309 ± 0.037** |
| DB | 0.771 ± 0.024 | 0.790 ± 0.024 | 0.655 ± 0.022 | 0.721 ± 0.046 | 0.725 ± 0.042 | 0.519 ± 0.053 |
| WDB-logcount | 0.767 ± 0.022 | 0.786 ± 0.020 | 0.678 ± 0.021 | 0.769 ± 0.037 | 0.774 ± 0.038 | 0.592 ± 0.058 |
| WDB-count | 0.816 ± 0.043 | 0.836 ± 0.043 | 0.703 ± 0.059 | 0.787 ± 0.094 | 0.809 ± 0.087 | 0.634 ± 0.113 |

### 5.2 Mean AUC over the common 600 s horizon

| Method | AUC(TV_lab) ↓ | AUC(TV_graph) ↓ | AUC(FCS) ↓ |
|---|---:|---:|---:|
| **TB** | **253.48 ± 21.43** | **262.48 ± 20.45** | **268.46 ± 22.17** |
| DB | 463.00 ± 14.34 | 474.50 ± 14.54 | 393.46 ± 12.99 |
| WDB-logcount | 460.28 ± 13.01 | 472.25 ± 11.79 | 406.85 ± 12.53 |
| WDB-count | 490.02 ± 26.12 | 502.28 ± 25.67 | 421.84 ± 35.20 |

TB has the lowest AUC for all three distributional metrics.

---

## 6. Per-Seed Results

### TV_lab

| Method | Seed | Time-avg | Final |
|---|---:|---:|---:|
| DB | 1 | 0.8058 | 0.6764 |
| DB | 2 | 0.7841 | 0.6883 |
| DB | 3 | 0.7533 | 0.7692 |
| DB | 4 | 0.7639 | 0.6981 |
| DB | 5 | 0.7480 | 0.7705 |
| **TB** | **1** | **0.4806** | **0.2377** |
| **TB** | **2** | **0.3920** | **0.2744** |
| **TB** | **3** | **0.4161** | **0.3265** |
| **TB** | **4** | **0.3956** | **0.2780** |
| **TB** | **5** | **0.4263** | **0.2814** |
| WDB-count | 1 | 0.7953 | 0.6904 |
| WDB-count | 2 | 0.8384 | 0.7591 |
| WDB-count | 3 | 0.8773 | 0.9430 |
| WDB-count | 4 | 0.8058 | 0.7882 |
| WDB-count | 5 | 0.7634 | 0.7543 |
| WDB-logcount | 1 | 0.7777 | 0.7673 |
| WDB-logcount | 2 | 0.7854 | 0.7060 |
| WDB-logcount | 3 | 0.7295 | 0.7974 |
| WDB-logcount | 4 | 0.7690 | 0.7911 |
| WDB-logcount | 5 | 0.7709 | 0.7832 |

### TV_graph

| Method | Seed | Time-avg | Final |
|---|---:|---:|---:|
| DB | 1 | 0.8242 | 0.6874 |
| DB | 2 | 0.8046 | 0.7014 |
| DB | 3 | 0.7739 | 0.7751 |
| DB | 4 | 0.7842 | 0.6957 |
| DB | 5 | 0.7641 | 0.7654 |
| **TB** | **1** | **0.4935** | **0.2564** |
| **TB** | **2** | **0.4097** | **0.2949** |
| **TB** | **3** | **0.4318** | **0.3681** |
| **TB** | **4** | **0.4111** | **0.3106** |
| **TB** | **5** | **0.4394** | **0.2999** |
| WDB-count | 1 | 0.8238 | 0.7253 |
| WDB-count | 2 | 0.8627 | 0.8074 |
| WDB-count | 3 | 0.8919 | 0.9531 |
| WDB-count | 4 | 0.8243 | 0.8000 |
| WDB-count | 5 | 0.7795 | 0.7609 |
| WDB-logcount | 1 | 0.7925 | 0.7753 |
| WDB-logcount | 2 | 0.8050 | 0.7095 |
| WDB-logcount | 3 | 0.7530 | 0.8086 |
| WDB-logcount | 4 | 0.7907 | 0.7916 |
| WDB-logcount | 5 | 0.7910 | 0.7853 |

### FCS

| Method | Seed | Time-avg | Final |
|---|---:|---:|---:|
| DB | 1 | 0.6731 | 0.4561 |
| DB | 2 | 0.6778 | 0.5677 |
| DB | 3 | 0.6581 | 0.4861 |
| DB | 4 | 0.6404 | 0.5810 |
| DB | 5 | 0.6267 | 0.5064 |
| **TB** | **1** | **0.5099** | **0.2840** |
| **TB** | **2** | **0.4242** | **0.3705** |
| **TB** | **3** | **0.4395** | **0.2784** |
| **TB** | **4** | **0.4168** | **0.2994** |
| **TB** | **5** | **0.4449** | **0.3132** |
| WDB-count | 1 | 0.7241 | 0.7397 |
| WDB-count | 2 | 0.7220 | 0.5961 |
| WDB-count | 3 | 0.6021 | 0.4563 |
| WDB-count | 4 | 0.7092 | 0.7071 |
| WDB-count | 5 | 0.7551 | 0.6719 |
| WDB-logcount | 1 | 0.7108 | 0.6855 |
| WDB-logcount | 2 | 0.6633 | 0.5333 |
| WDB-logcount | 3 | 0.6687 | 0.6021 |
| WDB-logcount | 4 | 0.6847 | 0.5641 |
| WDB-logcount | 5 | 0.6601 | 0.5740 |

---

## 7. Paired Statistical Comparisons

The comparison is paired by seed.

### TB vs DB

| Metric | Mean paired difference: TB − DB | Direction | Exact Wilcoxon p |
|---|---:|---:|---:|
| Time-avg TV_lab | **−0.3489** | TB better in **5/5** seeds | 0.062 |
| Time-avg TV_graph | **−0.3531** | TB better in **5/5** seeds | 0.062 |
| Time-avg FCS | **−0.2082** | TB better in **5/5** seeds | 0.062 |

The result is directionally completely consistent across seeds.

However:

> **Do not report this as p < 0.05 statistical significance.**

With only five non-zero paired observations, the two-sided exact Wilcoxon signed-rank test has very coarse resolution; when all five differences have the same sign, the smallest attainable two-sided exact p-value is approximately 0.0625.

### WDB vs DB

| Comparison | Metric | Mean paired difference | Better than DB in | Wilcoxon p |
|---|---|---:|---:|---:|
| WDB-count − DB | TV_lab | +0.0450 | 1/5 | 0.125 |
| WDB-count − DB | TV_graph | +0.0463 | 1/5 | 0.125 |
| WDB-count − DB | FCS | +0.0473 | 1/5 | 0.312 |
| WDB-logcount − DB | TV_lab | −0.0045 | 2/5 | 0.812 |
| WDB-logcount − DB | TV_graph | −0.0038 | 2/5 | 1.000 |
| WDB-logcount − DB | FCS | +0.0223 | 1/5 | 0.188 |

There is no evidence in this gate that either WDB variant improves distributional correctness relative to DB.

---

## 8. Detailed Diagnostic Example: TB Seed 1

The full FCS diagnostic for TB seed 1 gave:

```text
Policy set:
slope a              = 0.8100
Pearson r             = 0.8752
Spearman r            = 0.8875
residual SD           = 1.842 nats

Policy-FCS:
beta=2               = 0.1345
beta=4               = 0.2452
beta=8               = 0.3288
beta=16              = 0.3820
```

The sampler was therefore substantially better than DB/WDB in TV and FCS, but it was **not perfectly balanced**.

### K-convergence study

For 48 fixed policy networks:

```text
K=25:   slope=0.7000 ± 0.0012, residual SD=1.927 ± 0.002, FCS_8=0.3982 ± 0.0052
K=50:   slope=0.6979 ± 0.0010, residual SD=1.933 ± 0.003, FCS_8=0.3959 ± 0.0076
K=100:  slope=0.6991 ± 0.0003, residual SD=1.931 ± 0.002, FCS_8=0.3994 ± 0.0040
K=200:  slope=0.6987 ± 0.0007, residual SD=1.930 ± 0.002, FCS_8=0.4024 ± 0.0024
```

The diagnostic is stable over \(K=25\) to \(K=200\), indicating that the observed imbalance is not primarily caused by insufficient backward-path sampling.

### Residual screening

For TB seed 1:

```text
R                         eta^2 = 0.000
first RETICULATE step     eta^2 = 0.011
first closing-merge step  eta^2 = 0.002
3-node cycle              eta^2 = 0.002
```

None of these trajectory-level groupings explains a large fraction of residual variance in this run.

---

## 9. Main Scientific Finding

The N=4 result is a **negative result for DB/WDB and a positive result for the existing TB baseline**.

A defensible statement is:

> On the four-taxon numerical-reference problem, Trajectory Balance substantially outperformed Detailed Balance and both Weighted Detailed Balance variants in distributional accuracy. TB achieved lower time-averaged and final TV distances and lower FCS in all five seeds. WDB did not improve over DB. The TB-vs-DB direction was consistent in all five paired seeds, although the exact two-sided Wilcoxon test did not cross the conventional 0.05 threshold because only five paired seeds were available.

The result should **not** be stated as:

```text
"WDB improves PhyloGFN-Net."
```

The N=4 experiment shows the opposite under this configuration.

It should also **not** be generalized to all DB/WDB GFlowNets. This result is specific to:

- this phylogenetic-network MDP,
- this flow-head parameterization,
- these training budgets,
- this continuous-action model,
- and the network-specific topological descendant weighting proxy used here.

---

## 10. Gate Decision

The pre-registered purpose of N=4 is to test whether WDB improves distributional correctness relative to DB before expensive DS-scale experiments.

For N=4:

```text
WDB-count vs DB:
  no improvement

WDB-logcount vs DB:
  essentially tied in TV, worse in FCS

TB vs DB/WDB:
  substantially better
```

Therefore:

> **The WDB gate is not passed on N=4.**

This means the N=4 experiment does not justify proceeding directly to DS-scale WDB training.

---

## 11. Next Experiment: sim6

The next controlled gate is the N=6 simulated problem.

It compares:

```text
TB
DB
WDB-logcount
WDB-count
WDB-depth
WDB-logcount + Look-Ahead
```

The Look-Ahead method is **not part of the N=4 experiment**. It enters at sim6.

The primary sim6 endpoint is checkpoint-wise Policy-FCS against wall-clock time, supplemented by:

- slope \(a\),
- residual standard deviation,
- Stress-FCS,
- K-convergence diagnostics,
- and DB residual profiles for flow-head methods.

Only after the sim6 gate should a decision be made about DS-scale DB/WDB/Look-Ahead experiments.

---

## 12. Reproduction

### Environment

```bash
conda activate phylogfn
cd ~/Phylogfn-network-paper-methods
```

### Generate configurations

```bash
bash run_paper_methods.sh cfgs
```

### Run tests

```bash
bash run_paper_methods.sh test
```

### Regression check against the TB baseline

```bash
bash run_paper_methods.sh regress
```

### Run the canonical N=4 gate

```bash
bash run_paper_methods.sh n4
```

A CUDA-adapted runner can be used when the evaluation tools and training commands are configured with:

```text
--device=cuda
```

### Expected output files per run

Each run directory contains the training artifacts and evaluation outputs, including:

```text
config.yaml
checkpoints/
history.pkl
final_mlls.p
endpoints.csv
reference_tv.txt
fcs.txt
```

The aggregate comparison is written to:

```text
runs/n4_comparison.txt
```

---

## 13. Completed Run Directories

The completed N=4 experiment contains 20 runs:

```text
runs/20261107_125047_n4_tb_s1
runs/20261107_152947_n4_tb_s2
runs/20261107_162537_n4_tb_s3
runs/20261107_172215_n4_tb_s4
runs/20261107_181824_n4_tb_s5

runs/20261107_191512_n4_db_s1
runs/20261107_200153_n4_db_s2
runs/20261107_204912_n4_db_s3
runs/20261107_213835_n4_db_s4
runs/20261107_222724_n4_db_s5

runs/20261107_231626_n4_wdb_logcount_s1
runs/20261108_001020_n4_wdb_logcount_s2
runs/20261108_010005_n4_wdb_logcount_s3
runs/20261108_015419_n4_wdb_logcount_s4
runs/20261108_024512_n4_wdb_logcount_s5

runs/20261108_033410_n4_wdb_count_s1
runs/20261108_042142_n4_wdb_count_s2
runs/20261108_050517_n4_wdb_count_s3
runs/20261108_055142_n4_wdb_count_s4
runs/20261108_064304_n4_wdb_count_s5
```

All 20 runs produced `fcs.txt`.

---

## 14. Evaluation Implementation Note

During the first CUDA diagnostic run, two CPU/GPU conversion issues in `tools/fcs_eval.py` were exposed:

1. CUDA tensors were passed directly to NumPy.
2. CPU and CUDA tensors were combined before device normalization.

The diagnostic code was corrected so tensors are moved to CPU before NumPy conversion and before mixed-device arithmetic.

These were **post-training diagnostic compatibility bugs**. They did not invalidate the completed training checkpoints. After the correction, the FCS diagnostics were rerun successfully for the affected runs, and all 20 final runs have completed diagnostic outputs.

---

## 15. Current Status

```text
N=4 reference gate:
  COMPLETE

Runs:
  20 / 20 complete

TB:
  5 / 5

DB:
  5 / 5

WDB-logcount:
  5 / 5

WDB-count:
  5 / 5

FCS diagnostics:
  20 / 20 complete

Main result:
  TB > DB ≈ WDB-logcount > WDB-count
  for distributional correctness on this N=4 gate

Next gate:
  sim6
```
