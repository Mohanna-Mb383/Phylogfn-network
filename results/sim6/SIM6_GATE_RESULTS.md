# N=6 Gate Experiment: Distributional Correctness and Numerical Stability

## 1. Purpose

This experiment is the second distributional-correctness gate for the PhyloGFN-Net paper-methods study.

The N=4 gate showed that Trajectory Balance (TB) substantially outperformed Detailed Balance (DB) and the tested Weighted Detailed Balance (WDB) variants.

The N=6 experiment was used to test whether the N=4 result persists on a larger phylogenetic-network problem, evaluate additional WDB weighting schemes, evaluate the prospective-child Look-Ahead adaptation, and assess numerical stability before any DS1–DS8 scale-up.

## 2. Dataset

The simulated N=6 dataset contains 6 sequences, 300 sites per sequence, and 1 simulated reticulation.

```text
dataset/simulated/sim6_r1_nontrivial.pickle
```

## 3. Experimental Design

Six variants were scheduled, each with seeds 1, 2, and 3:

| Variant | Seeds |
|---|---:|
| TB | 1, 2, 3 |
| DB | 1, 2, 3 |
| WDB-logcount | 1, 2, 3 |
| WDB-count | 1, 2, 3 |
| WDB-depth | 1, 2, 3 |
| WDB-logcount + Look-Ahead | 1, 2, 3 |

Total planned runs: 18.

Training and evaluation were executed with CUDA on the NVIDIA L40S GPU.

The main distributional endpoint was FCS; lower FCS is better. The final comparison used a common training-time horizon of 1051 s.

## 4. Completion and Numerical Stability

| Variant | Completed | Failed |
|---|---:|---:|
| TB | 3/3 | 0/3 |
| DB | 3/3 | 0/3 |
| WDB-logcount | 1/3 | 2/3 |
| WDB-count | 1/3 | 2/3 |
| WDB-depth | 2/3 | 1/3 |
| WDB-logcount + Look-Ahead | 2/3 | 1/3 |

Overall, 12 of 18 planned runs completed and 6 failed during training. TB and DB completed all seeds. The WDB family completed 6 of 12 planned runs.

## 5. Training Failures

All six observed WDB-family training failures were associated with non-finite action logits (`NaN`) during categorical policy sampling.

Characteristic error:

```text
ValueError: Expected parameter logits ... to satisfy ...
but found invalid values:
tensor([[nan, nan, ...]], device='cuda:0')
```

| Variant | Seed | Last completed epoch | Failure |
|---|---:|---:|---|
| WDB-logcount | 2 | 33 | NaN action logits |
| WDB-logcount | 3 | 48 | NaN action logits |
| WDB-count | 1 | 15 | NaN action logits |
| WDB-count | 3 | 26 | NaN action logits |
| WDB-depth | 1 | 37 | NaN action logits |
| WDB-logcount + Look-Ahead | 3 | 46 | NaN action logits |

WDB-logcount seed 2 also failed again when rerun under the same configuration, showing that this particular failure was reproducible rather than a single transient interruption.

These failures were not GPU out-of-memory errors. They arose from non-finite policy logits during training. The stability conclusion is restricted to this N=6 network MDP, parametrization, optimizer/configuration, and training budget.

## 6. FCS Results

### 6.1 Trajectory Balance

| Seed | AUC | Time-averaged FCS | Final FCS |
|---:|---:|---:|---:|
| 1 | 526.44 | 0.5007 | 0.4130 |
| 2 | 576.37 | 0.5481 | 0.3745 |
| 3 | 547.27 | 0.5205 | 0.5686 |

Mean time-averaged FCS: **0.5231**  
Mean final FCS: **0.4520**

### 6.2 Detailed Balance

| Seed | AUC | Time-averaged FCS | Final FCS |
|---:|---:|---:|---:|
| 1 | 877.75 | 0.8348 | 0.9757 |
| 2 | 906.78 | 0.8624 | 0.9020 |
| 3 | 895.04 | 0.8512 | 0.8855 |

Mean time-averaged FCS: **0.8495**  
Mean final FCS: **0.9211**

### 6.3 WDB-logcount

| Seed | AUC | Time-averaged FCS | Final FCS | Status |
|---:|---:|---:|---:|---|
| 1 | 886.56 | 0.8431 | 0.7380 | Complete |
| 2 | — | — | — | NaN failure |
| 3 | — | — | — | NaN failure |

Only one seed completed, so no valid three-seed aggregate comparison is available.

### 6.4 WDB-count

| Seed | AUC | Time-averaged FCS | Final FCS | Status |
|---:|---:|---:|---:|---|
| 1 | — | — | — | NaN failure |
| 2 | 876.60 | 0.8337 | 0.6760 | Complete |
| 3 | — | — | — | NaN failure |

Only one seed completed, so no valid three-seed aggregate comparison is available.

### 6.5 WDB-depth

| Seed | AUC | Time-averaged FCS | Final FCS | Status |
|---:|---:|---:|---:|---|
| 1 | — | — | — | NaN failure |
| 2 | 816.31 | 0.7763 | 0.6119 | Complete |
| 3 | 753.86 | 0.7169 | 0.6280 | Complete |

Mean over the two successful seeds:

- time-averaged FCS ≈ **0.7466**
- final FCS ≈ **0.6200**

### 6.6 WDB-logcount + Look-Ahead

| Seed | AUC | Time-averaged FCS | Final FCS | Status |
|---:|---:|---:|---:|---|
| 1 | 838.07 | 0.7970 | 0.6869 | Complete |
| 2 | 849.43 | 0.8078 | 0.8837 | Complete |
| 3 | — | — | — | NaN failure |

Mean over the two successful seeds:

- time-averaged FCS ≈ **0.8024**
- final FCS ≈ **0.7853**

Look-Ahead improved modestly relative to DB on its two successful paired seeds, but remained substantially worse than TB.

## 7. Paired Comparisons

### TB versus DB

```text
mean difference = -0.3264
SE              = 0.0061
negative         = 3/3 seeds
Wilcoxon p       = 0.250
```

Since lower FCS is better, TB was better than DB in all three paired seeds. With only three paired observations, the exact Wilcoxon test has very limited resolution.

### WDB-count versus DB

```text
mean difference = -0.0287
paired seeds    = 1
```

No meaningful inferential conclusion should be drawn from one paired seed.

### WDB-depth versus DB

```text
mean difference = -0.1102
SE              = 0.0241
negative         = 2/2
Wilcoxon p       = 0.500
```

This suggests lower FCS than DB in the successful seeds, but the comparison is incomplete because one WDB-depth seed failed.

### WDB-logcount versus DB

```text
mean difference = +0.0084
paired seeds    = 1
```

No meaningful three-seed conclusion is possible.

### WDB-logcount + Look-Ahead versus DB

```text
mean difference = -0.0461
SE              = 0.0084
negative         = 2/2
Wilcoxon p       = 0.500
```

Look-Ahead produced a modest FCS improvement over DB on the two successful seeds, but did not approach TB performance and did not complete all three seeds.

## 8. Main Scientific Finding

The N=6 experiment reinforces the main result of the N=4 gate.

TB was numerically stable across all planned seeds and substantially better than DB in distributional correctness as measured by FCS.

Approximate mean time-averaged FCS:

```text
TB               ≈ 0.523
DB               ≈ 0.849
WDB-depth        ≈ 0.747   (2 successful seeds only)
WDB-logcount+LA  ≈ 0.802   (2 successful seeds only)
WDB-logcount      0.843   (1 successful seed only)
WDB-count         0.834   (1 successful seed only)
```

The WDB variants did not provide a stable improvement over the existing TB baseline.

The Look-Ahead adaptation showed a modest improvement relative to DB among its successful runs, but remained substantially worse than TB, completed only 2/3 seeds, and failed in seed 3 because of non-finite action logits.

## 9. Gate Decision

The confirmatory gate for direct scale-up to DS1–DS8 was not passed.

```text
N=4:
TB substantially outperformed DB and both tested WDB variants.

N=6:
TB again substantially outperformed DB.
WDB variants showed substantial numerical instability.
Look-Ahead did not recover TB-level distributional correctness.
```

Therefore DS1–DS8 should not be presented as the next confirmatory stage under the original gate logic.

Any subsequent DS1–DS8 experiments should be explicitly framed as exploratory scale-up experiments or as part of a revised methodology after addressing WDB numerical stability.

## 10. Raw Result Files

Raw aggregate comparison:

```text
results/sim6_comparison.txt
```

Recorded training failures:

```text
results/sim6_failures.txt
```

## 11. Reproducibility Note

The original `sim6` runner stopped when a WDB run produced NaN logits because the script used fail-fast behavior.

A safe resume runner was subsequently used to skip already-completed runs, preserve failed runs and logs, continue the remaining scheduled variants, and record failures rather than terminating the full experiment.

No learning-rate, WDB-weighting, or optimizer change was introduced to rescue failed seeds. This preserves comparability under the fixed experimental configuration.

## 12. Overall Status

```text
N4 gate:      COMPLETE
sim6 gate:    COMPLETE
DS scale-up:  NOT PASSED AS CONFIRMATORY GATE
```
