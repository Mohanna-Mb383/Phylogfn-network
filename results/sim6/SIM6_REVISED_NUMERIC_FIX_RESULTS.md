# Revised sim6 Results After Gaussian Scale Stabilization

## Purpose

The original sim6 gate experiment exhibited numerical failures in several WDB runs. Diagnostic analysis traced the failures to unbounded exponential Gaussian scale parameterizations in the continuous merge and reticulation heads.

The revised experiment reran all six methods and all three seeds after applying the minimal numerical stabilization: cap the raw Gaussian log-scale at 20 before exponentiation while retaining the standard PyTorch Normal distribution.

## Experiment Completion

All 18 revised runs completed successfully.

| Method | Seeds completed |
|---|---:|
| TB | 3/3 |
| DB | 3/3 |
| WDB-logcount | 3/3 |
| WDB-count | 3/3 |
| WDB-depth | 3/3 |
| WDB-logcount + Look-Ahead | 3/3 |

No training, AUC, FCS, or residual-profile failures were recorded.

## FCS Results

The common comparison horizon was H = 897 seconds.

| Method | Time-averaged FCS mean | SD | Final FCS mean | SD |
|---|---:|---:|---:|---:|
| DB | 0.8467 | 0.0143 | 0.9211 | 0.0480 |
| WDB-count | 0.8321 | 0.0184 | 0.7502 | 0.0812 |
| WDB-logcount | 0.8107 | 0.0194 | 0.7751 | 0.1449 |
| WDB-depth | 0.8013 | 0.0179 | 0.7733 | 0.1624 |
| WDB-logcount + Look-Ahead | 0.7749 | 0.0552 | 0.7620 | 0.1060 |
| TB | 0.5403 | 0.0252 | 0.4520 | 0.1028 |

## Paired Comparisons Against DB

- TB - DB: mean -0.3064; negative in 3/3 seeds; Wilcoxon p = 0.250.
- WDB-count - DB: mean -0.0146; negative in 2/3 seeds; Wilcoxon p = 0.750.
- WDB-depth - DB: mean -0.0455; negative in 3/3 seeds; Wilcoxon p = 0.250.
- WDB-logcount - DB: mean -0.0361; negative in 3/3 seeds; Wilcoxon p = 0.250.
- WDB-logcount + Look-Ahead - DB: mean -0.0719; negative in 3/3 seeds; Wilcoxon p = 0.250.

With only three paired seeds, the Wilcoxon test has very limited resolution and these p-values should not be interpreted as evidence of equivalence.

## Interpretation

The numerical stabilization successfully removed the NaN/Inf failure mode observed in the original sim6 experiment: all 18 revised runs completed.

However, stabilization does not reverse the scientific conclusion of the gate experiment.

DB achieved the highest mean time-averaged FCS and the highest mean final FCS.

WDB-count was the closest WDB variant to DB in time-averaged FCS, but it did not improve over DB on average.

WDB-logcount and WDB-depth also remained below DB.

Adding Look-Ahead to WDB-logcount did not improve the result and produced the lowest mean time-averaged FCS among the WDB variants.

Therefore, the revised sim6 experiment supports the conclusion that the earlier WDB failures were numerical rather than intrinsic training crashes, but it does not provide evidence that WDB or WDB + Look-Ahead improves distributional correctness over DB.

## Gate Decision

The numerical-stability gate is passed.

The methodological performance gate for WDB improvement over DB is not passed.

The original sim6 results are retained separately to document the numerical failure mode observed before stabilization.
