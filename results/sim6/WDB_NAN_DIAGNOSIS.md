# WDB NaN Diagnosis and Numerical Stabilization

## Problem

During the original sim6 gate experiment, several WDB runs failed with non-finite policy logits.

A reproducible case was `sim6_wdb_count`, seed 1, which failed shortly after epoch 15.

## Diagnosis

Instrumentation showed that the policy-logit failure was downstream of non-finite gradients.

The first detected failure was:

NONFINITE_GRADIENT name=net_model.summary_token nan=64 inf=0

PyTorch autograd anomaly detection localized the numerical failure to the continuous Gaussian distributions.

For the reticulation Gaussian head, the learned log-scale became extremely large:

raw_max = 44.7628937
std_max = 2.75599055e+19
std_nonfinite = 0
var_nonfinite = 2
var_max = inf

The scale itself remained finite, but scale squared overflowed in float32 inside Normal.log_prob, producing PowBackward0 NaNs.

Further diagnostics showed the same unbounded exponential scale parameterization in the merge-edge Gaussian mixture.

## Diagnostic Ablations

- Original implementation: failed shortly after epoch 15.
- Stable Normal log-prob only: progressed further but failed with ExpBackward0.
- Reticulation cap only: progressed further but later failed in the merge-edge Normal distribution.
- Shared StableNormal plus log-scale caps: completed all 60 epochs.
- Final minimal all-caps-only implementation: completed all 60 epochs and preserved regression behavior.

## Final Minimal Fix

The final implementation retains the standard PyTorch Normal distribution and caps the raw Gaussian log-scale before exponentiation:

raw_log_scale.clamp(max=20.0).exp()

The guard is applied consistently to:

1. merge one-edge Gaussian mixture;
2. merge two-edge Gaussian mixture;
3. reticulation Gaussian mixture.

No replacement of Normal.log_prob is required in the final implementation.

## Validation

The final all-caps-only implementation passed:

- paper-method unit tests;
- regression test with exit status 0;
- reproducible sim6_wdb_count, seed 1, through epoch 59 without NaN/Inf failure.

The original sim6 gate results are retained separately.

A complete revised sim6 experiment must be run under the stabilized implementation before reconsidering the scale-up gate.
