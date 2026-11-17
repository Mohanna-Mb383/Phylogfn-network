# DS1 Detailed Balance (DB) Experiment

## Overview

This document records the complete seed-1 Detailed Balance experiment on DS1, including the original run, numerical failure, diagnosis, stabilization, resumed training, final result, and reproducibility caveat.

The objective was to compare Detailed Balance (DB) against the completed DS1 Trajectory Balance (TB) baseline using the same DS1 dataset and network architecture.

## Configuration

- Dataset: `dataset/benchmark_datasets/DS1.pickle`
- DB configuration: `src/configs/network_cfgs/paper/DS1_db.yaml`
- Seed: `1`
- Device: `cuda`
- Original repository HEAD: `59df38d`

Relevant continuous merge-edge configuration:

```text
EXPR_INDEPENDENT: true
NB_COMPONENTS: 5
USE_SOFT_PLUS: false
```

## TB reference

The completed DS1 TB baseline produced:

```text
Epoch 299 MLL:      -7043.032
Final MLL:          -7043.143 ± 0.104
Epoch 299 PearsonR:  0.975
Epoch 299 mean R:    2.00
R histogram:         [0, 0, 1024]
```

## Original DB run

The original DB run completed through epoch 244.

```text
Epoch 235: MLL -9632.947  PearsonR 0.638  mean R 0.14  R histogram [889,127,8]
Epoch 244: MLL -9439.896  PearsonR 0.365  mean R 0.15  R histogram [883,130,11]
```

The run failed during epoch 245 because the policy logits became NaN. This was not an out-of-memory, SSH, or tmux failure.

## Checkpoint health

The last complete checkpoint was `checkpoint_000244.pt`.

Direct inspection showed:

```text
MODEL_TENSORS = 129
NONFINITE_MODEL_TENSORS = 0
NONFINITE_OPTIMIZER_TENSORS = 0
```

Therefore the saved model and optimizer state at epoch 244 were finite.

## Gradient diagnosis

A diagnostic replay from checkpoint 244 detected the first non-finite values during backward propagation before the optimizer step.

```text
=== FIRST NONFINITE GRADIENT ===
NONFINITE_GRADIENT name=net_model.summary_token nan=128 inf=0
NONFINITE_GRADIENT name=net_model.encoder.blocks.0.norm1.weight nan=128 inf=0
...
RuntimeError: Non-finite gradient detected before optimizer step
```

The NaNs observed in Transformer gradients were propagated downstream values and did not identify the original numerical source.

## Autograd anomaly localization

Full PyTorch autograd anomaly detection localized the failing computation to:

```text
training_data_loader
 -> rollout_worker_net.py
 -> generator(input_dict)
 -> src/gfn/tb_gfn_net.py
 -> edges_model(...)
 -> src/model/edges_model/continuous/network_heads.py
 -> dist.log_prob(log_action)
 -> MixtureSameFamily.log_prob
 -> Independent.log_prob
 -> Normal.log_prob
 -> DivBackward0
```

The final error was:

```text
RuntimeError: Function DivBackward0 returned nan values in its 1th output.
```

The numerical origin was therefore the continuous Gaussian density used by `MergeEdgeModel`.

## Merge-edge Gaussian scale collapse

For the active DS1 DB configuration, the independent merge-edge distribution computed its Gaussian scale as:

```python
var_ = logits[:, 3 * self.nb_components:]
var = var_.clamp(max=20.0).exp().reshape(-1, self.nb_components, 2)
dist = MixtureSameFamily(
    Categorical(logits=mixture_logits),
    Independent(Normal(mean, var), 1),
)
```

Despite the variable name `var`, this tensor is passed as the second argument to `torch.distributions.Normal`, so it represents Gaussian standard deviation / scale rather than variance.

The implementation imposed an upper bound but no lower bound on the scale.

Direct instrumentation during replay of epoch 245 showed progressive scale collapse:

```text
1.987679070e-06
1.144870907e-06
8.046113180e-07
2.964793850e-07
9.939437717e-08
1.020855489e-08
3.174622787e-09
1.480642720e-09
6.829268817e-10
5.074461962e-10
```

At the final recorded point, the minimum squared scale was approximately `2.575016349e-19`. Immediately afterward `DivBackward0` produced NaN gradients.

Complete underflow of the scale to zero was not required for the failure.

## Numerical stabilization

The minimal fix was a lower Gaussian scale floor of `1e-3`.

One-dimensional merge distribution:

```python
var = var_.clamp(max=20.0).exp().clamp_min(1e-3)
```

Independent two-edge distribution:

```python
var = var_.clamp(max=20.0).exp().clamp_min(1e-3).reshape(-1, self.nb_components, 2)
```

This leaves scales above `1e-3` unchanged and only removes the pathological small-scale regime.

## Replay from checkpoint 244

The healthy epoch-244 checkpoint was replayed with the lower Gaussian scale floor enabled.

The previous epoch-245 failure disappeared and training completed through epoch 299 without another non-finite-gradient failure.

Selected late-epoch results:

```text
Epoch 283: MLL -9324.550  PearsonR 0.216  mean R 0.01  R histogram [1015,9,0]
Epoch 290: MLL -9399.008  PearsonR -0.265 mean R 0.08  R histogram [950,71,3]
Epoch 296: MLL -8562.212  PearsonR 0.312  mean R 0.01  R histogram [1009,15,0]
Epoch 297: MLL -8800.840  PearsonR 0.216  mean R 0.00  R histogram [1020,4,0]
Epoch 298: MLL -8685.067  PearsonR -0.174 mean R 0.00  R histogram [1022,2,0]
Epoch 299: MLL -8808.919  PearsonR -0.101 mean R 0.00  R histogram [1019,5,0]

Final MLL: mean -8738.483 std 110.383
training time 36734 s
total time 39998 s
```

The timing above covers only resumed epochs 245-299.

## Final TB vs DB comparison

| Metric | DS1 TB | DS1 DB |
|---|---:|---:|
| Final MLL estimate | `-7043.143 ± 0.104` | `-8738.483 ± 110.383` |
| Epoch-299 MLL | `-7043.032` | `-8808.919` |
| Epoch-299 PearsonR | `0.975` | `-0.101` |
| Epoch-299 mean R | `2.00` | `0.00` |
| Epoch-299 R histogram | `[0,0,1024]` | `[1019,5,0]` |

The DB final MLL was approximately `1695.340` log-likelihood units below the TB estimate.

DB therefore did not improve the TB baseline in this DS1 experiment.

## Interpretation

**Numerical conclusion:** the original epoch-245 failure was caused by collapse of the MergeEdgeModel Gaussian scale to extremely small values. The `1e-3` lower scale floor removed the failure.

**Performance conclusion:** after numerical stabilization, DB still performed substantially worse than TB in final MLL and Pearson correlation.

## Reproducibility caveat

This result is a stabilized checkpoint continuation, not a bit-for-bit uninterrupted 300-epoch DB run.

```text
epochs 0-244: original DB implementation
epochs 245-299: resumed from checkpoint_000244.pt with the Gaussian scale floor active
```

The model and optimizer states were restored from the checkpoint, but the process RNG stream restarted during resume.

A fresh epoch-0 DB run with the stabilization enabled would be required for a strictly homogeneous full-training replication.
