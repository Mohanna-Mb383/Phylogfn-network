# DS1 — Network WDB-logcount Experiment

## Experiment

- Dataset: DS1
- Taxa: 27
- Alignment sites: 1949
- Training seed: 1
- Training epochs: 300
- Steps per epoch: 200
- NET_LOSS: WDB
- WDB_GAMMA: logcount
- LOSS_TYPE: TB
- Reticulation prior: EXP_R
- LAMBDA_R: 1.0
- R_MAX: 2
- KNN: disabled

## Final results

| Metric | Value |
|---|---:|
| Final MLL mean | -7604.791 |
| Final MLL reported std | 20.744 |
| Epoch 299 MLL | -7684.714 |
| Epoch 299 Pearson R | 0.198 |
| Epoch 299 R=0 count | 1021 |
| Epoch 299 R=1 count | 1 |
| Epoch 299 R=2 count | 2 |
| Training time (seconds) | 205114 |
| Total time (seconds) | 222411 |

## Interpretation

The final WDB-logcount sampler overwhelmingly selects
networks with R=0 (1021 out of 1024 samples).

The final MLL estimate exhibits substantial variability.
The reported standard deviation describes the final
evaluation, not variability across independent training seeds.

Compared with the DS1 TB baseline, WDB-logcount shows
lower reported MLL and much weaker reward correlation.

The comparison is descriptive until the reward,
target prior, clipping behavior, and MLL estimator
conventions have been checked for equivalence.

## Files

- `config.yaml`: recorded training configuration
- `training_log.txt`: original training console output
- `training_history.csv`: 300-epoch extracted metrics
- `source_sha256.txt`: original source-file checksums

The model checkpoint and serialized run state are retained
on the training server and are not included in GitHub.
