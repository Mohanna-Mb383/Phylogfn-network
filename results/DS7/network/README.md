# DS7 Network Baseline

PhyloGFN level-1 network baseline on DS7 with KNN disabled.

## Configuration

- Epochs: 300
- Steps per epoch: 200
- GFN batch: 64
- Replay/best-state batch: 64
- Effective batch size: 128
- R_MAX: 2
- LAMBDA_R: 1.0
- PRIOR_TYPE: EXP_R
- Loss: TB
- Backward policy: UNIFORM
- KNN: disabled
- Seed: 1
- LOW_MEMORY_TB: True
- MINI_BATCH_SPLITS: 1

## DS7 implementation note

DS7 is substantially larger than DS1-DS6, and the original TB implementation exceeded the available GPU memory when using the full batch.

For DS7, two engineering settings were changed:

- `LOW_MEMORY_TB: True`
- `MINI_BATCH_SPLITS: 1`

The low-memory implementation does not change the Trajectory Balance objective, reward, prior, or scientific model.

Instead of keeping the full autograd graph for all trajectory steps in GPU memory, it stores lightweight trajectory information and replays the same saved states and actions when computing the gradient. No new actions are sampled during this replay.

Controlled tests showed gradient agreement with the original TB implementation at approximately `1e-7`, within numerical precision.

This allowed DS7 to use the full effective batch size of 128 without splitting the batch into smaller parts.

DS1-DS6 were trained using the original implementation. The low-memory version was introduced for DS7 because of its higher GPU memory requirement.

## Final results

Three final MLL estimates:

- -37140.850554105076
- -37141.187009161695
- -37140.811919382555

Mean:

- **Raw final MLL: -37140.949828**
- **Sample SD: 0.206312**

Because the final samples have R histogram `[0, 0, 1024]`, all evaluation samples have R=2.

Applying the graph-level slot-label correction:

- **Corrected final MLL: -37142.336122**

## Epoch 299

- TB loss: 94.821702
- Learned logZ: -36914.058594
- Raw MLL: -37140.833449
- Corrected MLL: -37142.219744
- Pearson r: 0.580026
- Mean R: 2.0
- R histogram: [0, 0, 1024]
- Sampled log-score mean: -36398.234905
- Skipped updates: 0
- Peak memory: 6904.137216 MB
- Triangle fraction: 0.0
- Degenerate fraction: 0.0

## Figures

The `figures/` directory contains:

- `DS7_loss_curve_300ep.png`
- `DS7_logZ_curve_300ep.png`
- `DS7_sampled_log_score_300ep.png`
- `DS7_R_over_epochs_300ep.png`
- `DS7_R_distribution_300ep.png`
- `DS7_final_MLL_estimates.png`
- `DS7_MLL_last100_epochs.png`
- `DS7_Pearson_last100_epochs.png`
- `DS7_training_diagnostics_300ep.png`

## Metrics

Detailed epoch-level and final metrics are available in:

- `metrics/DS7_network_300ep_epochs.txt`
- `metrics/DS7_network_300ep_results.txt`

## Run directory

`runs/20261026_014933_ds7_network_no_knn_300ep_lowmem`
