# DS7 Network Baseline

PhyloGFN level-1 network baseline on DS7 with KNN disabled.

## Configuration

- Epochs: 300
- Steps per epoch: 200
- GFN batch: 64
- Replay/best-state batch: 64
- R_MAX: 2
- LAMBDA_R: 1.0
- PRIOR_TYPE: EXP_R
- Loss: TB
- Backward policy: UNIFORM
- KNN: disabled
- Seed: 1
- LOW_MEMORY_TB: True
- MINI_BATCH_SPLITS: 1

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

## Run directory

`runs/20261026_014933_ds7_network_no_knn_300ep_lowmem`
