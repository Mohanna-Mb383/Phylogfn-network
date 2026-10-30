# DS6 Network Baseline

PhyloGFN level-1 network baseline on DS6 with KNN disabled.

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

## Final results

Three final MLL estimates:

- -6715.6201171875
- -6716.107421875
- -6715.859375

Mean:

- **Raw final MLL: -6715.862305**
- **Sample SD: 0.243666**

Because the final samples have R histogram `[0, 0, 1024]`, all evaluation samples have R=2.

Applying the graph-level slot-label correction:

- **Corrected final MLL: -6717.248599**

## Epoch 299

- TB loss: 6.806073
- Learned logZ: -6529.500977
- Raw MLL: -6715.926758
- Corrected MLL: -6717.313052
- Pearson r: 0.971809
- Mean R: 2.0
- R histogram: [0, 0, 1024]
- Sampled log-score mean: -6061.319095

## Run directory

`runs/20261019_185203_ds6_network_no_knn_300ep`
