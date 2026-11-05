# DS8 — PhyloGFN-Net Training Results

## Experiment

- Dataset: DS8
- Model: PhyloGFN-Net
- Training: 300 epochs, 200 steps per epoch
- Configuration: Low-memory
- KNN: Disabled
- Maximum reticulations: R_MAX = 2

## Final Training Results

| Metric | Value |
|---|---:|
| Completed epochs | 300 |
| Final MLL mean | -8621.866 |
| Final MLL standard deviation | 0.450 |
| Last-epoch MLL | -8622.101 |
| Last-epoch Pearson correlation | -0.01049 |
| Last-epoch mean R | 2.0 |
| Last-epoch R histogram (R=0,1,2) | [0, 0, 1024] |
| Last-epoch triangle fraction | 0.0 |
| Last-epoch degenerate fraction | 0.0 |
| Skipped updates (last epoch) | 0 |

## Available Files

- `configs/cfg_net_DS8_300ep_lowmem.yaml`: experiment configuration.
- `metrics/DS8_network_300ep_epochs.csv`: metrics recorded across 300 epochs.
- `figures/DS8_training_diagnostics_300ep.png`: training diagnostic plots.

## Interpretation and Limitations

Training completed all 300 epochs. The last-epoch Pearson correlation was approximately zero; therefore, a strong correlation between the evaluated quantities is not demonstrated.

All 1,024 samples in the last-epoch reticulation histogram had R=2. This is the configured maximum reticulation count and does not establish that the underlying biological history contains exactly two reticulations.

A sampled-network visualization and its numerical edge-parameter table have not yet been validated for DS8. They are intentionally not included here.

## Provenance

Original server run:

`runs/20261030_225455_ds8_network_no_knn_300ep_lowmem/`
