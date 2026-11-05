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

The selected maximum-log-score network is available in `figures/DS8_network_radial_v3_annotated.png`, with numerical edge parameters in `metrics/DS8_network_radial_v3_edges.txt`. The figure uses a NetworkX layout, not a true radial layout.

## Provenance

Original server run:

`runs/20261030_225455_ds8_network_no_knn_300ep_lowmem/`


## Additional Diagnostic Figures

- `DS8_MLL_last100_epochs.png`
- `DS8_Pearson_last100_epochs.png`
- `DS8_R_distribution_300ep.png` — aggregate R histogram over 300 epochs
- `DS8_R_over_epochs_300ep.png`
- `DS8_final_MLL_estimates.png`
- `DS8_logZ_curve_300ep.png`
- `DS8_loss_curve_300ep.png`
- `DS8_sampled_log_score_300ep.png`

All figures use the original DS8 training history or final evaluation estimates.
