# DS3 — PhyloGFN-Net baseline

This folder contains the 300-epoch **DS3 network baseline** experiment.

## Experimental setup

- Dataset: DS3
- Taxa: 36
- Sequence length: 1812 sites
- Model: PhyloGFN-Net
- Network class: rooted level-1 networks
- KNN: disabled
- Branch lengths: continuous
- R_MAX: 2
- Reticulation prior: EXP_R
- LAMBDA_R: 1.0
- Loss: Trajectory Balance
- Backward policy: uniform
- Epochs: 300
- Steps per epoch: 200
- GFN batch size: 64
- Best-state batch size: 64
- Seed: 1

The total training budget is 7.68 million training items, corresponding
to the 24% PhyloGFN training budget used in the benchmark comparison.

## Final marginal likelihood

**-33545.219 ± 0.498**

The uncertainty above is the sample standard deviation across the three
saved final MLL estimates.

Final estimates:

- -33544.730469
- -33545.726562
- -33545.199219

## Last epoch

- Epoch: 299
- MLL: -33545.9453
- Pearson r: 0.9216
- Mean R: 2.00
- R histogram: [0, 0, 1024]

## Interpretation

The final R histogram should not be interpreted as proof that DS3 contains
exactly two biological reticulations. The experiment imposes R_MAX=2, and
the EXP_R prior together with the number of available network topologies can
affect the distribution over R.

## Files

- `metrics/DS3_network_300ep_results.txt` — final summary and configuration
- `metrics/DS3_network_300ep_epochs.txt` — complete epoch-by-epoch metrics
- `figures/DS3_training_diagnostics_300ep.png` — MLL, Pearson r, and mean R
- `figures/DS3_R_distribution_300ep.png` — final R distribution
