# DS4 — PhyloGFN-Net Network Baseline

This directory contains the completed **DS4 network baseline** experiment.

The experiment uses the same PhyloGFN-Net baseline setup used for the
previous DS1–DS3 network experiments. KNN candidate filtering is disabled.

## Dataset

- Dataset: **DS4**
- Number of taxa: **41**
- Sequence length: **1137 sites**
- Initial possible lineage pairs: **820**
- Approximate merge-only path pair count: **11480**
- Derived Transformer input size: **9100**

The Transformer input size is derived from the sequence length:

`2 × sequence_length × 4 + 4`

The factor 2 represents the two partial-likelihood feature versions used
for network reticulation, 4 represents the DNA states A/C/G/T, and the
final 4 values describe lineage/reticulation state information.

## Model configuration

- Model: **PhyloGFN-Net**
- Environment: `ONE_STEP_LEVEL1_NETWORK`
- Network class: rooted level-1 phylogenetic networks
- Branch lengths: **continuous**
- Evolution model: **JC69**
- Likelihood: Felsenstein-based network likelihood
- KNN candidate filtering: **OFF**
- Maximum reticulations (`R_MAX`): **2**
- Reticulation prior: **EXP_R**
- `LAMBDA_R`: **1.0**
- Training loss: **Trajectory Balance (TB)**
- Backward policy: **uniform**
- Seed: **1**

## Training budget

- Epochs: **300**
- Steps per epoch: **200**
- GFN batch size: **64**
- Best-state / replay batch size: **64**
- Total items per step: **128**
- Training items per epoch: **25,600**
- Total training items: **7,680,000**

This corresponds to the **24% PhyloGFN training budget** used in the
benchmark comparison.

## Final marginal log-likelihood

**-13267.793 ± 0.183**

The uncertainty is the **sample standard deviation across the three saved
final MLL estimates**.

- Estimate 1: `-13267.818359`
- Estimate 2: `-13267.598633`
- Estimate 3: `-13267.961914`

## Final epoch

- Epoch: **299**
- MLL: **-13267.846**
- Pearson r: **0.961**
- Mean R: **2.00**
- R histogram: **[0, 0, 1024]**
- TB loss: **7.833**
- Learned log Z: **-13122.359**

The R histogram is ordered as:

`[R=0, R=1, R=2]`

and is computed from 1024 sampled networks during evaluation.

## Late-training behaviour

For epochs 200–299:

- Mean MLL: **-13267.858**
- MLL sample SD: **0.491**
- Mean Pearson r: **0.916**
- Pearson sample SD: **0.060**

These values help describe stability near the end of training rather than
using only one final epoch.

## Important interpretation note

A concentration of samples at `R=2` must **not** be interpreted directly as
proof that the biological data contain exactly two reticulation events.

`R_MAX=2` is an imposed upper limit. In addition, the `EXP_R` prior and the
different numbers of available network topologies for different values of R
can influence the sampled R distribution.

Also, absolute MLL values should not be compared directly between different
datasets because the alignments have different taxa and sequence lengths.

The meaningful model comparison is performed within the same dataset and
under explicitly matched experimental settings.

## Metrics

- `metrics/DS4_network_300ep_results.txt`  
  Final results, dataset information, training setup, and interpretation notes.

- `metrics/DS4_network_300ep_epochs.txt`  
  Complete epoch-by-epoch training and evaluation history.

## Figures

- `figures/DS4_training_diagnostics_300ep.png`  
  MLL, Pearson r, and mean R across all 300 epochs.

- `figures/DS4_R_distribution_300ep.png`  
  Final distribution of R=0, R=1, and R=2.

- `figures/DS4_R_over_epochs_300ep.png`  
  Evolution of the reticulation-count distribution during training.

- `figures/DS4_MLL_last100_epochs.png`  
  MLL behaviour during epochs 200–299.

- `figures/DS4_Pearson_last100_epochs.png`  
  Pearson r behaviour during epochs 200–299.

- `figures/DS4_final_MLL_estimates.png`  
  The three saved final MLL estimates and their mean.

- `figures/DS4_logZ_curve_300ep.png`  
  Learned log partition function during training.

- `figures/DS4_loss_curve_300ep.png`  
  Trajectory Balance training loss.

- `figures/DS4_sampled_log_score_300ep.png`  
  Mean sampled network log score during training.

## Status

**Completed: 300 / 300 epochs**

This folder contains the DS4 **network baseline without KNN**.
