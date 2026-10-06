# DS2 — PhyloGFN-Net baseline

This directory contains the 300-epoch DS2 phylogenetic-network baseline run without KNN filtering.

## Experimental setup

The DS2 run uses the same baseline experimental configuration as the historical DS1 network run.

- Model: PhyloGFN-Net
- Dataset: DS2
- KNN: disabled
- R_MAX: 2
- LAMBDA_R: 1.0
- Prior: EXP_R
- Loss: Trajectory Balance (TB)
- Backward policy: uniform
- Transformer depth: 6
- GFN batch size: 64
- Best-state batch size: 64
- Epochs: 300
- Steps per epoch: 200
- LR_MODEL: 0.0001
- LR_Z: 0.01
- DS2 seed: 1

No DS2-specific training configuration was created. Dataset-dependent sequence length and the corresponding input dimension are determined from the alignment.

## Final results

Final marginal-likelihood estimates:

- -26273.58984375
- -26273.830078125
- -26273.646484375

Final MLL:

**-26273.689 ± 0.126**  
(sample SD across 3 final estimates)

Last epoch (299):

- MLL: -26273.6602
- Pearson r: 0.9588
- Mean R: 2.00
- R histogram: [0, 0, 1024]
- Temperature: 1.0

## Configuration audit against DS1

The resolved DS1 and DS2 run configurations had 182 common keys.

Only three common values differed:

1. sequence length: DS1 = 1949, DS2 = 2520
2. derived Transformer input size: DS1 = 15596, DS2 = 20164
3. output path

These are dataset/runtime differences rather than changes to the experimental hyperparameters.

The historical DS1 seed was not recorded. DS2 was launched with seed 1.

## Interpretation caution

The final R histogram must not be interpreted as biological evidence that DS2 contains exactly two reticulations. With the EXP_R prior, LAMBDA_R=1 and R_MAX=2, topology-class-size effects can strongly favor R=2.

## Files

- `metrics/DS2_network_300ep_results.txt` — final numerical summary
- `metrics/DS2_network_300ep_epochs.txt` — epoch-by-epoch training history
- `figures/DS2_training_diagnostics_300ep.png` — training diagnostics
