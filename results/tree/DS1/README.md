# DS1 — Phylogenetic Tree Baseline

This folder contains the phylogenetic-tree baseline used for comparison with the network experiments on DS1.

## Result

The PhyloGFN tree baseline for DS1 is approximately:

**MLL = -7108.4**

Our reproduced tree baseline was approximately:

**MLL = -7108.47**

These values are consistent and provide the reference point for the DS1 network experiments.

## Comparison role

The tree result is used as the baseline for:

- the standard phylogenetic-network experiment;
- the phylogenetic-network + KNN experiment.

This folder currently stores the baseline/reference result rather than a complete epoch-by-epoch tree training run.

## Files

- [Tree baseline metrics](metrics/DS1_tree_baseline.txt)
