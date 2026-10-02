# Experimental Results

The experimental results are organised by model type.

## Overview

| Experiment | Dataset | Main MLL result | Status |
|---|---|---:|---|
| Phylogenetic tree | DS1 | ~`-7108.47` | Baseline/reference |
| Phylogenetic network | DS1 | ~`-7043.06` | Completed |
| Network + KNN (`k=5`) | DS1 | `-6723.93 ± 5.20` | Diagnostic / provisional |

## 1. Phylogenetic Tree

The phylogenetic-tree result is used as the baseline for comparison with the network models.

[Open DS1 tree results](tree/DS1/README.md)

Main reference result:

**MLL ≈ -7108.47**

---

## 2. Phylogenetic Network

The network model extends the tree model with level-1 reticulation events, branch lengths, and inheritance probabilities.

[Open DS1 network results](network/DS1/README.md)

Main result:

**MLL ≈ -7043.06**

### Training diagnostics

![Network training diagnostics](network/DS1/figures/DS1_training_diagnostics_300ep.png)

### Example sampled network

![Network example](network/DS1/figures/network_radial_v3_annotated.png)

---

## 3. Phylogenetic Network + KNN

The KNN experiment reduces the set of merge candidates evaluated by the pair-scoring network.

[Open DS1 KNN results](network_knn/DS1/README.md)

Mean final MLL:

**-6723.93 ± 5.20**

This result is currently considered diagnostic because late-training non-finite gradients required some optimizer updates to be skipped.

### Training diagnostics

![KNN training diagnostics](network_knn/DS1/figures/DS1_KNN_training_diagnostics_300ep.png)

### Reticulation distribution

![KNN R distribution](network_knn/DS1/figures/DS1_KNN_R_distribution_300ep.png)

---

## Directory structure

```text
results/
├── tree/
│   └── DS1/
│       ├── README.md
│       └── metrics/
│
├── network/
│   └── DS1/
│       ├── README.md
│       ├── figures/
│       ├── metrics/
│       └── configs/
│
├── network_knn/
│   └── DS1/
│       ├── README.md
│       ├── figures/
│       ├── metrics/
│       └── configs/
│
└── legacy_experiments/
cat > results/tree/DS1/metrics/DS1_tree_baseline.txt <<'EOF'
DS1 PHYLOGENETIC TREE BASELINE
=============================

Dataset: DS1
Model class: phylogenetic tree

Published / benchmark PhyloGFN reference:
MLL approximately -7108.4

Locally reproduced tree baseline used for comparison:
MLL approximately -7108.47

Purpose:
This value is used as the tree baseline for comparison with the
phylogenetic-network experiments on the same DS1 dataset.

Important:
This folder currently documents the tree baseline/reference result.
A complete epoch-by-epoch tree training log is not stored in this repository.
