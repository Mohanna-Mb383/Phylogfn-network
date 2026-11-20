# Posterior Topology Analysis — DS1 to DS8

## Objective

Evaluate the structural diversity, similarity, and stability
of phylogenetic networks sampled by trained GFlowNet models.

The analysis includes:

- Sampling 1,000 phylogenetic networks per experiment.
- Exporting 1,000 Extended Newick networks.
- Selecting 100 networks for pairwise comparison.
- Measuring topology diversity and structural distances.
- Comparing rooted and root-equivalent topologies.
- Analysing reticulation signatures and displayed trees.
- Evaluating sampling and importance-weight diagnostics.

## Datasets

| Dataset | TB | DB | WDB |
|---------|----|----|-----|
| DS1 | Completed | Pending | Pending |
| DS2 | Pending | Pending | Pending |
| DS3 | Pending | Pending | Pending |
| DS4 | Pending | Pending | Pending |
| DS5 | Pending | Pending | Pending |
| DS6 | Pending | Pending | Pending |
| DS7 | Pending | Pending | Pending |
| DS8 | Pending | Pending | Pending |

## DS1 — Trajectory Balance

- Training: 300 epochs
- Final checkpoint: epoch 299
- Samples: 1,000
- Distinct rooted topologies: 537
- Root-equivalence classes: 100
- Reticulation signatures: 50
- Displayed-tree sets: 100
- Major trees: 106
- Normalized mean mu-distance: 0.2631
- Normalized Hausdorff RF distance: 0.0839
- Checkpoint audit: 16 PASS, 0 WARN, 0 FAIL

Results: [DS1/TB](DS1/TB/)

## Interpretation

These experiments characterize the topology distribution
sampled by the trained GFlowNet policy.

They do not establish exact equality with the Bayesian
target posterior. Independent posterior validation remains
a separate research task.
