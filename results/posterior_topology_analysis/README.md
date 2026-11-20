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
| DS2 | Completed (compatibility warning) | Pending | Pending |
| DS3 | Completed (1 audit warning) | Pending | Pending |
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


## DS2 — Trajectory Balance

- Training: 300 epochs
- Final checkpoint: epoch 299
- Samples: 1,000
- Rooted topologies: 76
- Root-equivalence classes: 2
- Reticulation signatures: 2
- Displayed-tree sets: 2
- Major trees: 8

The audit retains a configuration-compatibility warning
because the original training code contains additional options.
Results describe the sampled GFlowNet distribution, not an
independently verified exact Bayesian posterior.

Results: [DS2/TB](DS2/TB/)


## DS3 — Trajectory Balance

- Training: 300 epochs
- Final checkpoint: epoch 299
- Policy samples: 1,000
- Comparison subset: 100
- Rooted topologies: 293
- Root-equivalence classes: 17
- Reticulation signatures: 13
- Displayed-tree sets: 17
- Major trees: 36
- Pairwise comparisons: 4,950
- Importance-sampling ESS: 12.3 / 1,000
- Checkpoint audit: 12 PASS, 1 WARN, 0 FAIL

The remaining warning concerns configuration compatibility.
External Newick parser verification was not performed.
These results characterize the GFlowNet sampling distribution,
not an independently validated exact Bayesian posterior.

[DS3 Results](DS3/TB/)
