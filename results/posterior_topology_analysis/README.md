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
| DS4 | Completed (audit warning) | Pending | Pending |
| DS5 | Completed (audit warning) | Pending | Pending |
| DS6 | Completed (audit warning) | Pending | Pending |
| DS7 | Provisional (Class C unverified) | Pending | Pending |
| DS8 | Provisional (Class C unverified) | Pending | Pending |

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


## DS4 — Trajectory Balance

- Training: 300 epochs
- Final checkpoint: epoch 299
- Policy samples: 1,000
- Comparison subset: 100
- Rooted topologies: 735
- Root-equivalence classes: 152
- Reticulation signatures: 85
- Displayed-tree sets: 152
- Major trees: 116
- Pairwise comparisons: 4,950

Checkpoint audit: 12 PASS, 1 WARN, 0 FAIL.
Configuration compatibility warning retained.
External Newick parser validation not performed.

[DS4 Results](DS4/TB/)

## DS5 — Trajectory Balance

- Training: 300 epochs
- Final checkpoint: epoch 299
- Policy samples: 1,000
- Comparison subset: 100
- Rooted topologies: 1000
- Root-equivalence classes: 956
- Reticulation signatures: 649
- Displayed-tree sets: 956
- Major trees: 944
- Pairwise comparisons: 4,950

The compatibility audit retains a configuration warning.
Independent external Newick parser verification is pending.

[DS5 Results](DS5/TB/)


## DS6 — Trajectory Balance

- Training: 300 epochs
- Final checkpoint: epoch 299
- Policy samples: 1,000
- Validated networks: 1,000
- Comparison subset: 100
- Rooted topologies: 1000
- Root-equivalence classes: 982
- Reticulation signatures: 351
- Displayed-tree sets: 982
- Major trees: 981
- Pairwise comparisons: 4,950
- Importance-sampling ESS: 55.1 / 1,000
- Checkpoint audit: 12 PASS, 1 WARN, 0 FAIL

The compatibility warning concerns additional configuration
options in the original training code.
External Newick parser verification was not performed.

These results characterize the GFlowNet sampling distribution,
not an exact independently verified Bayesian posterior.

[DS6 Results](DS6/TB/)

## DS7 — Trajectory Balance

- Final checkpoint: epoch 299
- Samples: 1,000
- Comparison subset: 100
- Rooted topologies: 675
- Root-equivalence classes: 100
- Reticulation signatures: 6
- Displayed-tree sets: 100
- Major trees: 233
- Pairwise comparisons: 4,950

Status: Provisional.

The low-memory analysis required compatibility patches.
The Class C enumeration check remains unresolved.
External Extended Newick parser verification is pending.
These results describe the sampled GFlowNet distribution,
not an independently verified exact Bayesian posterior.

[DS7 Results](DS7/TB/)


## DS8 — Trajectory Balance

- Training: 300 epochs
- Final checkpoint: epoch 299
- Policy samples: 1,000
- Comparison subset: 100
- Rooted topologies: 922
- Root-equivalence classes: 319
- Reticulation signatures: 126
- Displayed-tree sets: 319
- Major trees: 369
- Pairwise comparisons: 4,950

Status: Provisional.

The low-memory analysis required documented compatibility
patches. Class C enumeration remains unverified.
Independent external Newick parser verification is pending.
The results characterize the sampled GFlowNet distribution,
not an independently verified exact Bayesian posterior.

[DS8 Results](DS8/TB/)
