# KNN implementation and network training method

## Problem and motivation

At a partial network-construction state with `l` open lineages, there are `l(l−1)/2` distinct unordered merge pairs before legality checks. The historical DS1 initial state has 27 lineages and 351 possible pairs. The KNN proposal reduces the *preferred* merge candidate set by using embedding-space proximity.

## KNN algorithm in the repaired/vectorized run

1. Represent observed sequences using **4-mer** features (`EMBEDDING=KMER`, `KMER_K=4`). The configured but unused DNABERT-2 model is **not** the active embedding option.
2. L2-normalize embeddings and calculate cosine distance `d(i,j)=1−e_iᵀe_j`.
3. For each open lineage `i`, select the `min(5,l−1)` lowest-distance *other* lineages. Form the symmetric union of all `(i,j)` selections; an edge selected in either direction is included.
4. Restrict the primary merge-candidate policy using this mask, while retaining legality and the epsilon-uniform legal-action exploration term. This does **not** mean all non-neighbour legal pairs have zero sampling probability.
5. After merging `u,v` with branch lengths `b_u,b_v`, propagate the ancestor embedding using the normalized weighted mean

   `w_u=1/(b_u+0.001)`, `w_v=1/(b_v+0.001)`, `e_parent=normalize((w_u e_u+w_v e_v)/(w_u+w_v))`.

6. `MODE=EXACT` recomputes all distances among current lineages at each step. This is distinct from the approximate `INCREMENTAL` mode, which was **not** used.

The model also selects continuous branch lengths and network reticulation parameters; KNN does not remove these computations. KNN is a *proposal/action-selection change* and not itself evidence of a faster end-to-end system.

## Optimization of rollout execution

The repaired/vectorized run uses `PHYLOGFN_FAST_FEATURES=on`, with a batch feature module installed by the environment hook. The code groups merge and reticulation feature calculations rather than processing them separately for each trajectory. This targets the overhead of many small tensor operations and host/device synchronizations.

An on-machine CPU fingerprint comparison of repaired and vectorized versions on three seeded batches (48 trajectories) reported identical discrete/continuous actions, masks, probabilities, reward, loss, 3,861,571 gradient entries and the optimizer update parameter deltas. Coverage included R=1 and R=2, all four merge cases and replay. It did *not* cover STOP, empty-candidate states, or all possible CUDA behaviors.

## Config recorded in previous server checks

| Setting | Value |
|---|---|
| Epochs | 300 |
| Steps per epoch | 200 |
| Seed | 1 |
| Device | CUDA, NVIDIA L40S |
| Objective | TB (previous assertion on smoke configuration; confirm against exported YAML) |
| KNN | enabled; k=5; exact; cosine |
| Embedding | 4-mer |
| Propagation | Dirichlet |
| Exploration | epsilon annealing: 0.5 to 0.001, T=150 epochs |
| Reward temperature | 16.0 to 1.0, T=200 epochs |
| Max reticulations | 2 (historical DS1 configuration; confirm against YAML) |

The exported `config.yaml` from `vm03` is the authoritative configuration once added.
