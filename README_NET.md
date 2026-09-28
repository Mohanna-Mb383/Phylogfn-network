# PhyloGFN-Net — level-1 phylogenetic networks with PhyloGFN

This is the official PhyloGFN repository (Zhou et al., ICLR 2024) extended so that the
GFlowNet generates **level-1 phylogenetic networks** instead of trees. The tree code path is
untouched (`ENV.ENVIRONMENT_TYPE: ONE_STEP_BINARY_TREE` still runs the original sampler).

The accompanying lecture (`lecture/main.pdf`, LaTeX sources in `lecture/`) explains every
modification and why it is correct.

## What changed (map)

| Id | File | Content |
|----|------|---------|
| M1–M7 | `src/env/phylo_network_env.py` (new) | network state, MERGE / RETICULATE actions, level-1 masks, two-version Felsenstein features (exact likelihood without 2^R displayed trees), reticulation prior, parent counting, backward sampling |
| M8 | `src/model/network_model/one_step_network_model.py`, `src/model/edges_model/continuous/network_heads.py` (new) | policy with pair + single logits; heads for branch lengths and (b_hx, theta) |
| M9 | `src/utils/level1_counts.py` (new) | exact number of level-1 networks per (N, R); topology-prior normaliser for the MLL |
| M10 | `src/gfn/tb_gfn_net.py` (new) | generator: routes the two action types, trajectory-balance loss |
| M11 | `src/gfn/rollout_worker_net.py`, `src/gfn/training_data_loader_net.py`, `src/gfn/gfn_evaluator_net.py` (new) | variable-length rollouts, replay buffer of networks, MLL / Pearson-r evaluation |
| M12 | `src/configs/defaults.py`, `src/env/__init__.py`, `src/gfn/build.py`, `src/utils/utils.py` (4 small edits) + `train_net.py`, `src/configs/network_cfgs/`, `tools/`, `tests/` | registrations, config, training script, data simulation, tests |

`git diff` against the original repository is in `phylogfn-net.patch`.

## Run

```bash
pip install torch numpy scipy tqdm dill docopt fvcore ete3 networkx
python tests/test_network_env.py                                   # verification (~3 min, CPU)
python tools/simulate_network_data.py dataset/simulated/sim6_r1.pickle --leaves=6 --sites=300 --rets=1 --seed=3
python train_net.py src/configs/network_cfgs/cfg_net_small_cpu.yaml dataset/simulated/sim6_r1.pickle runs/sim6_r1
python tools/compare_tree_mode.py dataset/simulated/sim6_r1.pickle 8 40   # R_MAX = 0 vs. original PhyloGFN
```

## Verified

* The MDP with the level-1 masks reaches exactly the class of level-1 networks: 36 (N=3), 603 (N=4, R<=2),
  11,460 (N=5, R<=2) distinct terminal networks — the published counts — and has no dead ends.
* The fast likelihood (two feature vectors per lineage, mixing at the closing merge) equals the explicit
  sum over the 2^R displayed trees to 1e-11.
* Backward sampling replays to the same network, score and parent counts (uniform P_B is a distribution over true parents).
* With `R_MAX: 0` the sampler is PhyloGFN (same evidence estimate within training noise).
