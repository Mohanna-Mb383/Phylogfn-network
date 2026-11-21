# Structural diversity of samples from the trained PhyloGFN-Net policy: DS6, checkpoint_000299.pt

Scope. This report describes n = 1000 independent draws from the learned distribution q_theta of one checkpoint. q_theta is trained to approximate the posterior p(x | Y); how closely it does so is a separate question, addressed in section 8 with the evidence that exists and the evidence that does not. Diversity statements below are statements about q_theta.

## 1. Questions
(a) Does the trained policy generate a diverse distribution over network topologies, and how much of that diversity corresponds to different displayed trees rather than to alternative reticulation representations, root positions or continuous parameters? (b) What evidence supports that the learned distribution approximates the intended posterior?

## 2. Checkpoint, data, provenance
Run directory `/srv/scratch1/mmousabeigi/phylogfn_proposal_20260929/phylogfn/runs/20261019_185203_ds6_network_no_knn_300ep`; checkpoint `/srv/scratch1/mmousabeigi/phylogfn_proposal_20260929/phylogfn/runs/20261019_185203_ds6_network_no_knn_300ep/checkpoints/checkpoint_000299.pt`, epoch 299 (IS the last configured epoch 299); SHA-256 `54ac394f6319a781ff92f1a56290c5d804fa762b11fd2fe5c6c6eddce0ad6f09`. Objective TB, training seed 1. Dataset `dataset/benchmark_datasets/DS6.pickle` (SHA-256 `cbd1fcc618fcdc1d4d526c8e79bd13b4d8ad41432444b341301c034e243f055c`): 50 taxa, 1133 sites. Prior: p(G) proportional to exp(-1 R) over rooted level-1 networks with R <= 2 (EXP_R), branch lengths Exp(10); reward constant C = 0; slot-label correction False. Code: commit None, None modified tracked files, uncommitted-diff SHA-256 None. Versions: python 3.10.21, torch 2.5.1+cu118, numpy 2.2.6, scipy 1.15.3, networkx 3.4.2, ete3 3.1.3, yacs unknown, matplotlib 3.10.9.
Source files executed by the sampler: 48, digest `4e90fed65fef4fd341321b0a84eb9e9a4aa0a8909bbc2d30290f098539de17ec` (SHA-256 of the list of their SHA-256; every file is listed in sampling_metadata.json, key code.source_files_sha256). The working tree was not a git repository: the digest is the only identification of the code.
Source files executed by this analysis: 46, digest `c676677bc0914c070ed13b2f699c7d03f2be7b610896a12ea3ef9cdbeac2ab2b` (analysis_provenance.json).
Training log: final MLL -6715.862 +- 0.199 (3 estimates).
ENV.NETWORK options of the run: {"EXPLICIT_STOP": false, "KNN": "DNABERT2_MODEL: zhihan1996/DNABERT-2-117M\nEMBEDDING: KMER\nEMBEDDING_PATH: \nENABLED: False\nEPS_EVAL: 0.001\nEPS_MIN: 0.001\nK: 5\nKMER_K: 4\nMETRIC: COSINE\nMODE: EXACT\nPROPAGATION: DIRICHLET", "LAMBDA_CONDITIONING": false, "LAMBDA_EVAL": -1.0, "LAMBDA_R": 1.0, "LAMBDA_RANGE": "[0.0, 4.0]", "PRIOR_EXPONENT_W": 1.0, "PRIOR_TYPE": "EXP_R", "R_MAX": 2}. The root-equivalence statistics below (root positions, class size, root uniformity) assume that the sampler generates every rooted binary level-1 network without parallel edges; the checkpoint audit verifies this for 4 taxa (line "class C" of checkpoint_audit.txt).
Options of the working tree that did not exist in the repository revision this pipeline was written against (reported, not interpreted; values of this run): ENV.NETWORK.EXPLICIT_STOP = False; ENV.NETWORK.KNN.DNABERT2_MODEL = 'zhihan1996/DNABERT-2-117M'; ENV.NETWORK.KNN.EMBEDDING = 'KMER'; ENV.NETWORK.KNN.EMBEDDING_PATH = ''; ENV.NETWORK.KNN.ENABLED = False; ENV.NETWORK.KNN.EPS_EVAL = 0.001; ENV.NETWORK.KNN.EPS_MIN = 0.001; ENV.NETWORK.KNN.K = 5; ENV.NETWORK.KNN.KMER_K = 4; ENV.NETWORK.KNN.METRIC = 'COSINE'; ENV.NETWORK.KNN.MODE = 'EXACT'; ENV.NETWORK.KNN.PROPAGATION = 'DIRICHLET'; ENV.NETWORK.LAMBDA_CONDITIONING = False; ENV.NETWORK.LAMBDA_EVAL = -1.0; ENV.NETWORK.LAMBDA_RANGE = '[0.0, 4.0]'; ENV.NETWORK.PRIOR_EXPONENT_W = 1.0; ENV.REWARD.QUARTET_Q = 100; ENV.REWARD.QUARTET_SEED = 0; ENV.REWARD.SCHEDULE = '[]'; ENV.REWARD.TYPE = 'LIKELIHOOD'; GFN.MODEL.BACKWARD_HEAD.ACT_FN = 'RELU'; GFN.MODEL.BACKWARD_HEAD.DROPOUT = 0.0; GFN.MODEL.BACKWARD_HEAD.HIDDEN_SIZE = 128; GFN.MODEL.BACKWARD_HEAD.INPUT_SIZE = 258; GFN.MODEL.BACKWARD_HEAD.LAYERS = 2; GFN.MODEL.BACKWARD_HEAD.OUTPUT_SIZE = 1; GFN.MODEL.EVALUATION_FINAL_MLL_REPEATS = 3; GFN.MODEL.GRAPH_ENCODER.ENABLED = False; GFN.MODEL.GRAPH_ENCODER.HIDDEN_SIZE = 64; GFN.MODEL.GRAPH_ENCODER.LAYERS = 2; GFN.MODEL.LAMBDA_EMB.ACT_FN = 'RELU'; GFN.MODEL.LAMBDA_EMB.DROPOUT = 0.0; GFN.MODEL.LAMBDA_EMB.HIDDEN_SIZE = 64; GFN.MODEL.LAMBDA_EMB.INPUT_SIZE = 1; GFN.MODEL.LAMBDA_EMB.LAYERS = 2; GFN.MODEL.LAMBDA_EMB.OUTPUT_SIZE = 128; GFN.MODEL.STATE_FLOW_HEAD.ACT_FN = 'RELU'; GFN.MODEL.STATE_FLOW_HEAD.DROPOUT = 0.0; GFN.MODEL.STATE_FLOW_HEAD.HIDDEN_SIZE = 128; GFN.MODEL.STATE_FLOW_HEAD.INPUT_SIZE = 128; GFN.MODEL.STATE_FLOW_HEAD.LAYERS = 2; GFN.MODEL.STATE_FLOW_HEAD.OUTPUT_SIZE = 1. An option among them that changes the generated class of networks, the bounds of the continuous parameters, the reward or the backward policy changes what the samples and the diagnostics below mean.
The run configuration holds every key the sampling code defines: no default was filled in.

## 3. Sampling
RolloutWorkerNet.rollout(generator, k, generate_full_trajectories=True), random_spec=None, generator.eval(), torch.no_grad(), reward scale 1. 1000 draws in rollouts of 64 trajectories on cpu (4 threads), seeds: python/numpy/torch = 42. Every draw was kept. The replay buffer of best networks was not read. Subset: 100 draws, `numpy.random.default_rng(subset_seed).choice(n, size=subset, replace=False), sorted` with seed 42; the pairwise analysis uses this subset, all other statistics use the 1000 draws.

## 4. Format and validation
`posterior_samples_1000.newick`: one Extended Newick string per line; a reticulation is written `(subtree)#Hk:length::gamma` at its first parent and `#Hk:length::gamma` at its second; `gamma` is the inheritance probability of that hybrid edge; plain Newick when R = 0. The file names keep the word "posterior" requested for the analysis; the content is policy samples.
Independent validation by this script of all 1000 + 100 exported samples: every string parsed and structurally validated (acyclic, degrees, level-1, tree-child, parameter domains): 1000; string and edge record are the same rooted network edge by edge (branch lengths exactly, inheritance probabilities within 4.5e-16): 1000; every CSV column recomputed from the string: 1000; scores finite and consistent: 1000; subset rows identical to the corresponding full rows and subset indices reproduced from the seed: True; log score recomputed from every exported string with the explicit displayed-tree likelihood: max |difference| 1.00e-11. Failures: 0 (any failure stops the script). External parsers: this file was NOT checked by an external parser (tools/check_enewick_interop.py was not run on it). The format was verified with PhyloZoo and phylox on other files as documented in docs/posterior_topology/ENEWICK_INTEROPERABILITY.md; not tested with PhyloNetworks.jl, Dendroscope or IcyTree.

## 5. Levels of structure (definitions)
| level | object | what it ignores |
|---|---|---|
| L1 | rooted network topology: the object the policy generates (canonical string, injective on the supported class) | branch lengths, inheritance probabilities, slot labels, child order |
| L2 | root-equivalence class: L1 up to moving the root to any position that keeps every reticulation entered from above and gives no parallel edges ("semi-directed topology") | + root position |
| L3 | set of reticulation signatures: for every reticulation the hybrid descendant block and the cyclic sequence of pendant taxon blocks around its cycle | + structure inside the blocks |
| L4 | set of unrooted topologies of the displayed trees | + reticulations that leave every displayed tree unchanged |
| L5 | displayed-tree topology of largest summed inheritance weight (major tree) | depends on the inheritance probabilities |

The likelihood and the prior density take the same value on all members of an L2 class and re-rooting is a measure-preserving bijection of the parameters (tools/verify_root_equivalence.py), so under the target every member of a class has the same mass and the class has c(S) times that mass, c(S) = number of its rooted members. The model's prior is uniform over rooted networks given R, not over classes. L1 is the object of every distributional statement; L2 removes the root position, which the data cannot inform. Mean c(S) over the draws: 97.0.

## 6. Results: how many distinct structures (n = 1000 draws)
| level | distinct | distinct / n | most frequent: count, frequency [Wilson 95%] | singletons | entropy (nats) +- jackknife SE | perplexity | unseen mass (Good-Turing) +- SE | Chao1 lower bound +- SE |
|---|---|---|---|---|---|---|---|---|
| L1 rooted | 1000 | 1.000 | 1, 0.001 [0.000, 0.006] | 1000 | 6.908 +- 0.000 | 1000.0 | 1.000 +- 0.000 | 500500 +- 31607 |
| L2 class | 982 | 0.982 | 3, 0.003 [0.001, 0.009] | 965 | 6.882 +- 0.008 | 974.8 | 0.965 +- 0.008 | 30083 +- 7515 |
| L3 signatures | 351 | 0.351 | 30, 0.030 [0.021, 0.043] | 196 | 5.351 +- 0.043 | 210.8 | 0.196 +- 0.016 | 694 +- 70 |
| L4 displayed set | 982 | 0.982 | 3, 0.003 [0.001, 0.009] | 965 | 6.882 +- 0.008 | 974.8 | 0.965 +- 0.008 | 30083 +- 7515 |
| L5 major tree | 981 | 0.981 | 2, 0.002 [0.001, 0.007] | 962 | 6.881 +- 0.008 | 974.0 | 0.962 +- 0.009 | 25335 +- 5806 |

Reading the table. "distinct" and the entropy depend on n (accumulation_curves.csv and figure): with 1000 of 1000 L1 values seen once, the L1 count is limited by the sample size. The Wilson interval of the most frequent value is conditional on which value is most frequent (selected after seeing the data, so slightly optimistic). Unseen mass = estimated probability that the next draw from q_theta shows a value not yet seen; it concerns q_theta, not the support of the posterior. Chao1 is a lower bound for the number of values with non-negligible mass under q_theta, valid for independent draws; it is not an estimate of the size of the posterior support. No bootstrap interval is given for counts and entropies (the bootstrap is inconsistent for them).

Entropy decomposition of the L1 distribution (plug-in, nats; exact chain rule for nested partitions): between displayed-tree sets 6.882 (99.6%), between classes that display the same set of trees 0.000 (0.0%), root position within a class 0.025 (0.4%); total 6.908. Through reticulation signatures instead: between signature sets 5.351, between classes with the same signatures 1.532, root position 0.025.
Displayed trees weighted by inheritance: averaging the displayed-tree distributions of the 1000 draws gives 3828 distinct unrooted tree topologies; the heaviest carries 0.001 of the weight; perplexity 3210.89 (displayed_tree_distribution.csv). This is the distribution of the tree followed by a site under q_theta.

Most frequent root-equivalence classes (topology_frequencies.csv has all of them with the full canonical strings):

| rank | class | count | frequency [Wilson 95%] | R | rooted forms seen / possible | major tree (mode) | reweighted mass +- SE |
|---|---|---|---|---|---|---|---|
| 1 | a810ef770829 | 3 | 0.003 [0.001, 0.009] | 2 | 3 / 97 | 82aa20a053cb (tie) | not reported (ESS) |
| 2 | aaec7316ad7b | 2 | 0.002 [0.001, 0.007] | 2 | 2 / 97 | 0b23c0b7c500 | not reported (ESS) |
| 3 | 3636ce2b569e | 2 | 0.002 [0.001, 0.007] | 2 | 2 / 97 | da84496bf90c | not reported (ESS) |
| 4 | d3253bf6c622 | 2 | 0.002 [0.001, 0.007] | 2 | 2 / 97 | ac682f75d2d4 | not reported (ESS) |
| 5 | f6ac2761833a | 2 | 0.002 [0.001, 0.007] | 2 | 2 / 97 | 9afb475dc4e6 | not reported (ESS) |
| 6 | f23747ac36c2 | 2 | 0.002 [0.001, 0.007] | 2 | 2 / 97 | 6cfe1efa245e | not reported (ESS) |
| 7 | 1d5386141a98 | 2 | 0.002 [0.001, 0.007] | 2 | 2 / 97 | a0da1e3cc5fc | not reported (ESS) |
| 8 | 8d4735e0a2ae | 2 | 0.002 [0.001, 0.007] | 2 | 2 / 97 | d887e6314100 | not reported (ESS) |
| 9 | 7e4d8e585764 | 2 | 0.002 [0.001, 0.007] | 2 | 2 / 97 | 50b0856537bd (tie) | not reported (ESS) |
| 10 | 05a952f53d58 | 2 | 0.002 [0.001, 0.007] | 2 | 2 / 97 | ed29ee9d0bac | not reported (ESS) |
| 11 | b779fa222e91 | 2 | 0.002 [0.001, 0.007] | 2 | 2 / 97 | a890c381691d | not reported (ESS) |
| 12 | ee276e1d9b6a | 2 | 0.002 [0.001, 0.007] | 2 | 2 / 97 | 3338a38fd697 | not reported (ESS) |

## 7. Results: reticulations
| R | count | frequency [Wilson 95%] | prior of the target of this run (2^R slot labellings per rooted graph) | prior on rooted graphs |
|---|---|---|---|---|
| 0 | 0 | 0.0000 [0.000, 0.004] | 0.0000 | 0.0000 |
| 1 | 0 | 0.0000 [0.000, 0.004] | 0.0021 | 0.0042 |
| 2 | 1000 | 1.0000 [0.996, 1.000] | 0.9979 | 0.9958 |

The two prior columns are exact (counts of level-1 networks, src/utils/level1_counts.py). The prior mass on R = R_MAX = 2 is 0.9979 on slot-labelled objects and 0.9958 on rooted graphs: a sample from the prior alone shows these frequencies, so the frequency of R = R_MAX in the sample is read against them, not against 0. R_MAX is a configuration bound; nothing here is evidence for a number of reticulation events in the history of the taxa.
Odds of R = 2 against R < 2. Sample (ends of the Wilson interval): 260 to unbounded. Prior: 474 (slot-labelled objects), 237 (rooted graphs). Sample odds divided by prior odds: 0.549 to unbounded (slot-labelled objects), 1.1 to unbounded (rooted graphs). This factor is the contribution of the data only if q_theta equals the posterior; a factor near 1 means the sample says about R what the prior already says.
Cycle lengths (semi-directed view): {"5": 12, "6": 81, "7": 462, "8": 791, "9": 264, "10": 82, "11": 49, "12": 137, "13": 101, "14": 19, "15": 2}. 0 of 2000 reticulations (0.0%) have cycle length <= 3: switching such a reticulation does not change the unrooted topology of the displayed tree (it changes branch lengths only). 0 of 1000 draws (0.000 [0.000, 0.004]) contain at least one such reticulation; 0 draws (0.000) have only such reticulations. This is a structural fact about the sampled networks; by itself it says nothing about identifiability or about hybridisation.

Most frequent reticulation signatures (hybrid block | cyclic sequence of the other blocks; a block with more than three taxa is abbreviated to its first two names and the number of the others; the full lists are in reticulation_signature_frequencies.csv):

| hybrid block | cycle blocks | cycle length | draws containing it | fraction of draws [Wilson 95%] | share of all reticulations |
|---|---|---|---|---|---|
| Gnomonia_padicola | Gnomonia_setacea / Discula_campestris, Discula_fraxinae, Plagiostoma_euphorbiae / Ambrosiozyma_platypodis_L36984, Amphiporthe_castanea + 39 more / Discula_destructiva_MD254 / Apioplagiostoma_aceriferum / Pleuroceras_pleurostylum / Linospora_capreae | 8 | 200 | 0.200 [0.176, 0.226] | 0.100 |
| Gnomonia_padicola | Gnomonia_setacea / Discula_campestris, Discula_fraxinae, Plagiostoma_euphorbiae / Discula_destructiva_MD254 / Ambrosiozyma_platypodis_L36984, Amphiporthe_castanea + 39 more / Apioplagiostoma_aceriferum / Pleuroceras_pleurostylum / Linospora_capreae | 8 | 157 | 0.157 [0.136, 0.181] | 0.079 |
| Ambrosiozyma_platypodis_L36984, Candida_albicans_E15168, Saccharomyces_cerevisiae | Claviceps_purpurea_U44040 / Hypocrea_pallida_U32408, Hypomyces_polyporinus_U32410 / Nectria_cinnabarina_U32412 / Ceratocystis_fimbriata_U32418, Ceratocystis_virescens_U32419 + 4 more / Amphiporthe_castanea, Apiognomonia_supraseptata + 31 more / Xylaria_curta_U32417, Xylaria_hypoxylon_U20378 / Daldinia_concentrica_U32402, Hypoxylon_atroroseum_U32411 | 8 | 149 | 0.149 [0.128, 0.172] | 0.074 |
| Gnomonia_padicola | Gnomonia_setacea / Discula_campestris, Discula_fraxinae, Plagiostoma_euphorbiae / Ambrosiozyma_platypodis_L36984, Amphiporthe_castanea + 39 more / Discula_destructiva_MD254 / Pleuroceras_pleurostylum / Apioplagiostoma_aceriferum / Linospora_capreae | 8 | 120 | 0.120 [0.101, 0.142] | 0.060 |
| Gnomonia_padicola | Gnomonia_setacea / Discula_campestris, Discula_fraxinae, Plagiostoma_euphorbiae / Ambrosiozyma_platypodis_L36984, Amphiporthe_castanea + 40 more / Apioplagiostoma_aceriferum / Pleuroceras_pleurostylum / Linospora_capreae | 7 | 110 | 0.110 [0.092, 0.131] | 0.055 |
| Gnomonia_padicola | Gnomonia_setacea / Discula_campestris, Discula_fraxinae, Plagiostoma_euphorbiae / Ambrosiozyma_platypodis_L36984, Amphiporthe_castanea + 39 more / Discula_destructiva_MD254 / Apioplagiostoma_aceriferum, Pleuroceras_pleurostylum / Linospora_capreae | 7 | 99 | 0.099 [0.082, 0.119] | 0.050 |
| Gnomonia_padicola | Gnomonia_setacea / Discula_campestris, Discula_fraxinae, Plagiostoma_euphorbiae / Discula_destructiva_MD254 / Ambrosiozyma_platypodis_L36984, Amphiporthe_castanea + 39 more / Pleuroceras_pleurostylum / Apioplagiostoma_aceriferum / Linospora_capreae | 8 | 98 | 0.098 [0.081, 0.118] | 0.049 |
| Ambrosiozyma_platypodis_L36984, Candida_albicans_E15168, Saccharomyces_cerevisiae | Claviceps_purpurea_U44040 / Hypocrea_pallida_U32408, Hypomyces_polyporinus_U32410 / Nectria_cinnabarina_U32412 / Ceratocystis_fimbriata_U32418, Ceratocystis_virescens_U32419 + 4 more / Amphiporthe_castanea, Apiognomonia_supraseptata + 31 more / Daldinia_concentrica_U32402, Hypoxylon_atroroseum_U32411 + 2 more | 7 | 90 | 0.090 [0.074, 0.109] | 0.045 |
| Gnomonia_padicola | Gnomonia_setacea / Discula_campestris, Discula_fraxinae, Plagiostoma_euphorbiae / Discula_destructiva_MD254 / Ambrosiozyma_platypodis_L36984, Amphiporthe_castanea + 39 more / Apioplagiostoma_aceriferum, Pleuroceras_pleurostylum / Linospora_capreae | 7 | 73 | 0.073 [0.058, 0.091] | 0.036 |
| Ambrosiozyma_platypodis_L36984, Candida_albicans_E15168, Saccharomyces_cerevisiae | Claviceps_purpurea_U44040 / Hypocrea_pallida_U32408, Hypomyces_polyporinus_U32410 / Nectria_cinnabarina_U32412 / Ceratocystis_fimbriata_U32418, Ceratocystis_virescens_U32419 + 4 more / Amphiporthe_castanea, Apiognomonia_supraseptata + 31 more / Daldinia_concentrica_U32402, Hypoxylon_atroroseum_U32411 / Xylaria_curta_U32417 / Xylaria_hypoxylon_U20378 | 9 | 71 | 0.071 [0.057, 0.089] | 0.035 |

A hybrid descendant clade alone does not fix the placement: 2 distinct hybrid clades correspond to 154 distinct signatures in this sample (hybrid_clade_frequencies.csv counts draws and occurrences separately).

## 8. Does the learned distribution approximate the posterior? Evidence available in this output
When q_theta = p. If the trajectory-balance residual is zero on every complete trajectory that has positive probability under P_F or P_B (exact balance with full support), with the uniform backward policy over the parents of each state (the parent count is checked against the backward sampler in tests/test_network_env.py), then the terminal density of the policy equals R(x) / Z. Here R(x) = likelihood x branch-length prior x exp(-LAMBDA_R R) and each rooted graph is reached as 2^R slot-labelled terminal objects, so the graph-level target is proportional to 2^R R(x) unless the slot-label correction is on (it is OFF for this run). Training gives approximate balance on the trajectories it visited. None of this is guaranteed to hold at convergence of a finite run.
Clipped parameters. The sampler clips log branch lengths to [-10, 0] and logit(theta) to +-8 and evaluates log P_F with the unclipped density at the clipped value. A parameter on a bound is therefore an atom of the sampling distribution for which log P_F is not a probability. Draws with at least one parameter on a bound: 303 of 1000 (30.30%). Independent of the configured bounds: among the 103000 sampled lengths the smallest (4.540e-05) occurs 357 time(s) and the largest (0.1739) occurs 1 time(s); a continuous distribution gives each once. The target has a density, so it gives these draws no mass: the Radon-Nikodym derivative of the target with respect to the sampling distribution is 0 on an atom. Their importance weight is therefore 0 in (i). The value the training code forms for such a draw (score divided by the unclipped density at the bound) is not a weight with respect to any measure; it is shown in (i) only for comparison with the training log. The fraction of draws on a bound, 0.3030 [0.275, 0.332], estimates the mass that q_theta puts where the target has none: the total variation between q_theta and any distribution with a density is at least this mass. log p_T of these draws is not a density either: they are left out of the regression in (iii). A draw is counted as on a bound when its log length is within 1e-6 of the lower or 1e-9 of the upper bound, or its logit(theta) within 1e-3 of a bound (single-precision resolution of the stored value).

(i) Importance weights. w_i = R(x_i) P_B(tau_i | x_i) / P_F(tau_i) for the 1000 sampled trajectories. Exact balance makes w constant. Kish effective sample size 55.1 of 1000 (5.5%); standard deviation of log w 2.641 nats; largest normalised weight 0.067. The implied MLL estimate is -6716.735 (delta-method SE 0.131, unreliable when the ESS is small), against -6715.862 +- 0.199 in the training log (the training log forms a weight for every draw, also for a draw on a clipping bound; without such draws the two estimators are the same, and the comparison is a functional check that checkpoint, dataset and code agree). 303 of the 1000 draws are on a clipping bound and have weight 0 here. With the weight the training code forms for them: ESS 18.6, MLL estimate -6715.957 (delta-method SE 0.230); that estimator is biased upwards by the mass of the atoms times the ratio of the score to the unclipped density at the bound. The estimate with weight 0 targets the mass of the posterior inside the clipping box; the posterior mass outside the box cannot be reached by the sampler and is not estimated.
The ESS is a trajectory-level quantity. The weight of a terminal object is the conditional mean of the trajectory weights that end in it, so the trajectory-level ESS cannot exceed the ESS at the level of terminal objects; it is small whenever a few trajectories dominate. With ESS / n = 0.055 < 0.1 the reweighted masses below are not interpretable and are not used.

| quantity | frequency in the sample | importance-reweighted mass +- SE |
|---|---|---|
| R = 2 | 1.0000 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| root_position_type = pendant | 0.4320 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| root_position_type = internal | 0.4300 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| root_position_type = cycle | 0.1380 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = b762133e9465 | 0.0030 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = c4862b0f0988 | 0.0020 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = 363ea62d2a28 | 0.0020 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = a8e9790c004b | 0.0020 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = eb6d0a166b91 | 0.0020 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = a6b027b3275d | 0.0020 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = 080896131d37 | 0.0020 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = 3338a38fd697 | 0.0020 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = ac682f75d2d4 | 0.0020 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = da84496bf90c | 0.0020 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |

(ii) Root-position uniformity. Under the target, given the semi-directed network and its parameters, the root is uniformly distributed over the in-class root positions (equal density and unit Jacobian under re-rooting; tools/verify_root_equivalence.py). Observed versus expected number of draws by root-position type: pendant edge 432 vs 474.2 (sd 15.8, z = -2.67); internal edge 430 vs 349.5 (sd 15.1, z = 5.34); cycle edge 138 vs 176.2 (sd 12.0, z = -3.18). A necessary condition for q_theta = p that needs no reference; it tests one coordinate only. Here max |z| = 5.3: q_theta is measurably different from the target in where it puts the root.

(iii) Terminal density: the optional stage tools/posterior_density_diagnostics.py was not run for this output.

(iv) Independent reference posterior: NONE for this dataset. No exact or independently sampled posterior over level-1 networks exists for it, so the accuracy of q_theta at the level of topology masses is NOT established by this output. Diagnostics (i)-(iii) are necessary conditions computed from the model's own quantities; they can reveal a mismatch and cannot prove a match. On small simulated problems the comparison with a numerical reference is available (--reference).

## 9. Results: pairwise comparison of the 100-draw subset (4950 unordered pairs)
The pairs share draws: the uncertainty of a mean over pairs is the jackknife standard error over the 100 draws, not sd / sqrt(4950).

| quantity | what it is | mean over pairs +- jackknife SE | median | min | max | pairs at 0 | mean self value |
|---|---|---|---|---|---|---|---|
| mu_distance_norm | metric | 0.3355 +- 0.0059 | 0.3396 | 0.0377 | 0.5660 | 0.0% | 0.0000 |
| dt_rf_hausdorff_norm | pseudo-metric | 0.2304 +- 0.0040 | 0.2340 | 0.0213 | 0.4043 | 0.0% | 0.0000 |
| dt_jaccard_distance | pseudo-metric | 1.0000 +- 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% | 0.0000 |
| dt_tv | pseudo-metric | 1.0000 +- 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% | 0.0000 |
| dt_js_distance | pseudo-metric | 1.0000 +- 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% | 0.0000 |
| split_l1_norm | pseudo-metric | 0.1906 +- 0.0038 | 0.1906 | 0.0432 | 0.3462 | 0.0% | 0.0000 |
| major_rf_norm | pseudo-metric | 0.1724 +- 0.0056 | 0.1702 | 0.0213 | 0.4043 | 0.0% | 0.0000 |
| dt_rf_min_norm | dissimilarity | 0.1410 +- 0.0037 | 0.1489 | 0.0213 | 0.2979 | 0.0% | 0.0000 |
| sig_jaccard_distance | pseudo-metric | 0.9436 +- 0.0046 | 1.0000 | 0.0000 | 1.0000 | 0.5% | 0.0000 |
| hybrid_clade_jaccard_distance | pseudo-metric | 0.0000 +- 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0% | 0.0000 |
| dt_rf_mean_norm | descriptive statistic | 0.2636 +- 0.0038 | 0.2660 | 0.1170 | 0.3883 | 0.0% | 0.1211 |
| dt_rf_expected_norm | descriptive statistic | 0.2365 +- 0.0038 | 0.2362 | 0.1049 | 0.3630 | 0.0% | 0.1005 |

Normalised quantities are divided by 2(N - 3) = 94 (unrooted RF maximum) or, for the mu-distance, by the number of internal nodes of the two networks. "mean self value" is the quantity of a network compared with itself, averaged over the subset: it is 0 for every metric, pseudo-metric and dissimilarity; for the two descriptive statistics it is positive as soon as a network displays more than one tree topology, which is why they are not distances. A pseudo-metric is 0 for different networks that share the derived object named in the definitions.

Pair classes (mutually exclusive, decided by the definitions; fractions +- jackknife SE over draws):

| class | pairs | fraction +- SE | of which same major tree | same R | same signature set |
|---|---|---|---|---|---|
| P1 same rooted topology | 0 | 0.0000 +- 0.0000 | 0 | 0 | 0 |
| P2 same class different rooting | 0 | 0.0000 +- 0.0000 | 0 | 0 | 0 |
| P3 different class same displayed set | 0 | 0.0000 +- 0.0000 | 0 | 0 | 0 |
| P4 displayed sets differ share a topology | 0 | 0.0000 +- 0.0000 | 0 | 0 | 0 |
| P5 no displayed topology in common | 4950 | 1.0000 +- 0.0000 | 0 | 4950 | 24 |

P1: only continuous parameters differ. P2: the root position differs. P3: the network topologies differ and display the same set of tree topologies. P4: the displayed sets differ and share at least one tree topology. P5: no displayed tree topology in common. The classes do not say why two networks differ beyond these definitions.

## 10. Results: continuous parameters (kept separate from topology)
Total branch length over the 1000 draws: mean 0.8391, sd 0.0415. Inheritance probability of the heavier hybrid edge over 2000 reticulations: mean 0.668, sd 0.104. 17 root-equivalence classes occur at least twice (35 draws). Within them every edge and every inheritance probability was matched across draws through root-invariant taxon-based keys (all 17 edge-key sets identical within a class; a mismatch stops the script): median over classes of the mean coefficient of variation of matched branch lengths 0.282; mean sd of matched inheritance probabilities 0.053; mean coefficient of variation of the total length 0.024 (duplicate_topology_parameter_variability.csv).

## 11. Scores
log score = log R(x) = log-likelihood + log branch-length prior + log topology prior (unnormalised), a pointwise density value of one draw with its continuous parameters. Over the 1000 draws: mean -6060.97, sd 8.58, range -6098.06 to -6034.72. A topology with a high pointwise score need not have a large posterior mass: mass is the integral of R over the continuous parameters, times the number of labellings. The frequency of a topology in the sample estimates its mass under q_theta; the reweighted mass (section 8) estimates its mass under the target when the ESS allows. score_by_topology.csv lists, for the most frequent classes, the frequency with its interval, the pointwise score summary and the reweighted mass, each in its own column. Spearman correlation between the pointwise log score and the split-frequency distance to the most frequent class in the subset: rho = -0.080 (n = 100 draws; descriptive).

## 12. Answers supported by this output
(a) Diversity of q_theta. Number of distinct values among 1000 draws: rooted topologies 1000, root-equivalence classes 982, sets of reticulation signatures 351, displayed-tree sets 982, major trees 981. The most frequent class has frequency 0.003 [0.001, 0.009]; 965 of the 982 classes were seen once and the estimated unseen mass at the class level is 0.965 +- 0.008. Of the plug-in entropy of the rooted-topology distribution (6.908 nats), 99.6% lies between displayed-tree sets, 0.0% between classes that display the same trees and 0.4% in the root position. In the 100-draw subset, 100.0% of the pairs (+- 0.0) have no displayed tree topology in common, 0.0% (+- 0.0) share one while their displayed sets differ, 0.0% (+- 0.0) are different classes displaying the same set of trees, 0.0% (+- 0.0) differ in root position only and 0.0% (+- 0.0) in continuous parameters only. The mean normalised Hausdorff RF between displayed-tree sets is 0.2304 +- 0.0040 and the mean normalised mu-distance between rooted networks is 0.3355 +- 0.0059.
These numbers describe one checkpoint and one sample of size 1000; intervals cover the Monte-Carlo variability of the draws only (not the variability between training runs).
(b) Accuracy of q_theta. Importance-weight ESS 55.1 of 1000 (5.5%): the trajectory weights are far from constant. Root-position uniformity z-scores pendant -2.67, internal 5.34, cycle -3.18: a measured mismatch between q_theta and the target in the root position. No independent reference posterior exists for this dataset. The diagnostics above are necessary conditions computed from the model itself: they can show that q_theta differs from the target and cannot show that it matches it. Whether q_theta approximates the posterior at the level of topology masses cannot be answered with the data in this output.

## 13. Limitations
- The draws come from the learned distribution q_theta of one checkpoint; agreement with the posterior is not established here beyond the diagnostics of section 8.
- The target itself is the model's posterior: displayed-tree likelihood with one tree per site, Jukes-Cantor, exponential branch-length prior, exp(-LAMBDA_R R) over rooted level-1 networks with R <= R_MAX, 2^R slot labellings per graph unless corrected. It is not evidence about the true history of the taxa.
- Counts of distinct structures, entropies and perplexities depend on the sample size; see the accumulation curves. Chao1 is a lower bound under q_theta.
- Wilson intervals assume independent draws (true for separate rollouts) and, for the most frequent value, ignore that it was selected after seeing the data.
- Jackknife standard errors of pairwise means treat the 100 subset draws as the independent units; the 4950 pairs are not independent.
- L2 classes are defined by re-rooting inside the supported class; the prior is uniform over rooted networks, so a class has c(S) times the mass of one of its members.
- The mu-distance is a metric on rooted networks and is sensitive to the root position. The displayed-tree quantities are pseudo-metrics or dissimilarities on networks: two different networks can be at 0. No metric on root-equivalence classes other than the identity indicator is reported.
- L5 (major tree), dt_tv, split_l1 and the expected RF depend on the sampled inheritance probabilities.
- Reticulations with cycle length <= 3 change no displayed-tree topology; whether they are supported by the data is not addressed.
- Extended Newick interoperability was verified with PhyloZoo and phylox (docs/posterior_topology/ENEWICK_INTEROPERABILITY.md); PhyloNetworks.jl, Dendroscope and IcyTree were not tested.
- Sampled parameters are clipped (log branch length, logit of the inheritance probability); a draw on a bound is an atom of q_theta whose reported log P_F is a density value (section 8 counts them).
- CPU sampling is reproducible for the same seed, chunk size, thread count, machine and library versions (byte-identical files in the test suite); CUDA sampling is not bit-reproducible.

## 14. Reproducibility
```
python tools/sample_posterior_newick.py /srv/scratch1/mmousabeigi/phylogfn_proposal_20260929/phylogfn/runs/20261019_185203_ds6_network_no_knn_300ep dataset/benchmark_datasets/DS6.pickle --epoch 299 --expect-objective TB --n 1000 --seed 42 --subset 100 --subset-seed 42 --chunk 64 --device cpu --threads 4 --out-dir /srv/scratch1/mmousabeigi/posterior_v2_ds6_tb_1000 --dataset-name DS6
python tools/analyze_posterior_topology.py --samples-dir /srv/scratch1/mmousabeigi/posterior_v2_ds6_tb_1000 --sequences dataset/benchmark_datasets/DS6.pickle --seed 42
```
Tables: topology_summary.csv, topology_frequencies.csv, topology_frequencies_by_level.csv, displayed_tree_distribution.csv, entropy_decomposition.csv, accumulation_curves.csv, reticulation_count_vs_prior.csv, reticulation_signature_frequencies.csv, hybrid_clade_frequencies.csv, importance_reweighting.csv, root_position_uniformity.csv, pairwise_topology_distances.csv (4950 rows), pairwise_summary.csv, pair_classes.csv, pairwise_matrices.npz, duplicate_topology_parameter_variability.csv, score_by_topology.csv, validation_report.json. Figures in figures/.
