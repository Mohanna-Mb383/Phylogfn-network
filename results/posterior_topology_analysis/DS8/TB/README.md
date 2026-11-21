# Structural diversity of samples from the trained PhyloGFN-Net policy: DS8, checkpoint_000299.pt

Scope. This report describes n = 1000 independent draws from the learned distribution q_theta of one checkpoint. q_theta is trained to approximate the posterior p(x | Y); how closely it does so is a separate question, addressed in section 8 with the evidence that exists and the evidence that does not. Diversity statements below are statements about q_theta.

## 1. Questions
(a) Does the trained policy generate a diverse distribution over network topologies, and how much of that diversity corresponds to different displayed trees rather than to alternative reticulation representations, root positions or continuous parameters? (b) What evidence supports that the learned distribution approximates the intended posterior?

## 2. Checkpoint, data, provenance
Run directory `/srv/scratch1/mmousabeigi/phylogfn-v3eng-bench/runs/20261030_225455_ds8_network_no_knn_300ep_lowmem`; checkpoint `/srv/scratch1/mmousabeigi/phylogfn-v3eng-bench/runs/20261030_225455_ds8_network_no_knn_300ep_lowmem/checkpoints/checkpoint_000299.pt`, epoch 299 (IS the last configured epoch 299); SHA-256 `bb6091835609022183ecbf20e5be5b7d9cfc61582a0c8ff2d215f1a637f67a7f`. Objective TB, training seed 1. Dataset `dataset/benchmark_datasets/DS8.pickle` (SHA-256 `d8cbb80d10e97f537ae8193e861426fa96d41bba49ba561c1bb818473f404ffa`): 64 taxa, 1008 sites. Prior: p(G) proportional to exp(-1 R) over rooted level-1 networks with R <= 2 (EXP_R), branch lengths Exp(10); reward constant C = 0; slot-label correction False. Code: commit None, None modified tracked files, uncommitted-diff SHA-256 None. Versions: python 3.10.21, torch 2.5.1+cu118, numpy 2.2.6, scipy 1.15.3, networkx 3.4.2, ete3 3.1.3, yacs unknown, matplotlib 3.10.9.
Source files executed by the sampler: 46, digest `5e40a9f3774d4045cf7bfd2c2ef66c695cc11d8b9495823a6eb8e91173b5383a` (SHA-256 of the list of their SHA-256; every file is listed in sampling_metadata.json, key code.source_files_sha256). The working tree was not a git repository: the digest is the only identification of the code.
Source files executed by this analysis: 44, digest `7cd50779382cd004d5442791ed3502db71e629efce74b5e0da1c1829ecb14ee2` (analysis_provenance.json).
Training log: final MLL -8621.866 +- 0.450 (3 estimates).
ENV.NETWORK options of the run: {"BLOCK_TRIANGLES": false, "CYCLE_MASK": "NONE", "LAMBDA_R": 1.0, "PRIOR_TYPE": "EXP_R", "R_MAX": 2, "SLOT_LABEL_CORRECTION": false}. The root-equivalence statistics below (root positions, class size, root uniformity) assume that the sampler generates every rooted binary level-1 network without parallel edges; the checkpoint audit verifies this for 4 taxa (line "class C" of checkpoint_audit.txt).
Options of the working tree that did not exist in the repository revision this pipeline was written against (reported, not interpreted; values of this run): ENV.NETWORK.BLOCK_TRIANGLES = False; ENV.NETWORK.CYCLE_MASK = 'NONE'; ENV.NETWORK.SLOT_LABEL_CORRECTION = False; GFN.MODEL.CONTINUOUS_BOUNDS = 'CLIP'; GFN.TRAINING_DATA_LOADER.LOW_MEMORY_TB = True. An option among them that changes the generated class of networks, the bounds of the continuous parameters, the reward or the backward policy changes what the samples and the diagnostics below mean.
The run configuration holds every key the sampling code defines: no default was filled in.

## 3. Sampling
RolloutWorkerNet.rollout(generator, k, generate_full_trajectories=True), random_spec=None, generator.eval(), torch.no_grad(), reward scale 1. 1000 draws in rollouts of 64 trajectories on cpu (4 threads), seeds: python/numpy/torch = 42. Every draw was kept. The replay buffer of best networks was not read. Subset: 100 draws, `numpy.random.default_rng(subset_seed).choice(n, size=subset, replace=False), sorted` with seed 42; the pairwise analysis uses this subset, all other statistics use the 1000 draws.

## 4. Format and validation
`posterior_samples_1000.newick`: one Extended Newick string per line; a reticulation is written `(subtree)#Hk:length::gamma` at its first parent and `#Hk:length::gamma` at its second; `gamma` is the inheritance probability of that hybrid edge; plain Newick when R = 0. The file names keep the word "posterior" requested for the analysis; the content is policy samples.
Independent validation by this script of all 1000 + 100 exported samples: every string parsed and structurally validated (acyclic, degrees, level-1, tree-child, parameter domains): 1000; string and edge record are the same rooted network edge by edge (branch lengths exactly, inheritance probabilities within 4.5e-16): 1000; every CSV column recomputed from the string: 1000; scores finite and consistent: 1000; subset rows identical to the corresponding full rows and subset indices reproduced from the seed: True; log score recomputed from every exported string with the explicit displayed-tree likelihood: max |difference| 1.18e-11. Failures: 0 (any failure stops the script). External parsers: this file was NOT checked by an external parser (tools/check_enewick_interop.py was not run on it). The format was verified with PhyloZoo and phylox on other files as documented in docs/posterior_topology/ENEWICK_INTEROPERABILITY.md; not tested with PhyloNetworks.jl, Dendroscope or IcyTree.

## 5. Levels of structure (definitions)
| level | object | what it ignores |
|---|---|---|
| L1 | rooted network topology: the object the policy generates (canonical string, injective on the supported class) | branch lengths, inheritance probabilities, slot labels, child order |
| L2 | root-equivalence class: L1 up to moving the root to any position that keeps every reticulation entered from above and gives no parallel edges ("semi-directed topology") | + root position |
| L3 | set of reticulation signatures: for every reticulation the hybrid descendant block and the cyclic sequence of pendant taxon blocks around its cycle | + structure inside the blocks |
| L4 | set of unrooted topologies of the displayed trees | + reticulations that leave every displayed tree unchanged |
| L5 | displayed-tree topology of largest summed inheritance weight (major tree) | depends on the inheritance probabilities |

The likelihood and the prior density take the same value on all members of an L2 class and re-rooting is a measure-preserving bijection of the parameters (tools/verify_root_equivalence.py), so under the target every member of a class has the same mass and the class has c(S) times that mass, c(S) = number of its rooted members. The model's prior is uniform over rooted networks given R, not over classes. L1 is the object of every distributional statement; L2 removes the root position, which the data cannot inform. Mean c(S) over the draws: 123.0.

## 6. Results: how many distinct structures (n = 1000 draws)
| level | distinct | distinct / n | most frequent: count, frequency [Wilson 95%] | singletons | entropy (nats) +- jackknife SE | perplexity | unseen mass (Good-Turing) +- SE | Chao1 lower bound +- SE |
|---|---|---|---|---|---|---|---|---|
| L1 rooted | 922 | 0.922 | 4, 0.004 [0.002, 0.010] | 858 | 6.792 +- 0.017 | 890.7 | 0.858 +- 0.015 | 8139 +- 1128 |
| L2 class | 319 | 0.319 | 114, 0.114 [0.096, 0.135] | 204 | 4.814 +- 0.060 | 123.2 | 0.204 +- 0.016 | 803 +- 103 |
| L3 signatures | 126 | 0.126 | 167, 0.167 [0.145, 0.191] | 55 | 3.697 +- 0.051 | 40.3 | 0.055 +- 0.010 | 198 +- 26 |
| L4 displayed set | 319 | 0.319 | 114, 0.114 [0.096, 0.135] | 204 | 4.814 +- 0.060 | 123.2 | 0.204 +- 0.016 | 803 +- 103 |
| L5 major tree | 369 | 0.369 | 74, 0.074 [0.059, 0.092] | 224 | 5.208 +- 0.053 | 182.7 | 0.224 +- 0.017 | 802 +- 84 |

Reading the table. "distinct" and the entropy depend on n (accumulation_curves.csv and figure): with 858 of 922 L1 values seen once, the L1 count is limited by the sample size. The Wilson interval of the most frequent value is conditional on which value is most frequent (selected after seeing the data, so slightly optimistic). Unseen mass = estimated probability that the next draw from q_theta shows a value not yet seen; it concerns q_theta, not the support of the posterior. Chao1 is a lower bound for the number of values with non-negligible mass under q_theta, valid for independent draws; it is not an estimate of the size of the posterior support. No bootstrap interval is given for counts and entropies (the bootstrap is inconsistent for them).

Entropy decomposition of the L1 distribution (plug-in, nats; exact chain rule for nested partitions): between displayed-tree sets 4.814 (70.9%), between classes that display the same set of trees 0.000 (0.0%), root position within a class 1.978 (29.1%); total 6.792. Through reticulation signatures instead: between signature sets 3.697, between classes with the same signatures 1.117, root position 1.978.
Displayed trees weighted by inheritance: averaging the displayed-tree distributions of the 1000 draws gives 906 distinct unrooted tree topologies; the heaviest carries 0.040 of the weight; perplexity 305.79 (displayed_tree_distribution.csv). This is the distribution of the tree followed by a site under q_theta.

Most frequent root-equivalence classes (topology_frequencies.csv has all of them with the full canonical strings):

| rank | class | count | frequency [Wilson 95%] | R | rooted forms seen / possible | major tree (mode) | reweighted mass +- SE |
|---|---|---|---|---|---|---|---|
| 1 | 2f49be24627c | 114 | 0.114 [0.096, 0.135] | 2 | 78 / 123 | 556dfbcf5345 | not reported (ESS) |
| 2 | 2517144a3488 | 53 | 0.053 [0.041, 0.069] | 2 | 44 / 123 | 45ea4b8844af | not reported (ESS) |
| 3 | b65c3e8f4722 | 43 | 0.043 [0.032, 0.057] | 2 | 34 / 123 | 416c6fa66d85 | not reported (ESS) |
| 4 | f5a74c55c9c4 | 35 | 0.035 [0.025, 0.048] | 2 | 31 / 123 | b21487bb0ae1 | not reported (ESS) |
| 5 | 035639614afb | 32 | 0.032 [0.023, 0.045] | 2 | 26 / 123 | 2a54824cf733 | not reported (ESS) |
| 6 | 786751b4f286 | 18 | 0.018 [0.011, 0.028] | 2 | 17 / 123 | 51471ef3e57f | not reported (ESS) |
| 7 | 5cadde0454d4 | 18 | 0.018 [0.011, 0.028] | 2 | 17 / 123 | a49e06950a58 | not reported (ESS) |
| 8 | 36057ee896e6 | 17 | 0.017 [0.011, 0.027] | 2 | 15 / 123 | bde2f41aca44 | not reported (ESS) |
| 9 | 5816f1ac4e0f | 17 | 0.017 [0.011, 0.027] | 2 | 16 / 123 | 80e58fb74db4 | not reported (ESS) |
| 10 | 8691912fa21d | 16 | 0.016 [0.010, 0.026] | 2 | 14 / 123 | 58ac70416c65 | not reported (ESS) |
| 11 | 279ee9e0b1bd | 14 | 0.014 [0.008, 0.023] | 2 | 14 / 123 | 0b539139122f | not reported (ESS) |
| 12 | a568fcd803c3 | 13 | 0.013 [0.008, 0.022] | 2 | 13 / 123 | d650bf875eb3 | not reported (ESS) |

## 7. Results: reticulations
| R | count | frequency [Wilson 95%] | prior of the target of this run (2^R slot labellings per rooted graph) | prior on rooted graphs |
|---|---|---|---|---|
| 0 | 0 | 0.0000 [0.000, 0.004] | 0.0000 | 0.0000 |
| 1 | 0 | 0.0000 [0.000, 0.004] | 0.0012 | 0.0025 |
| 2 | 1000 | 1.0000 [0.996, 1.000] | 0.9988 | 0.9975 |

The two prior columns are exact (counts of level-1 networks, src/utils/level1_counts.py). The prior mass on R = R_MAX = 2 is 0.9988 on slot-labelled objects and 0.9975 on rooted graphs: a sample from the prior alone shows these frequencies, so the frequency of R = R_MAX in the sample is read against them, not against 0. R_MAX is a configuration bound; nothing here is evidence for a number of reticulation events in the history of the taxa.
Odds of R = 2 against R < 2. Sample (ends of the Wilson interval): 260 to unbounded. Prior: 801 (slot-labelled objects), 401 (rooted graphs). Sample odds divided by prior odds: 0.325 to unbounded (slot-labelled objects), 0.65 to unbounded (rooted graphs). This factor is the contribution of the data only if q_theta equals the posterior; a factor near 1 means the sample says about R what the prior already says.
Cycle lengths (semi-directed view): {"6": 1000, "10": 1, "11": 1, "12": 14, "13": 107, "14": 369, "15": 507, "16": 1}. 0 of 2000 reticulations (0.0%) have cycle length <= 3: switching such a reticulation does not change the unrooted topology of the displayed tree (it changes branch lengths only). 0 of 1000 draws (0.000 [0.000, 0.004]) contain at least one such reticulation; 0 draws (0.000) have only such reticulations. This is a structural fact about the sampled networks; by itself it says nothing about identifiability or about hybridisation.

Most frequent reticulation signatures (hybrid block | cyclic sequence of the other blocks; a block with more than three taxa is abbreviated to its first two names and the number of the others; the full lists are in reticulation_signature_frequencies.csv):

| hybrid block | cycle blocks | cycle length | draws containing it | fraction of draws [Wilson 95%] | share of all reticulations |
|---|---|---|---|---|---|
| Hypocrea_lutea, Hypocrea_schweintzii | Hypocrea_pallida / Acremonium_alternatum, Albosynnema_elegans + 55 more / Hypomyces_aurantius, Hypomyces_australis / Sphaerostilbella_aureonitens / Sphaerostilbella_berkeleyana | 6 | 1000 | 1.000 [0.996, 1.000] | 0.500 |
| Kallichroma_glabrum, Kallichroma_tethys_1, Kallichroma_tethys_2 | Cordyceps_capitata / Cordycepioideus_bisporus / Epichloe_typhina, Neotyphodium_coenophialum / Balansia_obtecta / Hypocrea_lutea, Hypocrea_pallida + 5 more / Albosynnema_elegans, Calonectria_morganii + 14 more / Leucosphaerina_indica / Bionectria_cf._aureofulva_1, Bionectria_cf._aureofulva_2 + 20 more / Emericellopsis_terricola, Stanjemonium_fuscescens, Stanjemonium_grisellum / Acremonium_alternatum, Geosmithia_lavendula, Geosmithia_putterillii / Bulbithecium_hyalosporum / Nigrosabulum_globosum / Hapsidospora_irregularis / Mycoarachis_inversa | 15 | 167 | 0.167 [0.145, 0.191] | 0.084 |
| Kallichroma_glabrum, Kallichroma_tethys_1, Kallichroma_tethys_2 | Cordyceps_capitata / Cordycepioideus_bisporus / Epichloe_typhina, Neotyphodium_coenophialum / Balansia_obtecta / Hypocrea_lutea, Hypocrea_pallida + 5 more / Albosynnema_elegans, Calonectria_morganii + 14 more / Leucosphaerina_indica / Bionectria_cf._aureofulva_1, Bionectria_cf._aureofulva_2 + 20 more / Emericellopsis_terricola, Stanjemonium_fuscescens, Stanjemonium_grisellum / Acremonium_alternatum, Geosmithia_lavendula, Geosmithia_putterillii / Bulbithecium_hyalosporum / Hapsidospora_irregularis, Nigrosabulum_globosum / Mycoarachis_inversa | 14 | 84 | 0.084 [0.068, 0.103] | 0.042 |
| Kallichroma_glabrum, Kallichroma_tethys_1, Kallichroma_tethys_2 | Cordyceps_capitata / Cordycepioideus_bisporus / Epichloe_typhina, Neotyphodium_coenophialum / Balansia_obtecta / Hypocrea_lutea, Hypocrea_pallida + 5 more / Albosynnema_elegans, Calonectria_morganii + 14 more / Leucosphaerina_indica / Bionectria_cf._aureofulva_1, Bionectria_cf._aureofulva_2 + 20 more / Emericellopsis_terricola, Stanjemonium_fuscescens, Stanjemonium_grisellum / Acremonium_alternatum, Geosmithia_lavendula, Geosmithia_putterillii / Bulbithecium_hyalosporum / Hapsidospora_irregularis / Nigrosabulum_globosum / Mycoarachis_inversa | 15 | 68 | 0.068 [0.054, 0.085] | 0.034 |
| Kallichroma_glabrum, Kallichroma_tethys_1, Kallichroma_tethys_2 | Cordyceps_capitata / Cordycepioideus_bisporus / Epichloe_typhina, Neotyphodium_coenophialum / Balansia_obtecta / Hypocrea_lutea, Hypocrea_pallida + 5 more / Albosynnema_elegans, Calonectria_morganii + 14 more / Leucosphaerina_indica / Bionectria_cf._aureofulva_1, Bionectria_cf._aureofulva_2 + 20 more / Emericellopsis_terricola, Stanjemonium_fuscescens, Stanjemonium_grisellum / Acremonium_alternatum, Geosmithia_lavendula, Geosmithia_putterillii / Bulbithecium_hyalosporum / Mycoarachis_inversa / Nigrosabulum_globosum / Hapsidospora_irregularis | 15 | 52 | 0.052 [0.040, 0.068] | 0.026 |
| Kallichroma_glabrum, Kallichroma_tethys_1, Kallichroma_tethys_2 | Cordyceps_capitata / Cordycepioideus_bisporus / Epichloe_typhina, Neotyphodium_coenophialum / Balansia_obtecta / Hypocrea_lutea, Hypocrea_pallida + 5 more / Albosynnema_elegans, Calonectria_morganii + 14 more / Bionectria_cf._aureofulva_1, Bionectria_cf._aureofulva_2 + 20 more / Leucosphaerina_indica / Emericellopsis_terricola, Stanjemonium_fuscescens, Stanjemonium_grisellum / Acremonium_alternatum, Geosmithia_lavendula, Geosmithia_putterillii / Bulbithecium_hyalosporum / Nigrosabulum_globosum / Hapsidospora_irregularis / Mycoarachis_inversa | 15 | 46 | 0.046 [0.035, 0.061] | 0.023 |
| Kallichroma_glabrum, Kallichroma_tethys_1, Kallichroma_tethys_2 | Cordyceps_capitata / Cordycepioideus_bisporus / Epichloe_typhina, Neotyphodium_coenophialum / Balansia_obtecta / Hypocrea_lutea, Hypocrea_pallida + 5 more / Albosynnema_elegans, Calonectria_morganii + 14 more / Leucosphaerina_indica / Bionectria_cf._aureofulva_1, Bionectria_cf._aureofulva_2 + 20 more / Emericellopsis_terricola, Stanjemonium_fuscescens, Stanjemonium_grisellum / Acremonium_alternatum, Geosmithia_lavendula, Geosmithia_putterillii / Bulbithecium_hyalosporum / Hapsidospora_irregularis, Mycoarachis_inversa, Nigrosabulum_globosum | 13 | 35 | 0.035 [0.025, 0.048] | 0.018 |
| Kallichroma_glabrum, Kallichroma_tethys_1, Kallichroma_tethys_2 | Cordyceps_capitata / Cordycepioideus_bisporus / Epichloe_typhina, Neotyphodium_coenophialum / Balansia_obtecta / Hypocrea_lutea, Hypocrea_pallida + 5 more / Albosynnema_elegans, Calonectria_morganii + 14 more / Leucosphaerina_indica / Bionectria_cf._aureofulva_1, Bionectria_cf._aureofulva_2 + 20 more / Emericellopsis_terricola, Stanjemonium_fuscescens, Stanjemonium_grisellum / Acremonium_alternatum, Geosmithia_lavendula, Geosmithia_putterillii / Bulbithecium_hyalosporum / Mycoarachis_inversa / Hapsidospora_irregularis, Nigrosabulum_globosum | 14 | 28 | 0.028 [0.019, 0.040] | 0.014 |
| Kallichroma_glabrum, Kallichroma_tethys_1, Kallichroma_tethys_2 | Cordyceps_capitata / Cordycepioideus_bisporus / Epichloe_typhina, Neotyphodium_coenophialum / Balansia_obtecta / Hypocrea_lutea, Hypocrea_pallida + 5 more / Albosynnema_elegans, Calonectria_morganii + 14 more / Leucosphaerina_indica / Bionectria_cf._aureofulva_1, Bionectria_cf._aureofulva_2 + 20 more / Emericellopsis_terricola, Stanjemonium_fuscescens, Stanjemonium_grisellum / Acremonium_alternatum, Bulbithecium_hyalosporum + 2 more / Nigrosabulum_globosum / Hapsidospora_irregularis / Mycoarachis_inversa | 14 | 26 | 0.026 [0.018, 0.038] | 0.013 |
| Kallichroma_glabrum, Kallichroma_tethys_1, Kallichroma_tethys_2 | Cordyceps_capitata / Cordycepioideus_bisporus / Epichloe_typhina, Neotyphodium_coenophialum / Balansia_obtecta / Hypocrea_lutea, Hypocrea_pallida + 5 more / Albosynnema_elegans, Calonectria_morganii + 14 more / Leucosphaerina_indica / Bionectria_cf._aureofulva_1, Bionectria_cf._aureofulva_2 + 20 more / Emericellopsis_terricola, Stanjemonium_fuscescens, Stanjemonium_grisellum / Acremonium_alternatum, Geosmithia_lavendula, Geosmithia_putterillii / Bulbithecium_hyalosporum / Mycoarachis_inversa / Hapsidospora_irregularis / Nigrosabulum_globosum | 15 | 25 | 0.025 [0.017, 0.037] | 0.013 |

A hybrid descendant clade alone does not fix the placement: 2 distinct hybrid clades correspond to 127 distinct signatures in this sample (hybrid_clade_frequencies.csv counts draws and occurrences separately).

## 8. Does the learned distribution approximate the posterior? Evidence available in this output
When q_theta = p. If the trajectory-balance residual is zero on every complete trajectory that has positive probability under P_F or P_B (exact balance with full support), with the uniform backward policy over the parents of each state (the parent count is checked against the backward sampler in tests/test_network_env.py), then the terminal density of the policy equals R(x) / Z. Here R(x) = likelihood x branch-length prior x exp(-LAMBDA_R R) and each rooted graph is reached as 2^R slot-labelled terminal objects, so the graph-level target is proportional to 2^R R(x) unless the slot-label correction is on (it is OFF for this run). Training gives approximate balance on the trajectories it visited. None of this is guaranteed to hold at convergence of a finite run.
Clipped parameters. The sampler clips log branch lengths to [-10, 0] and logit(theta) to +-8 and evaluates log P_F with the unclipped density at the clipped value. A parameter on a bound is therefore an atom of the sampling distribution for which log P_F is not a probability. Draws with at least one parameter on a bound: 183 of 1000 (18.30%). Independent of the configured bounds: among the 131000 sampled lengths the smallest (4.540e-05) occurs 200 time(s) and the largest (0.1481) occurs 1 time(s); a continuous distribution gives each once. The target has a density, so it gives these draws no mass: the Radon-Nikodym derivative of the target with respect to the sampling distribution is 0 on an atom. Their importance weight is therefore 0 in (i). The value the training code forms for such a draw (score divided by the unclipped density at the bound) is not a weight with respect to any measure; it is shown in (i) only for comparison with the training log. The fraction of draws on a bound, 0.1830 [0.160, 0.208], estimates the mass that q_theta puts where the target has none: the total variation between q_theta and any distribution with a density is at least this mass. log p_T of these draws is not a density either: they are left out of the regression in (iii). A draw is counted as on a bound when its log length is within 1e-6 of the lower or 1e-9 of the upper bound, or its logit(theta) within 1e-3 of a bound (single-precision resolution of the stored value).

(i) Importance weights. w_i = R(x_i) P_B(tau_i | x_i) / P_F(tau_i) for the 1000 sampled trajectories. Exact balance makes w constant. Kish effective sample size 20.3 of 1000 (2.0%); standard deviation of log w 2.019 nats; largest normalised weight 0.199. The implied MLL estimate is -8622.244 (delta-method SE 0.220, unreliable when the ESS is small), against -8621.866 +- 0.450 in the training log (the training log forms a weight for every draw, also for a draw on a clipping bound; without such draws the two estimators are the same, and the comparison is a functional check that checkpoint, dataset and code agree). 183 of the 1000 draws are on a clipping bound and have weight 0 here. With the weight the training code forms for them: ESS 27.8, MLL estimate -8622.038 (delta-method SE 0.187); that estimator is biased upwards by the mass of the atoms times the ratio of the score to the unclipped density at the bound. The estimate with weight 0 targets the mass of the posterior inside the clipping box; the posterior mass outside the box cannot be reached by the sampler and is not estimated.
The ESS is a trajectory-level quantity. The weight of a terminal object is the conditional mean of the trajectory weights that end in it, so the trajectory-level ESS cannot exceed the ESS at the level of terminal objects; it is small whenever a few trajectories dominate. With ESS / n = 0.020 < 0.1 the reweighted masses below are not interpretable and are not used.

| quantity | frequency in the sample | importance-reweighted mass +- SE |
|---|---|---|
| R = 2 | 1.0000 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| root_position_type = pendant | 0.4450 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| root_position_type = internal | 0.4120 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| root_position_type = cycle | 0.1430 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = 33c02b93066c | 0.1140 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = 3b76b2dd073b | 0.0530 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = 1c49a28b2b97 | 0.0430 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = 34c98ec094a6 | 0.0350 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = 1e045bde9369 | 0.0320 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = 556dfbcf5345 | 0.0740 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = d9f9073f6fbb | 0.0420 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = 45ea4b8844af | 0.0300 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = 416c6fa66d85 | 0.0240 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = bc2216db2d0a | 0.0220 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |

(ii) Root-position uniformity. Under the target, given the semi-directed network and its parameters, the root is uniformly distributed over the in-class root positions (equal density and unit Jacobian under re-rooting; tools/verify_root_equivalence.py). Observed versus expected number of draws by root-position type: pendant edge 445 vs 479.7 (sd 15.8, z = -2.19); internal edge 412 vs 354.7 (sd 15.1, z = 3.78); cycle edge 143 vs 165.6 (sd 11.8, z = -1.92). A necessary condition for q_theta = p that needs no reference; it tests one coordinate only. Here max |z| = 3.8: q_theta is measurably different from the target in where it puts the root.

(iii) Terminal density: the optional stage tools/posterior_density_diagnostics.py was not run for this output.

(iv) Independent reference posterior: NONE for this dataset. No exact or independently sampled posterior over level-1 networks exists for it, so the accuracy of q_theta at the level of topology masses is NOT established by this output. Diagnostics (i)-(iii) are necessary conditions computed from the model's own quantities; they can reveal a mismatch and cannot prove a match. On small simulated problems the comparison with a numerical reference is available (--reference).

## 9. Results: pairwise comparison of the 100-draw subset (4950 unordered pairs)
The pairs share draws: the uncertainty of a mean over pairs is the jackknife standard error over the 100 draws, not sd / sqrt(4950).

| quantity | what it is | mean over pairs +- jackknife SE | median | min | max | pairs at 0 | mean self value |
|---|---|---|---|---|---|---|---|
| mu_distance_norm | metric | 0.1637 +- 0.0056 | 0.1642 | 0.0000 | 0.3582 | 0.0% | 0.0000 |
| dt_rf_hausdorff_norm | pseudo-metric | 0.0529 +- 0.0043 | 0.0492 | 0.0000 | 0.2295 | 2.7% | 0.0000 |
| dt_jaccard_distance | pseudo-metric | 0.9636 +- 0.0092 | 1.0000 | 0.0000 | 1.0000 | 2.7% | 0.0000 |
| dt_tv | pseudo-metric | 0.9693 +- 0.0077 | 1.0000 | 0.0018 | 1.0000 | 0.0% | 0.0000 |
| dt_js_distance | pseudo-metric | 0.9718 +- 0.0075 | 1.0000 | 0.0017 | 1.0000 | 0.0% | 0.0000 |
| split_l1_norm | pseudo-metric | 0.0650 +- 0.0032 | 0.0627 | 0.0004 | 0.1926 | 0.0% | 0.0000 |
| major_rf_norm | pseudo-metric | 0.0766 +- 0.0043 | 0.0656 | 0.0000 | 0.2787 | 1.4% | 0.0000 |
| dt_rf_min_norm | dissimilarity | 0.0369 +- 0.0022 | 0.0328 | 0.0000 | 0.1148 | 5.5% | 0.0000 |
| sig_jaccard_distance | pseudo-metric | 0.6314 +- 0.0068 | 0.6667 | 0.0000 | 0.6667 | 5.3% | 0.0000 |
| hybrid_clade_jaccard_distance | pseudo-metric | 0.0000 +- 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0% | 0.0000 |
| dt_rf_mean_norm | descriptive statistic | 0.1500 +- 0.0021 | 0.1475 | 0.1066 | 0.2254 | 0.0% | 0.1176 |
| dt_rf_expected_norm | descriptive statistic | 0.1436 +- 0.0025 | 0.1414 | 0.0864 | 0.2401 | 0.0% | 0.1050 |

Normalised quantities are divided by 2(N - 3) = 122 (unrooted RF maximum) or, for the mu-distance, by the number of internal nodes of the two networks. "mean self value" is the quantity of a network compared with itself, averaged over the subset: it is 0 for every metric, pseudo-metric and dissimilarity; for the two descriptive statistics it is positive as soon as a network displays more than one tree topology, which is why they are not distances. A pseudo-metric is 0 for different networks that share the derived object named in the definitions.

Pair classes (mutually exclusive, decided by the definitions; fractions +- jackknife SE over draws):

| class | pairs | fraction +- SE | of which same major tree | same R | same signature set |
|---|---|---|---|---|---|
| P1 same rooted topology | 1 | 0.0002 +- 0.0003 | 0 | 1 | 1 |
| P2 same class different rooting | 134 | 0.0271 +- 0.0083 | 65 | 134 | 134 |
| P3 different class same displayed set | 0 | 0.0000 +- 0.0000 | 0 | 0 | 0 |
| P4 displayed sets differ share a topology | 135 | 0.0273 +- 0.0072 | 2 | 135 | 0 |
| P5 no displayed topology in common | 4680 | 0.9455 +- 0.0121 | 0 | 4680 | 127 |

P1: only continuous parameters differ. P2: the root position differs. P3: the network topologies differ and display the same set of tree topologies. P4: the displayed sets differ and share at least one tree topology. P5: no displayed tree topology in common. The classes do not say why two networks differ beyond these definitions.

## 10. Results: continuous parameters (kept separate from topology)
Total branch length over the 1000 draws: mean 1.5682, sd 0.0552. Inheritance probability of the heavier hybrid edge over 2000 reticulations: mean 0.626, sd 0.074. 115 root-equivalence classes occur at least twice (796 draws). Within them every edge and every inheritance probability was matched across draws through root-invariant taxon-based keys (all 115 edge-key sets identical within a class; a mismatch stops the script): median over classes of the mean coefficient of variation of matched branch lengths 0.334; mean sd of matched inheritance probabilities 0.065; mean coefficient of variation of the total length 0.027 (duplicate_topology_parameter_variability.csv).

## 11. Scores
log score = log R(x) = log-likelihood + log branch-length prior + log topology prior (unnormalised), a pointwise density value of one draw with its continuous parameters. Over the 1000 draws: mean -7811.05, sd 9.52, range -7889.33 to -7788.27. A topology with a high pointwise score need not have a large posterior mass: mass is the integral of R over the continuous parameters, times the number of labellings. The frequency of a topology in the sample estimates its mass under q_theta; the reweighted mass (section 8) estimates its mass under the target when the ESS allows. score_by_topology.csv lists, for the most frequent classes, the frequency with its interval, the pointwise score summary and the reweighted mass, each in its own column. Spearman correlation between the pointwise log score and the split-frequency distance to the most frequent class in the subset: rho = -0.139 (n = 100 draws; descriptive).

## 12. Answers supported by this output
(a) Diversity of q_theta. Number of distinct values among 1000 draws: rooted topologies 922, root-equivalence classes 319, sets of reticulation signatures 126, displayed-tree sets 319, major trees 369. The most frequent class has frequency 0.114 [0.096, 0.135]; 204 of the 319 classes were seen once and the estimated unseen mass at the class level is 0.204 +- 0.016. Of the plug-in entropy of the rooted-topology distribution (6.792 nats), 70.9% lies between displayed-tree sets, 0.0% between classes that display the same trees and 29.1% in the root position. In the 100-draw subset, 94.5% of the pairs (+- 1.2) have no displayed tree topology in common, 2.7% (+- 0.7) share one while their displayed sets differ, 0.0% (+- 0.0) are different classes displaying the same set of trees, 2.7% (+- 0.8) differ in root position only and 0.0% (+- 0.0) in continuous parameters only. The mean normalised Hausdorff RF between displayed-tree sets is 0.0529 +- 0.0043 and the mean normalised mu-distance between rooted networks is 0.1637 +- 0.0056.
These numbers describe one checkpoint and one sample of size 1000; intervals cover the Monte-Carlo variability of the draws only (not the variability between training runs).
(b) Accuracy of q_theta. Importance-weight ESS 20.3 of 1000 (2.0%): the trajectory weights are far from constant. Root-position uniformity z-scores pendant -2.19, internal 3.78, cycle -1.92: a measured mismatch between q_theta and the target in the root position. No independent reference posterior exists for this dataset. The diagnostics above are necessary conditions computed from the model itself: they can show that q_theta differs from the target and cannot show that it matches it. Whether q_theta approximates the posterior at the level of topology masses cannot be answered with the data in this output.

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
python tools/sample_posterior_newick.py /srv/scratch1/mmousabeigi/phylogfn-v3eng-bench/runs/20261030_225455_ds8_network_no_knn_300ep_lowmem dataset/benchmark_datasets/DS8.pickle --epoch 299 --expect-objective TB --n 1000 --seed 42 --subset 100 --subset-seed 42 --chunk 64 --device cpu --threads 4 --out-dir /srv/scratch1/mmousabeigi/posterior_v2_ds8_tb_1000 --dataset-name DS8
python tools/analyze_posterior_topology.py --samples-dir /srv/scratch1/mmousabeigi/posterior_v2_ds8_tb_1000 --sequences dataset/benchmark_datasets/DS8.pickle --seed 42
```
Tables: topology_summary.csv, topology_frequencies.csv, topology_frequencies_by_level.csv, displayed_tree_distribution.csv, entropy_decomposition.csv, accumulation_curves.csv, reticulation_count_vs_prior.csv, reticulation_signature_frequencies.csv, hybrid_clade_frequencies.csv, importance_reweighting.csv, root_position_uniformity.csv, pairwise_topology_distances.csv (4950 rows), pairwise_summary.csv, pair_classes.csv, pairwise_matrices.npz, duplicate_topology_parameter_variability.csv, score_by_topology.csv, validation_report.json. Figures in figures/.
