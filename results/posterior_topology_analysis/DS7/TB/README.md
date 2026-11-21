# Structural diversity of samples from the trained PhyloGFN-Net policy: DS7, checkpoint_000299.pt

Scope. This report describes n = 1000 independent draws from the learned distribution q_theta of one checkpoint. q_theta is trained to approximate the posterior p(x | Y); how closely it does so is a separate question, addressed in section 8 with the evidence that exists and the evidence that does not. Diversity statements below are statements about q_theta.

## 1. Questions
(a) Does the trained policy generate a diverse distribution over network topologies, and how much of that diversity corresponds to different displayed trees rather than to alternative reticulation representations, root positions or continuous parameters? (b) What evidence supports that the learned distribution approximates the intended posterior?

## 2. Checkpoint, data, provenance
Run directory `/srv/scratch1/mmousabeigi/phylogfn-v3eng-bench/runs/20261026_014933_ds7_network_no_knn_300ep_lowmem`; checkpoint `/srv/scratch1/mmousabeigi/phylogfn-v3eng-bench/runs/20261026_014933_ds7_network_no_knn_300ep_lowmem/checkpoints/checkpoint_000299.pt`, epoch 299 (IS the last configured epoch 299); SHA-256 `651ac6186e6bba6e1314d8232a4001d9bfd59586f41038db0d4a10c446e3f601`. Objective TB, training seed 1. Dataset `dataset/benchmark_datasets/DS7.pickle` (SHA-256 `f1c3cd066c5157d79d70b8843e1377fc7fa906e173c9150457b6769104d92bf9`): 59 taxa, 1824 sites. Prior: p(G) proportional to exp(-1 R) over rooted level-1 networks with R <= 2 (EXP_R), branch lengths Exp(10); reward constant C = 0; slot-label correction False. Code: commit None, None modified tracked files, uncommitted-diff SHA-256 None. Versions: python 3.10.21, torch 2.5.1+cu118, numpy 2.2.6, scipy 1.15.3, networkx 3.4.2, ete3 3.1.3, yacs unknown, matplotlib 3.10.9.
Source files executed by the sampler: 46, digest `5e40a9f3774d4045cf7bfd2c2ef66c695cc11d8b9495823a6eb8e91173b5383a` (SHA-256 of the list of their SHA-256; every file is listed in sampling_metadata.json, key code.source_files_sha256). The working tree was not a git repository: the digest is the only identification of the code.
Source files executed by this analysis: 44, digest `7cd50779382cd004d5442791ed3502db71e629efce74b5e0da1c1829ecb14ee2` (analysis_provenance.json).
Training log: final MLL -37140.950 +- 0.168 (3 estimates).
ENV.NETWORK options of the run: {"BLOCK_TRIANGLES": false, "CYCLE_MASK": "NONE", "LAMBDA_R": 1.0, "PRIOR_TYPE": "EXP_R", "R_MAX": 2, "SLOT_LABEL_CORRECTION": false}. The root-equivalence statistics below (root positions, class size, root uniformity) assume that the sampler generates every rooted binary level-1 network without parallel edges; the checkpoint audit verifies this for 4 taxa (line "class C" of checkpoint_audit.txt).
Options of the working tree that did not exist in the repository revision this pipeline was written against (reported, not interpreted; values of this run): ENV.NETWORK.BLOCK_TRIANGLES = False; ENV.NETWORK.CYCLE_MASK = 'NONE'; ENV.NETWORK.SLOT_LABEL_CORRECTION = False; GFN.MODEL.CONTINUOUS_BOUNDS = 'CLIP'; GFN.TRAINING_DATA_LOADER.LOW_MEMORY_TB = True. An option among them that changes the generated class of networks, the bounds of the continuous parameters, the reward or the backward policy changes what the samples and the diagnostics below mean.
The run configuration holds every key the sampling code defines: no default was filled in.

## 3. Sampling
RolloutWorkerNet.rollout(generator, k, generate_full_trajectories=True), random_spec=None, generator.eval(), torch.no_grad(), reward scale 1. 1000 draws in rollouts of 64 trajectories on cpu (4 threads), seeds: python/numpy/torch = 42. Every draw was kept. The replay buffer of best networks was not read. Subset: 100 draws, `numpy.random.default_rng(subset_seed).choice(n, size=subset, replace=False), sorted` with seed 42; the pairwise analysis uses this subset, all other statistics use the 1000 draws.

## 4. Format and validation
`posterior_samples_1000.newick`: one Extended Newick string per line; a reticulation is written `(subtree)#Hk:length::gamma` at its first parent and `#Hk:length::gamma` at its second; `gamma` is the inheritance probability of that hybrid edge; plain Newick when R = 0. The file names keep the word "posterior" requested for the analysis; the content is policy samples.
Independent validation by this script of all 1000 + 100 exported samples: every string parsed and structurally validated (acyclic, degrees, level-1, tree-child, parameter domains): 1000; string and edge record are the same rooted network edge by edge (branch lengths exactly, inheritance probabilities within 4.5e-16): 1000; every CSV column recomputed from the string: 1000; scores finite and consistent: 1000; subset rows identical to the corresponding full rows and subset indices reproduced from the seed: True; log score recomputed from every exported string with the explicit displayed-tree likelihood: max |difference| 2.91e-11. Failures: 0 (any failure stops the script). External parsers: this file was NOT checked by an external parser (tools/check_enewick_interop.py was not run on it). The format was verified with PhyloZoo and phylox on other files as documented in docs/posterior_topology/ENEWICK_INTEROPERABILITY.md; not tested with PhyloNetworks.jl, Dendroscope or IcyTree.

## 5. Levels of structure (definitions)
| level | object | what it ignores |
|---|---|---|
| L1 | rooted network topology: the object the policy generates (canonical string, injective on the supported class) | branch lengths, inheritance probabilities, slot labels, child order |
| L2 | root-equivalence class: L1 up to moving the root to any position that keeps every reticulation entered from above and gives no parallel edges ("semi-directed topology") | + root position |
| L3 | set of reticulation signatures: for every reticulation the hybrid descendant block and the cyclic sequence of pendant taxon blocks around its cycle | + structure inside the blocks |
| L4 | set of unrooted topologies of the displayed trees | + reticulations that leave every displayed tree unchanged |
| L5 | displayed-tree topology of largest summed inheritance weight (major tree) | depends on the inheritance probabilities |

The likelihood and the prior density take the same value on all members of an L2 class and re-rooting is a measure-preserving bijection of the parameters (tools/verify_root_equivalence.py), so under the target every member of a class has the same mass and the class has c(S) times that mass, c(S) = number of its rooted members. The model's prior is uniform over rooted networks given R, not over classes. L1 is the object of every distributional statement; L2 removes the root position, which the data cannot inform. Mean c(S) over the draws: 119.0.

## 6. Results: how many distinct structures (n = 1000 draws)
| level | distinct | distinct / n | most frequent: count, frequency [Wilson 95%] | singletons | entropy (nats) +- jackknife SE | perplexity | unseen mass (Good-Turing) +- SE | Chao1 lower bound +- SE |
|---|---|---|---|---|---|---|---|---|
| L1 rooted | 675 | 0.675 | 6, 0.006 [0.003, 0.013] | 474 | 6.374 +- 0.030 | 586.2 | 0.474 +- 0.022 | 1603 +- 124 |
| L2 class | 100 | 0.100 | 78, 0.078 [0.063, 0.096] | 36 | 3.753 +- 0.039 | 42.7 | 0.036 +- 0.008 | 138 +- 17 |
| L3 signatures | 6 | 0.006 | 654, 0.654 [0.624, 0.683] | 0 | 0.908 +- 0.028 | 2.5 | 0.000 +- 0.001 | 6 +- 0 |
| L4 displayed set | 100 | 0.100 | 78, 0.078 [0.063, 0.096] | 36 | 3.753 +- 0.039 | 42.7 | 0.036 +- 0.008 | 138 +- 17 |
| L5 major tree | 233 | 0.233 | 26, 0.026 [0.018, 0.038] | 99 | 4.987 +- 0.036 | 146.5 | 0.099 +- 0.012 | 391 +- 44 |

Reading the table. "distinct" and the entropy depend on n (accumulation_curves.csv and figure): with 474 of 675 L1 values seen once, the L1 count is limited by the sample size. The Wilson interval of the most frequent value is conditional on which value is most frequent (selected after seeing the data, so slightly optimistic). Unseen mass = estimated probability that the next draw from q_theta shows a value not yet seen; it concerns q_theta, not the support of the posterior. Chao1 is a lower bound for the number of values with non-negligible mass under q_theta, valid for independent draws; it is not an estimate of the size of the posterior support. No bootstrap interval is given for counts and entropies (the bootstrap is inconsistent for them).

Entropy decomposition of the L1 distribution (plug-in, nats; exact chain rule for nested partitions): between displayed-tree sets 3.753 (58.9%), between classes that display the same set of trees 0.000 (0.0%), root position within a class 2.620 (41.1%); total 6.374. Through reticulation signatures instead: between signature sets 0.908, between classes with the same signatures 2.845, root position 2.620.
Displayed trees weighted by inheritance: averaging the displayed-tree distributions of the 1000 draws gives 400 distinct unrooted tree topologies; the heaviest carries 0.020 of the weight; perplexity 170.56 (displayed_tree_distribution.csv). This is the distribution of the tree followed by a site under q_theta.

Most frequent root-equivalence classes (topology_frequencies.csv has all of them with the full canonical strings):

| rank | class | count | frequency [Wilson 95%] | R | rooted forms seen / possible | major tree (mode) | reweighted mass +- SE |
|---|---|---|---|---|---|---|---|
| 1 | b156de004aae | 78 | 0.078 [0.063, 0.096] | 2 | 28 / 119 | 775b16994be9 | not reported (ESS) |
| 2 | 77ecabce5ce9 | 77 | 0.077 [0.062, 0.095] | 2 | 34 / 119 | 3644f1d9fa10 | not reported (ESS) |
| 3 | 7521b250bc6e | 61 | 0.061 [0.048, 0.078] | 2 | 27 / 119 | 1c3c5081b391 | not reported (ESS) |
| 4 | 8fe8ef097bb8 | 41 | 0.041 [0.030, 0.055] | 2 | 25 / 119 | 716c2b33dbe1 (tie) | not reported (ESS) |
| 5 | a07926efaa9f | 38 | 0.038 [0.028, 0.052] | 2 | 25 / 119 | e812bf873937 | not reported (ESS) |
| 6 | b50e4cd508f7 | 37 | 0.037 [0.027, 0.051] | 2 | 23 / 119 | e8e5638a3219 | not reported (ESS) |
| 7 | 949fa4d5a4fe | 35 | 0.035 [0.025, 0.048] | 2 | 25 / 119 | ad34cc00f2db | not reported (ESS) |
| 8 | 8a35821b9484 | 35 | 0.035 [0.025, 0.048] | 2 | 21 / 119 | 77b57d27c1fc | not reported (ESS) |
| 9 | 4a61e82d1d81 | 35 | 0.035 [0.025, 0.048] | 2 | 22 / 119 | cb9ad62822d4 | not reported (ESS) |
| 10 | b31afdf95124 | 34 | 0.034 [0.024, 0.047] | 2 | 19 / 119 | a7910e5c6603 | not reported (ESS) |
| 11 | ba3e6dcade0e | 31 | 0.031 [0.022, 0.044] | 2 | 20 / 119 | 6d0726a67170 | not reported (ESS) |
| 12 | 0b6534751ebf | 30 | 0.030 [0.021, 0.043] | 2 | 21 / 119 | 2f83245bcd5c | not reported (ESS) |

## 7. Results: reticulations
| R | count | frequency [Wilson 95%] | prior of the target of this run (2^R slot labellings per rooted graph) | prior on rooted graphs |
|---|---|---|---|---|
| 0 | 0 | 0.0000 [0.000, 0.004] | 0.0000 | 0.0000 |
| 1 | 0 | 0.0000 [0.000, 0.004] | 0.0015 | 0.0030 |
| 2 | 1000 | 1.0000 [0.996, 1.000] | 0.9985 | 0.9970 |

The two prior columns are exact (counts of level-1 networks, src/utils/level1_counts.py). The prior mass on R = R_MAX = 2 is 0.9985 on slot-labelled objects and 0.9970 on rooted graphs: a sample from the prior alone shows these frequencies, so the frequency of R = R_MAX in the sample is read against them, not against 0. R_MAX is a configuration bound; nothing here is evidence for a number of reticulation events in the history of the taxa.
Odds of R = 2 against R < 2. Sample (ends of the Wilson interval): 260 to unbounded. Prior: 674 (slot-labelled objects), 337 (rooted graphs). Sample odds divided by prior odds: 0.386 to unbounded (slot-labelled objects), 0.772 to unbounded (rooted graphs). This factor is the contribution of the data only if q_theta equals the posterior; a factor near 1 means the sample says about R what the prior already says.
Cycle lengths (semi-directed view): {"8": 1016, "9": 318, "10": 666}. 0 of 2000 reticulations (0.0%) have cycle length <= 3: switching such a reticulation does not change the unrooted topology of the displayed tree (it changes branch lengths only). 0 of 1000 draws (0.000 [0.000, 0.004]) contain at least one such reticulation; 0 draws (0.000) have only such reticulations. This is a structural fact about the sampled networks; by itself it says nothing about identifiability or about hybridisation.

Most frequent reticulation signatures (hybrid block | cyclic sequence of the other blocks; a block with more than three taxa is abbreviated to its first two names and the number of the others; the full lists are in reticulation_signature_frequencies.csv):

| hybrid block | cycle blocks | cycle length | draws containing it | fraction of draws [Wilson 95%] | share of all reticulations |
|---|---|---|---|---|---|
| Propithecus_tattersalli | Hapalemur_griseus / Lemur_catta / Eulemur_fulvus_albifrons, Eulemur_fulvus_collaris + 5 more / Varecia_variegata_rubra, Varecia_variegata_variegata / Cheirogaleus_major, Microcebus__rufus_1.138 + 25 more / Balaenoptera_physalus, Bos_taurus + 17 more / Lepilemur_ruficaudatus | 8 | 1000 | 1.000 [0.996, 1.000] | 0.500 |
| Didelphis_virginiana | Canis_familiaris / Ursus_americanus / Felis_catus / Balaenoptera_physalus, Bos_taurus + 2 more / Equus_caballus, Rhinocerus / Callithrix_jacchus, Gorilla_gorilla + 4 more / Galago_demidoff, Loris_tardigradus / Cheirogaleus_major, Eulemur_fulvus_albifrons + 38 more / Daubentonia | 10 | 654 | 0.654 [0.624, 0.683] | 0.327 |
| Didelphis_virginiana | Canis_familiaris / Felis_catus, Ursus_americanus / Balaenoptera_physalus, Bos_taurus + 2 more / Equus_caballus, Rhinocerus / Callithrix_jacchus, Gorilla_gorilla + 4 more / Galago_demidoff, Loris_tardigradus / Cheirogaleus_major, Eulemur_fulvus_albifrons + 38 more / Daubentonia | 9 | 268 | 0.268 [0.241, 0.296] | 0.134 |
| Didelphis_virginiana | Canis_familiaris / Ursus_americanus / Felis_catus / Balaenoptera_physalus, Bos_taurus + 4 more / Callithrix_jacchus, Gorilla_gorilla + 4 more / Galago_demidoff, Loris_tardigradus / Cheirogaleus_major, Eulemur_fulvus_albifrons + 38 more / Daubentonia | 9 | 48 | 0.048 [0.036, 0.063] | 0.024 |
| Didelphis_virginiana | Canis_familiaris / Felis_catus, Ursus_americanus / Balaenoptera_physalus, Bos_taurus + 4 more / Callithrix_jacchus, Gorilla_gorilla + 4 more / Galago_demidoff, Loris_tardigradus / Cheirogaleus_major, Eulemur_fulvus_albifrons + 38 more / Daubentonia | 8 | 16 | 0.016 [0.010, 0.026] | 0.008 |
| Didelphis_virginiana | Canis_familiaris / Felis_catus / Ursus_americanus / Balaenoptera_physalus, Bos_taurus + 2 more / Equus_caballus, Rhinocerus / Callithrix_jacchus, Gorilla_gorilla + 4 more / Galago_demidoff, Loris_tardigradus / Cheirogaleus_major, Eulemur_fulvus_albifrons + 38 more / Daubentonia | 10 | 12 | 0.012 [0.007, 0.021] | 0.006 |
| Didelphis_virginiana | Canis_familiaris / Felis_catus / Ursus_americanus / Balaenoptera_physalus, Bos_taurus + 4 more / Callithrix_jacchus, Gorilla_gorilla + 4 more / Galago_demidoff, Loris_tardigradus / Cheirogaleus_major, Eulemur_fulvus_albifrons + 38 more / Daubentonia | 9 | 2 | 0.002 [0.001, 0.007] | 0.001 |

A hybrid descendant clade alone does not fix the placement: 2 distinct hybrid clades correspond to 7 distinct signatures in this sample (hybrid_clade_frequencies.csv counts draws and occurrences separately).

## 8. Does the learned distribution approximate the posterior? Evidence available in this output
When q_theta = p. If the trajectory-balance residual is zero on every complete trajectory that has positive probability under P_F or P_B (exact balance with full support), with the uniform backward policy over the parents of each state (the parent count is checked against the backward sampler in tests/test_network_env.py), then the terminal density of the policy equals R(x) / Z. Here R(x) = likelihood x branch-length prior x exp(-LAMBDA_R R) and each rooted graph is reached as 2^R slot-labelled terminal objects, so the graph-level target is proportional to 2^R R(x) unless the slot-label correction is on (it is OFF for this run). Training gives approximate balance on the trajectories it visited. None of this is guaranteed to hold at convergence of a finite run.
Clipped parameters. The sampler clips log branch lengths to [-10, 0] and logit(theta) to +-8 and evaluates log P_F with the unclipped density at the clipped value. A parameter on a bound is therefore an atom of the sampling distribution for which log P_F is not a probability. Draws with at least one parameter on a bound: 234 of 1000 (23.40%). Independent of the configured bounds: among the 121000 sampled lengths the smallest (4.540e-05) occurs 257 time(s) and the largest (0.2420) occurs 1 time(s); a continuous distribution gives each once. The target has a density, so it gives these draws no mass: the Radon-Nikodym derivative of the target with respect to the sampling distribution is 0 on an atom. Their importance weight is therefore 0 in (i). The value the training code forms for such a draw (score divided by the unclipped density at the bound) is not a weight with respect to any measure; it is shown in (i) only for comparison with the training log. The fraction of draws on a bound, 0.2340 [0.209, 0.261], estimates the mass that q_theta puts where the target has none: the total variation between q_theta and any distribution with a density is at least this mass. log p_T of these draws is not a density either: they are left out of the regression in (iii). A draw is counted as on a bound when its log length is within 1e-6 of the lower or 1e-9 of the upper bound, or its logit(theta) within 1e-3 of a bound (single-precision resolution of the stored value).

(i) Importance weights. w_i = R(x_i) P_B(tau_i | x_i) / P_F(tau_i) for the 1000 sampled trajectories. Exact balance makes w constant. Kish effective sample size 5.8 of 1000 (0.6%); standard deviation of log w 3.362 nats; largest normalised weight 0.336. The implied MLL estimate is -37140.666 (delta-method SE 0.415, unreliable when the ESS is small), against -37140.950 +- 0.168 in the training log (the training log forms a weight for every draw, also for a draw on a clipping bound; without such draws the two estimators are the same, and the comparison is a functional check that checkpoint, dataset and code agree). 234 of the 1000 draws are on a clipping bound and have weight 0 here. With the weight the training code forms for them: ESS 7.3, MLL estimate -37140.534 (delta-method SE 0.369); that estimator is biased upwards by the mass of the atoms times the ratio of the score to the unclipped density at the bound. The estimate with weight 0 targets the mass of the posterior inside the clipping box; the posterior mass outside the box cannot be reached by the sampler and is not estimated.
The ESS is a trajectory-level quantity. The weight of a terminal object is the conditional mean of the trajectory weights that end in it, so the trajectory-level ESS cannot exceed the ESS at the level of terminal objects; it is small whenever a few trajectories dominate. With ESS / n = 0.006 < 0.1 the reweighted masses below are not interpretable and are not used.

| quantity | frequency in the sample | importance-reweighted mass +- SE |
|---|---|---|
| R = 2 | 1.0000 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| root_position_type = internal | 0.3540 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| root_position_type = pendant | 0.3370 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| root_position_type = cycle | 0.3090 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = 22cce09150a4 | 0.0780 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = d4be80a13472 | 0.0770 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = 68ebe4cca145 | 0.0610 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = 81b59302d013 | 0.0410 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = 9f8093789c3b | 0.0380 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = 3644f1d9fa10 | 0.0260 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = 775b16994be9 | 0.0230 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = ccc147c5045b | 0.0210 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = 2ac3e719cff0 | 0.0200 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = 1c11ae44b975 | 0.0190 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |

(ii) Root-position uniformity. Under the target, given the semi-directed network and its parameters, the root is uniformly distributed over the in-class root positions (equal density and unit Jacobian under re-rooting; tools/verify_root_equivalence.py). Observed versus expected number of draws by root-position type: pendant edge 337 vs 479.0 (sd 15.8, z = -8.99); internal edge 354 vs 372.7 (sd 15.3, z = -1.22); cycle edge 309 vs 148.3 (sd 11.2, z = 14.30). A necessary condition for q_theta = p that needs no reference; it tests one coordinate only. Here max |z| = 14.3: q_theta is measurably different from the target in where it puts the root.

(iii) Terminal density: the optional stage tools/posterior_density_diagnostics.py was not run for this output.

(iv) Independent reference posterior: NONE for this dataset. No exact or independently sampled posterior over level-1 networks exists for it, so the accuracy of q_theta at the level of topology masses is NOT established by this output. Diagnostics (i)-(iii) are necessary conditions computed from the model's own quantities; they can reveal a mismatch and cannot prove a match. On small simulated problems the comparison with a numerical reference is available (--reference).

## 9. Results: pairwise comparison of the 100-draw subset (4950 unordered pairs)
The pairs share draws: the uncertainty of a mean over pairs is the jackknife standard error over the 100 draws, not sd / sqrt(4950).

| quantity | what it is | mean over pairs +- jackknife SE | median | min | max | pairs at 0 | mean self value |
|---|---|---|---|---|---|---|---|
| mu_distance_norm | metric | 0.1112 +- 0.0031 | 0.1129 | 0.0000 | 0.2581 | 0.2% | 0.0000 |
| dt_rf_hausdorff_norm | pseudo-metric | 0.0390 +- 0.0016 | 0.0357 | 0.0000 | 0.0893 | 4.2% | 0.0000 |
| dt_jaccard_distance | pseudo-metric | 0.9580 +- 0.0060 | 1.0000 | 0.0000 | 1.0000 | 4.2% | 0.0000 |
| dt_tv | pseudo-metric | 0.9606 +- 0.0057 | 1.0000 | 0.0064 | 1.0000 | 0.0% | 0.0000 |
| dt_js_distance | pseudo-metric | 0.9605 +- 0.0057 | 1.0000 | 0.0055 | 1.0000 | 0.0% | 0.0000 |
| split_l1_norm | pseudo-metric | 0.0487 +- 0.0016 | 0.0472 | 0.0009 | 0.1094 | 0.0% | 0.0000 |
| major_rf_norm | pseudo-metric | 0.1426 +- 0.0012 | 0.1429 | 0.0000 | 0.2679 | 1.2% | 0.0000 |
| dt_rf_min_norm | dissimilarity | 0.0390 +- 0.0016 | 0.0357 | 0.0000 | 0.0893 | 4.2% | 0.0000 |
| sig_jaccard_distance | pseudo-metric | 0.2671 +- 0.0351 | 0.0000 | 0.0000 | 0.6667 | 59.9% | 0.0000 |
| hybrid_clade_jaccard_distance | pseudo-metric | 0.0000 +- 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0% | 0.0000 |
| dt_rf_mean_norm | descriptive statistic | 0.1417 +- 0.0011 | 0.1429 | 0.0893 | 0.1875 | 0.0% | 0.1048 |
| dt_rf_expected_norm | descriptive statistic | 0.1417 +- 0.0011 | 0.1429 | 0.0887 | 0.1880 | 0.0% | 0.1041 |

Normalised quantities are divided by 2(N - 3) = 112 (unrooted RF maximum) or, for the mu-distance, by the number of internal nodes of the two networks. "mean self value" is the quantity of a network compared with itself, averaged over the subset: it is 0 for every metric, pseudo-metric and dissimilarity; for the two descriptive statistics it is positive as soon as a network displays more than one tree topology, which is why they are not distances. A pseudo-metric is 0 for different networks that share the derived object named in the definitions.

Pair classes (mutually exclusive, decided by the definitions; fractions +- jackknife SE over draws):

| class | pairs | fraction +- SE | of which same major tree | same R | same signature set |
|---|---|---|---|---|---|
| P1 same rooted topology | 9 | 0.0018 +- 0.0009 | 2 | 9 | 9 |
| P2 same class different rooting | 199 | 0.0402 +- 0.0057 | 58 | 199 | 199 |
| P3 different class same displayed set | 0 | 0.0000 +- 0.0000 | 0 | 0 | 0 |
| P4 displayed sets differ share a topology | 0 | 0.0000 +- 0.0000 | 0 | 0 | 0 |
| P5 no displayed topology in common | 4742 | 0.9580 +- 0.0060 | 0 | 4742 | 2759 |

P1: only continuous parameters differ. P2: the root position differs. P3: the network topologies differ and display the same set of tree topologies. P4: the displayed sets differ and share at least one tree topology. P5: no displayed tree topology in common. The classes do not say why two networks differ beyond these definitions.

## 10. Results: continuous parameters (kept separate from topology)
Total branch length over the 1000 draws: mean 4.3110, sd 0.0702. Inheritance probability of the heavier hybrid edge over 2000 reticulations: mean 0.533, sd 0.025. 64 root-equivalence classes occur at least twice (964 draws). Within them every edge and every inheritance probability was matched across draws through root-invariant taxon-based keys (all 64 edge-key sets identical within a class; a mismatch stops the script): median over classes of the mean coefficient of variation of matched branch lengths 0.251; mean sd of matched inheritance probabilities 0.034; mean coefficient of variation of the total length 0.014 (duplicate_topology_parameter_variability.csv).

## 11. Scores
log score = log R(x) = log-likelihood + log branch-length prior + log topology prior (unnormalised), a pointwise density value of one draw with its continuous parameters. Over the 1000 draws: mean -36398.26, sd 8.30, range -36446.49 to -36376.80. A topology with a high pointwise score need not have a large posterior mass: mass is the integral of R over the continuous parameters, times the number of labellings. The frequency of a topology in the sample estimates its mass under q_theta; the reweighted mass (section 8) estimates its mass under the target when the ESS allows. score_by_topology.csv lists, for the most frequent classes, the frequency with its interval, the pointwise score summary and the reweighted mass, each in its own column. Spearman correlation between the pointwise log score and the split-frequency distance to the most frequent class in the subset: rho = -0.034 (n = 100 draws; descriptive).

## 12. Answers supported by this output
(a) Diversity of q_theta. Number of distinct values among 1000 draws: rooted topologies 675, root-equivalence classes 100, sets of reticulation signatures 6, displayed-tree sets 100, major trees 233. The most frequent class has frequency 0.078 [0.063, 0.096]; 36 of the 100 classes were seen once and the estimated unseen mass at the class level is 0.036 +- 0.008. Of the plug-in entropy of the rooted-topology distribution (6.374 nats), 58.9% lies between displayed-tree sets, 0.0% between classes that display the same trees and 41.1% in the root position. In the 100-draw subset, 95.8% of the pairs (+- 0.6) have no displayed tree topology in common, 0.0% (+- 0.0) share one while their displayed sets differ, 0.0% (+- 0.0) are different classes displaying the same set of trees, 4.0% (+- 0.6) differ in root position only and 0.2% (+- 0.1) in continuous parameters only. The mean normalised Hausdorff RF between displayed-tree sets is 0.0390 +- 0.0016 and the mean normalised mu-distance between rooted networks is 0.1112 +- 0.0031.
These numbers describe one checkpoint and one sample of size 1000; intervals cover the Monte-Carlo variability of the draws only (not the variability between training runs).
(b) Accuracy of q_theta. Importance-weight ESS 5.8 of 1000 (0.6%): the trajectory weights are far from constant. Root-position uniformity z-scores pendant -8.99, internal -1.22, cycle 14.30: a measured mismatch between q_theta and the target in the root position. No independent reference posterior exists for this dataset. The diagnostics above are necessary conditions computed from the model itself: they can show that q_theta differs from the target and cannot show that it matches it. Whether q_theta approximates the posterior at the level of topology masses cannot be answered with the data in this output.

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
python tools/sample_posterior_newick.py /srv/scratch1/mmousabeigi/phylogfn-v3eng-bench/runs/20261026_014933_ds7_network_no_knn_300ep_lowmem dataset/benchmark_datasets/DS7.pickle --epoch 299 --expect-objective TB --n 1000 --seed 42 --subset 100 --subset-seed 42 --chunk 64 --device cpu --threads 4 --out-dir /srv/scratch1/mmousabeigi/posterior_v2_ds7_tb_1000 --dataset-name DS7
python tools/analyze_posterior_topology.py --samples-dir /srv/scratch1/mmousabeigi/posterior_v2_ds7_tb_1000 --sequences dataset/benchmark_datasets/DS7.pickle --seed 42
```
Tables: topology_summary.csv, topology_frequencies.csv, topology_frequencies_by_level.csv, displayed_tree_distribution.csv, entropy_decomposition.csv, accumulation_curves.csv, reticulation_count_vs_prior.csv, reticulation_signature_frequencies.csv, hybrid_clade_frequencies.csv, importance_reweighting.csv, root_position_uniformity.csv, pairwise_topology_distances.csv (4950 rows), pairwise_summary.csv, pair_classes.csv, pairwise_matrices.npz, duplicate_topology_parameter_variability.csv, score_by_topology.csv, validation_report.json. Figures in figures/.
