# Structural diversity of samples from the trained PhyloGFN-Net policy: DS1, checkpoint_000299.pt

Scope. This report describes n = 1000 independent draws from the learned distribution q_theta of one checkpoint. q_theta is trained to approximate the posterior p(x | Y); how closely it does so is a separate question, addressed in section 8 with the evidence that exists and the evidence that does not. Diversity statements below are statements about q_theta.

## 1. Questions
(a) Does the trained policy generate a diverse distribution over network topologies, and how much of that diversity corresponds to different displayed trees rather than to alternative reticulation representations, root positions or continuous parameters? (b) What evidence supports that the learned distribution approximates the intended posterior?

## 2. Checkpoint, data, provenance
Run directory `runs_ds1_complete/20261113_220512_ds1_baseline_resume_s1`; checkpoint `runs_ds1_complete/20261113_220512_ds1_baseline_resume_s1/checkpoints/checkpoint_000299.pt`, epoch 299 (IS the last configured epoch 299); SHA-256 `77c6014982552e8fab55d536584d4c74e3eaef6d6d5fe52c00f2c758de228e9b`. Objective TB, training seed 1. Dataset `dataset/benchmark_datasets/DS1.pickle` (SHA-256 `dbdaed72f67b69058b8f6208dd0ac7e55c89878e88108da08dc3f450b1ae729c`): 27 taxa, 1949 sites. Prior: p(G) proportional to exp(-1 R) over rooted level-1 networks with R <= 2 (EXP_R), branch lengths Exp(10); reward constant C = 0; slot-label correction False. Code: commit 788584912f175cc9d8480cbc3a6d39d9cf3e1530, 0 modified tracked files, uncommitted-diff SHA-256 None. Versions: python 3.10.21, torch 2.5.1+cu118, numpy 2.2.6, scipy 1.15.3, networkx 3.4.2, ete3 3.1.3, yacs unknown, matplotlib 3.10.9.
Source files executed by the sampler: 46, digest `f206017e9cf31ed8839669ac8d279c036de8529896527c268894149b5f2b7235` (SHA-256 of the list of their SHA-256; every file is listed in sampling_metadata.json, key code.source_files_sha256). 6 of these files are not tracked by git and 0 are tracked with uncommitted changes: the commit above does not identify them, the digest does.
Source files executed by this analysis: 44, digest `06a10b214798d5d2a87c140bbeaac122c02169b429bdb876e48eab236f4ba934` (analysis_provenance.json).
Training log: final MLL -7043.143 +- 0.104 (3 estimates).
ENV.NETWORK options of the run: {"LAMBDA_R": 1.0, "PRIOR_TYPE": "EXP_R", "R_MAX": 2}. The root-equivalence statistics below (root positions, class size, root uniformity) assume that the sampler generates every rooted binary level-1 network without parallel edges; the checkpoint audit verifies this for 4 taxa (line "class C" of checkpoint_audit.txt).
Every configuration option of the working tree existed in the repository revision this pipeline was written against.
The run configuration holds every key the sampling code defines: no default was filled in.

## 3. Sampling
RolloutWorkerNet.rollout(generator, k, generate_full_trajectories=True), random_spec=None, generator.eval(), torch.no_grad(), reward scale 1. 1000 draws in rollouts of 64 trajectories on cpu (4 threads), seeds: python/numpy/torch = 42. Every draw was kept. The replay buffer of best networks was not read. Subset: 100 draws, `numpy.random.default_rng(subset_seed).choice(n, size=subset, replace=False), sorted` with seed 42; the pairwise analysis uses this subset, all other statistics use the 1000 draws.

## 4. Format and validation
`posterior_samples_1000.newick`: one Extended Newick string per line; a reticulation is written `(subtree)#Hk:length::gamma` at its first parent and `#Hk:length::gamma` at its second; `gamma` is the inheritance probability of that hybrid edge; plain Newick when R = 0. The file names keep the word "posterior" requested for the analysis; the content is policy samples.
Independent validation by this script of all 1000 + 100 exported samples: every string parsed and structurally validated (acyclic, degrees, level-1, tree-child, parameter domains): 1000; string and edge record are the same rooted network edge by edge (branch lengths exactly, inheritance probabilities within 4.5e-16): 1000; every CSV column recomputed from the string: 1000; scores finite and consistent: 1000; subset rows identical to the corresponding full rows and subset indices reproduced from the seed: True; log score recomputed from every exported string with the explicit displayed-tree likelihood: max |difference| 1.55e-11. Failures: 0 (any failure stops the script). External parsers: this file was NOT checked by an external parser (tools/check_enewick_interop.py was not run on it). The format was verified with PhyloZoo and phylox on other files as documented in docs/posterior_topology/ENEWICK_INTEROPERABILITY.md; not tested with PhyloNetworks.jl, Dendroscope or IcyTree.

## 5. Levels of structure (definitions)
| level | object | what it ignores |
|---|---|---|
| L1 | rooted network topology: the object the policy generates (canonical string, injective on the supported class) | branch lengths, inheritance probabilities, slot labels, child order |
| L2 | root-equivalence class: L1 up to moving the root to any position that keeps every reticulation entered from above and gives no parallel edges ("semi-directed topology") | + root position |
| L3 | set of reticulation signatures: for every reticulation the hybrid descendant block and the cyclic sequence of pendant taxon blocks around its cycle | + structure inside the blocks |
| L4 | set of unrooted topologies of the displayed trees | + reticulations that leave every displayed tree unchanged |
| L5 | displayed-tree topology of largest summed inheritance weight (major tree) | depends on the inheritance probabilities |

The likelihood and the prior density take the same value on all members of an L2 class and re-rooting is a measure-preserving bijection of the parameters (tools/verify_root_equivalence.py), so under the target every member of a class has the same mass and the class has c(S) times that mass, c(S) = number of its rooted members. The model's prior is uniform over rooted networks given R, not over classes. L1 is the object of every distributional statement; L2 removes the root position, which the data cannot inform. Mean c(S) over the draws: 45.0.

## 6. Results: how many distinct structures (n = 1000 draws)
| level | distinct | distinct / n | most frequent: count, frequency [Wilson 95%] | singletons | entropy (nats) +- jackknife SE | perplexity | unseen mass (Good-Turing) +- SE | Chao1 lower bound +- SE |
|---|---|---|---|---|---|---|---|---|
| L1 rooted | 537 | 0.537 | 11, 0.011 [0.006, 0.020] | 357 | 6.002 +- 0.038 | 404.0 | 0.357 +- 0.020 | 1324 +- 124 |
| L2 class | 100 | 0.100 | 240, 0.240 [0.215, 0.267] | 45 | 3.064 +- 0.057 | 21.4 | 0.045 +- 0.009 | 156 +- 23 |
| L3 signatures | 50 | 0.050 | 502, 0.502 [0.471, 0.533] | 20 | 2.060 +- 0.056 | 7.8 | 0.020 +- 0.006 | 83 +- 21 |
| L4 displayed set | 100 | 0.100 | 240, 0.240 [0.215, 0.267] | 45 | 3.064 +- 0.057 | 21.4 | 0.045 +- 0.009 | 156 +- 23 |
| L5 major tree | 106 | 0.106 | 170, 0.170 [0.148, 0.195] | 39 | 3.372 +- 0.053 | 29.1 | 0.039 +- 0.009 | 144 +- 16 |

Reading the table. "distinct" and the entropy depend on n (accumulation_curves.csv and figure): with 357 of 537 L1 values seen once, the L1 count is limited by the sample size. The Wilson interval of the most frequent value is conditional on which value is most frequent (selected after seeing the data, so slightly optimistic). Unseen mass = estimated probability that the next draw from q_theta shows a value not yet seen; it concerns q_theta, not the support of the posterior. Chao1 is a lower bound for the number of values with non-negligible mass under q_theta, valid for independent draws; it is not an estimate of the size of the posterior support. No bootstrap interval is given for counts and entropies (the bootstrap is inconsistent for them).

Entropy decomposition of the L1 distribution (plug-in, nats; exact chain rule for nested partitions): between displayed-tree sets 3.064 (51.1%), between classes that display the same set of trees 0.000 (0.0%), root position within a class 2.937 (48.9%); total 6.002. Through reticulation signatures instead: between signature sets 2.060, between classes with the same signatures 1.004, root position 2.937.
Displayed trees weighted by inheritance: averaging the displayed-tree distributions of the 1000 draws gives 216 distinct unrooted tree topologies; the heaviest carries 0.118 of the weight; perplexity 42.86 (displayed_tree_distribution.csv). This is the distribution of the tree followed by a site under q_theta.

Most frequent root-equivalence classes (topology_frequencies.csv has all of them with the full canonical strings):

| rank | class | count | frequency [Wilson 95%] | R | rooted forms seen / possible | major tree (mode) | reweighted mass +- SE |
|---|---|---|---|---|---|---|---|
| 1 | 1573634fc582 | 240 | 0.240 [0.215, 0.267] | 2 | 45 / 45 | 7b905af9c1ec | not reported (ESS) |
| 2 | 7c21abd8c286 | 186 | 0.186 [0.163, 0.211] | 2 | 45 / 45 | 30e62211c363 | not reported (ESS) |
| 3 | c38c167f2523 | 68 | 0.068 [0.054, 0.085] | 2 | 33 / 45 | 7b905af9c1ec | not reported (ESS) |
| 4 | 1020510af9fb | 56 | 0.056 [0.043, 0.072] | 2 | 34 / 45 | ccd8087d4a6a | not reported (ESS) |
| 5 | 0c46908af162 | 43 | 0.043 [0.032, 0.057] | 2 | 27 / 45 | 1ab57c313cb4 | not reported (ESS) |
| 6 | 4251c26c5689 | 39 | 0.039 [0.029, 0.053] | 2 | 22 / 45 | 30e62211c363 | not reported (ESS) |
| 7 | 4c88bff2e748 | 26 | 0.026 [0.018, 0.038] | 2 | 19 / 45 | 7b905af9c1ec (tie) | not reported (ESS) |
| 8 | 140c2abf4e41 | 24 | 0.024 [0.016, 0.035] | 2 | 20 / 45 | 30e62211c363 | not reported (ESS) |
| 9 | 30ac2db7f54f | 21 | 0.021 [0.014, 0.032] | 2 | 16 / 45 | cc90a638e988 | not reported (ESS) |
| 10 | 0f8bd0736ec1 | 20 | 0.020 [0.013, 0.031] | 2 | 15 / 45 | c33116293321 | not reported (ESS) |
| 11 | 550021fda916 | 18 | 0.018 [0.011, 0.028] | 2 | 14 / 45 | 3e27233b9949 | not reported (ESS) |
| 12 | 10a588ffa275 | 18 | 0.018 [0.011, 0.028] | 2 | 14 / 45 | fce2eac7af4f | not reported (ESS) |

## 7. Results: reticulations
| R | count | frequency [Wilson 95%] | prior of the target of this run (2^R slot labellings per rooted graph) | prior on rooted graphs |
|---|---|---|---|---|
| 0 | 0 | 0.0000 [0.000, 0.004] | 0.0000 | 0.0000 |
| 1 | 0 | 0.0000 [0.000, 0.004] | 0.0079 | 0.0157 |
| 2 | 1000 | 1.0000 [0.996, 1.000] | 0.9921 | 0.9843 |

The two prior columns are exact (counts of level-1 networks, src/utils/level1_counts.py). The prior mass on R = R_MAX = 2 is 0.9921 on slot-labelled objects and 0.9843 on rooted graphs: a sample from the prior alone shows these frequencies, so the frequency of R = R_MAX in the sample is read against them, not against 0. R_MAX is a configuration bound; nothing here is evidence for a number of reticulation events in the history of the taxa.
Odds of R = 2 against R < 2. Sample (ends of the Wilson interval): 260 to unbounded. Prior: 125 (slot-labelled objects), 62.5 (rooted graphs). Sample odds divided by prior odds: 2.08 to unbounded (slot-labelled objects), 4.16 to unbounded (rooted graphs). This factor is the contribution of the data only if q_theta equals the posterior; a factor near 1 means the sample says about R what the prior already says.
Cycle lengths (semi-directed view): {"6": 1000, "7": 22, "8": 206, "9": 703, "10": 58, "11": 11}. 0 of 2000 reticulations (0.0%) have cycle length <= 3: switching such a reticulation does not change the unrooted topology of the displayed tree (it changes branch lengths only). 0 of 1000 draws (0.000 [0.000, 0.004]) contain at least one such reticulation; 0 draws (0.000) have only such reticulations. This is a structural fact about the sampled networks; by itself it says nothing about identifiability or about hybridisation.

Most frequent reticulation signatures (hybrid block | cyclic sequence of the other blocks; a block with more than three taxa is abbreviated to its first two names and the number of the others; the full lists are in reticulation_signature_frequencies.csv):

| hybrid block | cycle blocks | cycle length | draws containing it | fraction of draws [Wilson 95%] | share of all reticulations |
|---|---|---|---|---|---|
| Hyla_cinerea | Bufo_valliceps / Alligator_mississippiensis, Ambystoma_mexicanum + 20 more / Gastrophryne_carolinensis / Nesomantis_thomasseti / Eleutherodactylus_cuneatus | 6 | 891 | 0.891 [0.870, 0.909] | 0.446 |
| Alligator_mississippiensis, Gallus_gallus + 4 more | Homo_sapiens, Mus_musculus + 2 more / Latimeria_chalumnae / Xenopus_laevis / Bufo_valliceps, Eleutherodactylus_cuneatus + 3 more / Amphiuma_tridactylum, Discoglossus_pictus + 5 more / Typhlonectes_natans / Ambystoma_mexicanum / Siren_intermedia | 9 | 561 | 0.561 [0.530, 0.591] | 0.281 |
| Alligator_mississippiensis, Gallus_gallus + 4 more | Ambystoma_mexicanum, Siren_intermedia / Typhlonectes_natans / Amphiuma_tridactylum, Discoglossus_pictus + 5 more / Bufo_valliceps, Eleutherodactylus_cuneatus + 3 more / Xenopus_laevis / Latimeria_chalumnae / Homo_sapiens, Mus_musculus + 2 more | 8 | 141 | 0.141 [0.121, 0.164] | 0.070 |
| Alligator_mississippiensis, Gallus_gallus + 4 more | Ambystoma_mexicanum / Siren_intermedia / Typhlonectes_natans / Amphiuma_tridactylum, Discoglossus_pictus + 5 more / Bufo_valliceps, Eleutherodactylus_cuneatus + 3 more / Xenopus_laevis / Latimeria_chalumnae / Homo_sapiens, Mus_musculus + 2 more | 9 | 86 | 0.086 [0.070, 0.105] | 0.043 |
| Nesomantis_thomasseti | Alligator_mississippiensis, Ambystoma_mexicanum + 20 more / Bufo_valliceps / Hyla_cinerea / Gastrophryne_carolinensis / Eleutherodactylus_cuneatus | 6 | 80 | 0.080 [0.065, 0.098] | 0.040 |
| Alligator_mississippiensis, Gallus_gallus + 4 more | Homo_sapiens, Mus_musculus + 2 more / Latimeria_chalumnae / Xenopus_laevis / Bufo_valliceps, Eleutherodactylus_cuneatus + 3 more / Amphiuma_tridactylum, Discoglossus_pictus + 5 more / Ambystoma_mexicanum, Siren_intermedia / Typhlonectes_natans | 8 | 53 | 0.053 [0.041, 0.069] | 0.026 |
| Alligator_mississippiensis, Gallus_gallus + 4 more | Homo_sapiens, Mus_musculus + 2 more / Latimeria_chalumnae / Xenopus_laevis / Bufo_valliceps, Eleutherodactylus_cuneatus + 3 more / Typhlonectes_natans / Amphiuma_tridactylum, Discoglossus_pictus + 5 more / Ambystoma_mexicanum / Siren_intermedia | 9 | 41 | 0.041 [0.030, 0.055] | 0.021 |
| Alligator_mississippiensis, Gallus_gallus + 4 more | Homo_sapiens, Mus_musculus, Rattus_norvegicus / Oryctolagus_cuniculus / Latimeria_chalumnae / Xenopus_laevis / Bufo_valliceps, Eleutherodactylus_cuneatus + 3 more / Amphiuma_tridactylum, Discoglossus_pictus + 5 more / Typhlonectes_natans / Ambystoma_mexicanum / Siren_intermedia | 10 | 26 | 0.026 [0.018, 0.038] | 0.013 |
| Hyla_cinerea | Bufo_valliceps / Alligator_mississippiensis, Ambystoma_mexicanum + 20 more / Nesomantis_thomasseti / Gastrophryne_carolinensis / Eleutherodactylus_cuneatus | 6 | 22 | 0.022 [0.015, 0.033] | 0.011 |
| Alligator_mississippiensis, Gallus_gallus + 4 more | Ambystoma_mexicanum, Siren_intermedia, Typhlonectes_natans / Amphiuma_tridactylum, Discoglossus_pictus + 5 more / Bufo_valliceps, Eleutherodactylus_cuneatus + 3 more / Xenopus_laevis / Latimeria_chalumnae / Homo_sapiens, Mus_musculus + 2 more | 7 | 21 | 0.021 [0.014, 0.032] | 0.011 |

A hybrid descendant clade alone does not fix the placement: 4 distinct hybrid clades correspond to 34 distinct signatures in this sample (hybrid_clade_frequencies.csv counts draws and occurrences separately).

## 8. Does the learned distribution approximate the posterior? Evidence available in this output
When q_theta = p. If the trajectory-balance residual is zero on every complete trajectory that has positive probability under P_F or P_B (exact balance with full support), with the uniform backward policy over the parents of each state (the parent count is checked against the backward sampler in tests/test_network_env.py), then the terminal density of the policy equals R(x) / Z. Here R(x) = likelihood x branch-length prior x exp(-LAMBDA_R R) and each rooted graph is reached as 2^R slot-labelled terminal objects, so the graph-level target is proportional to 2^R R(x) unless the slot-label correction is on (it is OFF for this run). Training gives approximate balance on the trajectories it visited. None of this is guaranteed to hold at convergence of a finite run.
Clipped parameters. The sampler clips log branch lengths to [-10, 0] and logit(theta) to +-8 and evaluates log P_F with the unclipped density at the clipped value. A parameter on a bound is therefore an atom of the sampling distribution for which log P_F is not a probability. Draws with at least one parameter on a bound: 55 of 1000 (5.50%). Independent of the configured bounds: among the 57000 sampled lengths the smallest (4.540e-05) occurs 58 time(s) and the largest (0.0730) occurs 1 time(s); a continuous distribution gives each once. The target has a density, so it gives these draws no mass: the Radon-Nikodym derivative of the target with respect to the sampling distribution is 0 on an atom. Their importance weight is therefore 0 in (i). The value the training code forms for such a draw (score divided by the unclipped density at the bound) is not a weight with respect to any measure; it is shown in (i) only for comparison with the training log. The fraction of draws on a bound, 0.0550 [0.042, 0.071], estimates the mass that q_theta puts where the target has none: the total variation between q_theta and any distribution with a density is at least this mass. log p_T of these draws is not a density either: they are left out of the regression in (iii). A draw is counted as on a bound when its log length is within 1e-6 of the lower or 1e-9 of the upper bound, or its logit(theta) within 1e-3 of a bound (single-precision resolution of the stored value).

(i) Importance weights. w_i = R(x_i) P_B(tau_i | x_i) / P_F(tau_i) for the 1000 sampled trajectories. Exact balance makes w constant. Kish effective sample size 85.6 of 1000 (8.6%); standard deviation of log w 1.453 nats; largest normalised weight 0.074. The implied MLL estimate is -7043.113 (delta-method SE 0.103, unreliable when the ESS is small), against -7043.143 +- 0.104 in the training log (the training log forms a weight for every draw, also for a draw on a clipping bound; without such draws the two estimators are the same, and the comparison is a functional check that checkpoint, dataset and code agree). 55 of the 1000 draws are on a clipping bound and have weight 0 here. With the weight the training code forms for them: ESS 97.9, MLL estimate -7042.997 (delta-method SE 0.096); that estimator is biased upwards by the mass of the atoms times the ratio of the score to the unclipped density at the bound. The estimate with weight 0 targets the mass of the posterior inside the clipping box; the posterior mass outside the box cannot be reached by the sampler and is not estimated.
The ESS is a trajectory-level quantity. The weight of a terminal object is the conditional mean of the trajectory weights that end in it, so the trajectory-level ESS cannot exceed the ESS at the level of terminal objects; it is small whenever a few trajectories dominate. With ESS / n = 0.086 < 0.1 the reweighted masses below are not interpretable and are not used.

| quantity | frequency in the sample | importance-reweighted mass +- SE |
|---|---|---|
| R = 2 | 1.0000 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| root_position_type = pendant | 0.4360 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| root_position_type = cycle | 0.3220 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| root_position_type = internal | 0.2420 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = b3935f59f776 | 0.2400 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = fb140a2ffdf8 | 0.1860 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = 7aac35902193 | 0.0680 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = 510a0eb5cc51 | 0.0560 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L4_displayed_set_top = 8ef5af28f5d4 | 0.0430 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = 7b905af9c1ec | 0.1700 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = 30e62211c363 | 0.1300 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = 39007d9bd62a | 0.1120 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = e1ef28a9b8bf | 0.0840 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |
| L5_major_tree_top = ccd8087d4a6a | 0.0360 | not interpretable (ESS / n < 0.1); value in importance_reweighting.csv |

(ii) Root-position uniformity. Under the target, given the semi-directed network and its parameters, the root is uniformly distributed over the in-class root positions (equal density and unit Jacobian under re-rooting; tools/verify_root_equivalence.py). Observed versus expected number of draws by root-position type: pendant edge 436 vs 444.4 (sd 15.7, z = -0.54); internal edge 242 vs 226.0 (sd 13.2, z = 1.21); cycle edge 322 vs 329.6 (sd 14.9, z = -0.51). A necessary condition for q_theta = p that needs no reference; it tests one coordinate only. Here max |z| = 1.2: no departure from root-position uniformity is detected at this sample size (which does not prove uniformity).

(iii) Terminal density: the optional stage tools/posterior_density_diagnostics.py was not run for this output.

(iv) Independent reference posterior: NONE for this dataset. No exact or independently sampled posterior over level-1 networks exists for it, so the accuracy of q_theta at the level of topology masses is NOT established by this output. Diagnostics (i)-(iii) are necessary conditions computed from the model's own quantities; they can reveal a mismatch and cannot prove a match. On small simulated problems the comparison with a numerical reference is available (--reference).

## 9. Results: pairwise comparison of the 100-draw subset (4950 unordered pairs)
The pairs share draws: the uncertainty of a mean over pairs is the jackknife standard error over the 100 draws, not sd / sqrt(4950).

| quantity | what it is | mean over pairs +- jackknife SE | median | min | max | pairs at 0 | mean self value |
|---|---|---|---|---|---|---|---|
| mu_distance_norm | metric | 0.2631 +- 0.0139 | 0.2667 | 0.0000 | 0.7000 | 0.2% | 0.0000 |
| dt_rf_hausdorff_norm | pseudo-metric | 0.0839 +- 0.0070 | 0.0833 | 0.0000 | 0.3333 | 10.5% | 0.0000 |
| dt_jaccard_distance | pseudo-metric | 0.8536 +- 0.0209 | 1.0000 | 0.0000 | 1.0000 | 10.5% | 0.0000 |
| dt_tv | pseudo-metric | 0.8574 +- 0.0189 | 1.0000 | 0.0051 | 1.0000 | 0.0% | 0.0000 |
| dt_js_distance | pseudo-metric | 0.8741 +- 0.0180 | 1.0000 | 0.0054 | 1.0000 | 0.0% | 0.0000 |
| split_l1_norm | pseudo-metric | 0.0948 +- 0.0043 | 0.0927 | 0.0015 | 0.2804 | 0.0% | 0.0000 |
| major_rf_norm | pseudo-metric | 0.2024 +- 0.0081 | 0.2083 | 0.0000 | 0.4583 | 6.9% | 0.0000 |
| dt_rf_min_norm | dissimilarity | 0.0488 +- 0.0037 | 0.0417 | 0.0000 | 0.2083 | 22.9% | 0.0000 |
| sig_jaccard_distance | pseudo-metric | 0.5059 +- 0.0402 | 0.6667 | 0.0000 | 1.0000 | 30.0% | 0.0000 |
| hybrid_clade_jaccard_distance | pseudo-metric | 0.1114 +- 0.0328 | 0.0000 | 0.0000 | 0.6667 | 83.3% | 0.0000 |
| dt_rf_mean_norm | descriptive statistic | 0.2322 +- 0.0030 | 0.2292 | 0.1458 | 0.3542 | 0.0% | 0.1850 |
| dt_rf_expected_norm | descriptive statistic | 0.2295 +- 0.0032 | 0.2282 | 0.1270 | 0.3590 | 0.0% | 0.1767 |

Normalised quantities are divided by 2(N - 3) = 48 (unrooted RF maximum) or, for the mu-distance, by the number of internal nodes of the two networks. "mean self value" is the quantity of a network compared with itself, averaged over the subset: it is 0 for every metric, pseudo-metric and dissimilarity; for the two descriptive statistics it is positive as soon as a network displays more than one tree topology, which is why they are not distances. A pseudo-metric is 0 for different networks that share the derived object named in the definitions.

Pair classes (mutually exclusive, decided by the definitions; fractions +- jackknife SE over draws):

| class | pairs | fraction +- SE | of which same major tree | same R | same signature set |
|---|---|---|---|---|---|
| P1 same rooted topology | 10 | 0.0020 +- 0.0011 | 5 | 10 | 10 |
| P2 same class different rooting | 510 | 0.1030 +- 0.0193 | 200 | 510 | 510 |
| P3 different class same displayed set | 0 | 0.0000 +- 0.0000 | 0 | 0 | 0 |
| P4 displayed sets differ share a topology | 614 | 0.1240 +- 0.0194 | 138 | 614 | 0 |
| P5 no displayed topology in common | 3816 | 0.7709 +- 0.0279 | 0 | 3816 | 963 |

P1: only continuous parameters differ. P2: the root position differs. P3: the network topologies differ and display the same set of tree topologies. P4: the displayed sets differ and share at least one tree topology. P5: no displayed tree topology in common. The classes do not say why two networks differ beyond these definitions.

## 10. Results: continuous parameters (kept separate from topology)
Total branch length over the 1000 draws: mean 0.4334, sd 0.0200. Inheritance probability of the heavier hybrid edge over 2000 reticulations: mean 0.595, sd 0.080. 55 root-equivalence classes occur at least twice (955 draws). Within them every edge and every inheritance probability was matched across draws through root-invariant taxon-based keys (all 55 edge-key sets identical within a class; a mismatch stops the script): median over classes of the mean coefficient of variation of matched branch lengths 0.334; mean sd of matched inheritance probabilities 0.074; mean coefficient of variation of the total length 0.036 (duplicate_topology_parameter_variability.csv).

## 11. Scores
log score = log R(x) = log-likelihood + log branch-length prior + log topology prior (unnormalised), a pointwise density value of one draw with its continuous parameters. Over the 1000 draws: mean -6694.44, sd 5.85, range -6713.52 to -6681.30. A topology with a high pointwise score need not have a large posterior mass: mass is the integral of R over the continuous parameters, times the number of labellings. The frequency of a topology in the sample estimates its mass under q_theta; the reweighted mass (section 8) estimates its mass under the target when the ESS allows. score_by_topology.csv lists, for the most frequent classes, the frequency with its interval, the pointwise score summary and the reweighted mass, each in its own column. Spearman correlation between the pointwise log score and the split-frequency distance to the most frequent class in the subset: rho = -0.184 (n = 100 draws; descriptive).

## 12. Answers supported by this output
(a) Diversity of q_theta. Number of distinct values among 1000 draws: rooted topologies 537, root-equivalence classes 100, sets of reticulation signatures 50, displayed-tree sets 100, major trees 106. The most frequent class has frequency 0.240 [0.215, 0.267]; 45 of the 100 classes were seen once and the estimated unseen mass at the class level is 0.045 +- 0.009. Of the plug-in entropy of the rooted-topology distribution (6.002 nats), 51.1% lies between displayed-tree sets, 0.0% between classes that display the same trees and 48.9% in the root position. In the 100-draw subset, 77.1% of the pairs (+- 2.8) have no displayed tree topology in common, 12.4% (+- 1.9) share one while their displayed sets differ, 0.0% (+- 0.0) are different classes displaying the same set of trees, 10.3% (+- 1.9) differ in root position only and 0.2% (+- 0.1) in continuous parameters only. The mean normalised Hausdorff RF between displayed-tree sets is 0.0839 +- 0.0070 and the mean normalised mu-distance between rooted networks is 0.2631 +- 0.0139.
These numbers describe one checkpoint and one sample of size 1000; intervals cover the Monte-Carlo variability of the draws only (not the variability between training runs).
(b) Accuracy of q_theta. Importance-weight ESS 85.6 of 1000 (8.6%): the trajectory weights are far from constant. Root-position uniformity z-scores pendant -0.54, internal 1.21, cycle -0.51: no mismatch detected in the root position at this sample size. No independent reference posterior exists for this dataset. The diagnostics above are necessary conditions computed from the model itself: they can show that q_theta differs from the target and cannot show that it matches it. Whether q_theta approximates the posterior at the level of topology masses cannot be answered with the data in this output.

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
python tools/sample_posterior_newick.py runs_ds1_complete/20261113_220512_ds1_baseline_resume_s1 dataset/benchmark_datasets/DS1.pickle --epoch 299 --expect-objective TB --n 1000 --seed 42 --subset 100 --subset-seed 42 --chunk 64 --device cpu --threads 4 --out-dir /srv/scratch1/mmousabeigi/posterior_v2_ds1_full --dataset-name DS1
python tools/analyze_posterior_topology.py --samples-dir /srv/scratch1/mmousabeigi/posterior_v2_ds1_full --sequences dataset/benchmark_datasets/DS1.pickle --seed 42
```
Tables: topology_summary.csv, topology_frequencies.csv, topology_frequencies_by_level.csv, displayed_tree_distribution.csv, entropy_decomposition.csv, accumulation_curves.csv, reticulation_count_vs_prior.csv, reticulation_signature_frequencies.csv, hybrid_clade_frequencies.csv, importance_reweighting.csv, root_position_uniformity.csv, pairwise_topology_distances.csv (4950 rows), pairwise_summary.csv, pair_classes.csv, pairwise_matrices.npz, duplicate_topology_parameter_variability.csv, score_by_topology.csv, validation_report.json. Figures in figures/.
