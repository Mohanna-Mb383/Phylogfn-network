# Short supervisor report — DS1 + kNN

The repaired/vectorized DS1 KNN experiment (`k=5`, exact cosine neighbours, 4-mer embeddings) completed all 300 epochs with 200 optimizer updates per epoch. The training monitor recorded **60,000 accepted updates and 0 rejected updates**; the previous silent optimizer freeze after about epoch 200 did not recur.

The final three-repeat MLL summary was **−7153.974 ± 4.074**, with **1,024 trajectories per repeat**. The last training epoch (299) reported MLL **−7154.572**, Pearson R **0.783**, and reticulation counts **[831, 183, 10]** for R=0,1,2 over 1,024 samples.

The repair addressed Gaussian scale numerical instability, added accepted/rejected-update monitoring, and used batched/vectorized feature calculations. The small CPU equivalence tests matched original and optimized computations exactly for the sampled action cases.

**Interpretation:** This establishes training stability for the full recorded DS1 run. It **does not establish** that KNN improves MLL or wall-clock speed. Matched evaluation against no-KNN, with ESS and fixed evaluation conditions, and matched idle-GPU timing are the next requirements.
