# Reproduction and provenance

## Machine / historical source paths (user-reported)

- Server: `vm03` (NVIDIA L40S), Conda environment `phylogfn`.
- Vectorized source: `~/phylogfn_knn_vectorized_ds1_20261123_013714`.
- Historical project: `~/phylogfn_proposal_20260929/phylogfn`.
- Config: `~/phylogfn_proposal_20260929/phylogfn/runs/20260929_192648_ds1_knn_only_300ep_final/config.yaml`.
- DS1 input: `~/phylogfn_proposal_20260929/phylogfn/dataset/benchmark_datasets/DS1.pickle`.
- New training parent: `~/knn_fix_runs/ds1_knn_free_gpu_20261123_093107/`.
- New run output: `.../ds1_knn_300`, containing a timestamped subdirectory with `history.pkl` and checkpoints.

These paths describe the researcher's server environment; they are **not** public GitHub URLs.

## Command used to start the completed run

```bash
export VEC="$HOME/phylogfn_knn_vectorized_ds1_20261123_013714"
export HIST="$HOME/phylogfn_proposal_20260929/phylogfn"
export RUN="$HIST/runs/20260929_192648_ds1_knn_only_300ep_final"
export CFG="$RUN/config.yaml"
export DS1="$HIST/dataset/benchmark_datasets/DS1.pickle"
export OUTPUT="$HOME/knn_fix_runs/ds1_knn_free_gpu_20261123_093107"
export PHYLOGFN_FAST_FEATURES=on
export PHYLOGFN_MAX_CONSEC_REJECT=5
export PHYLOGFN_MAX_EPOCH_REJECT_FRAC=0.10
cd "$VEC"
python -u train_net.py "$CFG" "$DS1" "$OUTPUT/ds1_knn_300" \
  --device=cuda --seed=1 --quiet 2>&1 | tee "$OUTPUT/train.log"
```

The exact prefix output path from the supplied terminal was constructed using a time-stamped shell variable. The above export uses that known resulting directory. Re-running the command would **start new training**, not resume from the completed checkpoints; do not run it just to inspect results.

## Export the actual result files from `vm03`

After copying this package inside a clone of `Mohanna-Mb383/Phylogfn-network` on `vm03`:

```bash
python results/network/DS1/knn_k5_vectorized_20261123/tools/collect_vm03.py \
  --run-root "$HOME/knn_fix_runs/ds1_knn_free_gpu_20261123_093107" \
  --source-root "$HOME/phylogfn_knn_vectorized_ds1_20261123_013714" \
  --destination results/network/DS1/knn_k5_vectorized_20261123
```

The export is **read-only with respect to training files**. It saves log/config, CSV summaries of history, and JSON result samples, plus hashes of relevant source files. Raw pickle copying and full checkpoint hashing are optional flags and are off by default. It never copies massive checkpoints or raw DNA by default.

Review `git status`, then commit the new folder to your own repository. The optional source snapshot can be enabled with `--include-source-snapshot`; see the tool help.

## Crucial publication note

- Do not upload giant checkpoint `.pt` files without checking size, privacy, licensing and Git LFS needs.
- Do not claim code is reproducible from this results folder alone if the exact vectorized source version is absent from the repository.
- Do not edit, overwrite or remove the historical runs when exporting.
