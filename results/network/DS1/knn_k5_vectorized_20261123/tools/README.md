# Tools

- `collect_vm03.py`: Collect exact config, training log, summarized history, final MLL samples, checkpoint metadata and source-code hashes from the completed run **without modifying the original training files**.
- `--include-source-snapshot`: Add selected implementation files (not a standalone complete repository) as an optional source audit snapshot.

Run from the *root of a safe Git clone*, after copying this folder into `results/network/DS1/knn_k5_vectorized_20261123/`. See `REPRODUCIBILITY.md` for exact arguments.
