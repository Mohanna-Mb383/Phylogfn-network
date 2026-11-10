#!/usr/bin/env bash

set -u
set -o pipefail

cd "$(dirname "$0")"

SIM6_DATA=dataset/simulated/sim6_r1_nontrivial.pickle
FAIL_LOG=runs/sim6_failures.txt

mkdir -p runs

echo "=== sim6 safe resume started: $(date) ===" | tee -a "$FAIL_LOG"

for v in tb db wdb_logcount wdb_count wdb_depth wdb_logcount_la; do
  for s in 1 2 3; do

    completed=""

    for r in runs/*_sim6_${v}_s${s}; do
      [ -d "$r" ] || continue

      if [ -f "$r/fcs.txt" ] && [ -f "$r/endpoints.csv" ]; then
        completed="$r"
      fi
    done

    if [ -n "$completed" ]; then
      echo "SKIP COMPLETE: $v seed $s -> $completed"
      continue
    fi

    echo
    echo "============================================================"
    echo "RUN: $v seed $s"
    echo "START: $(date)"
    echo "============================================================"

    if ! python train_net.py \
      src/configs/network_cfgs/paper/sim6_${v}.yaml \
      "$SIM6_DATA" \
      runs/sim6_${v}_s${s} \
      --device=cuda \
      --quiet \
      --seed=${s} \
      > logs_sim6_${v}_s${s}.txt 2>&1
    then
      echo "$(date) TRAIN_FAILED variant=$v seed=$s" | tee -a "$FAIL_LOG"
      continue
    fi

    run=$(ls -d runs/*_sim6_${v}_s${s} 2>/dev/null | tail -1)

    if [ -z "$run" ]; then
      echo "$(date) RUN_DIRECTORY_MISSING variant=$v seed=$s" | tee -a "$FAIL_LOG"
      continue
    fi

    echo "TRAIN COMPLETE: $run"

    if ! python tools/auc_endpoints.py \
      "$run" \
      "$SIM6_DATA" \
      --states=128 \
      --paths=50 \
      --device=cuda \
      --out="$run/endpoints.csv"
    then
      echo "$(date) AUC_FAILED variant=$v seed=$s run=$run" | tee -a "$FAIL_LOG"
      continue
    fi

    if ! python tools/fcs_eval.py \
      "$run" \
      "$SIM6_DATA" \
      --states=256 \
      --paths=100 \
      --stress=1 \
      --kstudy=25,50,100,200 \
      --device=cuda \
      --out="$run/fcs.txt"
    then
      echo "$(date) FCS_FAILED variant=$v seed=$s run=$run" | tee -a "$FAIL_LOG"
      continue
    fi

    if [ "$v" != "tb" ]; then
      if ! python tools/db_residual_profile.py \
        "$run" \
        "$SIM6_DATA" \
        --trajectories=512 \
        --out="$run/residual_profile.txt"
      then
        echo "$(date) RESIDUAL_FAILED variant=$v seed=$s run=$run" | tee -a "$FAIL_LOG"
      fi
    fi

    echo "COMPLETE: $v seed $s"
    echo "END: $(date)"

  done
done

echo
echo "=== FINAL SIM6 COMPARISON ==="

python tools/auc_endpoints.py \
  --compare 'runs/*_sim6_*/endpoints.csv' \
  --metric=fcs \
  --control=db \
  | tee runs/sim6_comparison.txt

echo "=== sim6 safe resume finished: $(date) ===" | tee -a "$FAIL_LOG"
