#!/usr/bin/env bash
# Methods of da Silva et al. (ICLR 2025) on PhyloGFN-Net: gate experiments, then DS1-DS8.
#   bash run_paper_methods.sh cfgs            write the configurations (src/configs/network_cfgs/paper/)
#   bash run_paper_methods.sh test            unit tests (tests/test_db_wdb.py, tests/test_network_env.py)
#   bash run_paper_methods.sh regress         byte-level regression of the TB path against the unpatched 54510ac (git worktree)
#   bash run_paper_methods.sh n4              N = 4 reference problem: TB / DB / WDB x 5 seeds, numerical reference (cached),
#                                             checkpoint-wise TV + FCS (endpoints.csv), paired AUC comparison
#   bash run_paper_methods.sh sim6            N = 6 gate: TB / DB / WDB(logcount, count, depth) / WDB+look-ahead x 3 seeds,
#                                             checkpoint-wise FCS, FCS diagnostics + residual profile, paired AUC comparison
#   bash run_paper_methods.sh membench <d> <variant> [device]   memory benchmark + 10-update dry run of one DS configuration
#                                             (mandatory before DS7 / DS8), e.g.  bash run_paper_methods.sh membench 7 wdb_logcount cuda
#   bash run_paper_methods.sh ds <d> <variant> [device] [seed]  one DS run (seeded; default seed 1), e.g.  bash run_paper_methods.sh ds 1 wdb_logcount cuda
#   bash run_paper_methods.sh eval <run_dir> <sequences> [device] [every]   checkpoint-wise FCS (every k-th checkpoint), fcs_eval
#                                             (+ residual profile for DB/WDB runs) on the final checkpoint
# Every run directory gets config.yaml (with SEED), checkpoints/, history.pkl (with train_time_s / elapsed_s per
# epoch), final_mlls.p; evaluation outputs (endpoints.csv, fcs.txt, reference_tv.txt, residual_profile.txt) go next to it.
set -e
cd "$(dirname "$0")"
N4_CFG=src/configs/network_cfgs/paper/n4_tb.yaml
N4_DATA=dataset/simulated/sim4_r1.pickle
N4_REF=runs/n4_reference.pkl
SIM6_DATA=dataset/simulated/sim6_r1_nontrivial.pickle
case "$1" in
  cfgs)
    python tools/make_paper_method_cfgs.py ;;
  test)
    python tests/test_db_wdb.py && python tests/test_network_env.py ;;
  regress)
    python tools/regression_vs_baseline.py --seed=0 && python tools/regression_vs_baseline.py --seed=1 --n=5 --m=12 ;;
  n4)
    mkdir -p runs
    [ -f $N4_DATA ] || python tools/simulate_network_data.py $N4_DATA --leaves=4 --sites=100 --rets=1 --seed=5
    # numerical reference, computed once and cached; its two error floors are printed in runs/n4_reference.txt
    [ -f $N4_REF ] || python tools/reference_posterior_small_n.py $N4_CFG $N4_DATA --draws=4000 --samples=20000 --save_reference=$N4_REF --out=runs/n4_reference.txt
    for v in tb db wdb_logcount wdb_count; do for s in 1 2 3 4 5; do
      python train_net.py src/configs/network_cfgs/paper/n4_$v.yaml $N4_DATA runs/n4_${v}_s$s --device=cpu --quiet --seed=$s > logs_n4_${v}_s$s.txt 2>&1
      run=$(ls -d runs/*_n4_${v}_s$s | tail -1)
      python tools/auc_endpoints.py $run $N4_DATA --reference=$N4_REF --ref_cfg=$N4_CFG --states=128 --paths=50 --samples=8192 --device=cpu --out=$run/endpoints.csv
      python tools/reference_posterior_small_n.py $N4_CFG $N4_DATA --draws=4000 --samples=20000 --load_reference=$N4_REF --run_dir=$run --out=$run/reference_tv.txt
      python tools/fcs_eval.py $run $N4_DATA --states=256 --paths=100 --stress=1 --kstudy=25,50,100,200 --device=cpu --out=$run/fcs.txt
    done; done
    for metric in tv_lab tv_graph fcs; do python tools/auc_endpoints.py --compare 'runs/*_n4_*/endpoints.csv' --metric=$metric --control=db; done | tee runs/n4_comparison.txt ;;
  sim6)
    mkdir -p runs
    [ -f $SIM6_DATA ] || python tools/simulate_network_data.py $SIM6_DATA --leaves=6 --sites=300 --rets=1 --seed=3
    for v in tb db wdb_logcount wdb_count wdb_depth wdb_logcount_la; do for s in 1 2 3; do
      python train_net.py src/configs/network_cfgs/paper/sim6_$v.yaml $SIM6_DATA runs/sim6_${v}_s$s --device=cpu --quiet --seed=$s > logs_sim6_${v}_s$s.txt 2>&1
      run=$(ls -d runs/*_sim6_${v}_s$s | tail -1)
      python tools/auc_endpoints.py $run $SIM6_DATA --states=128 --paths=50 --device=cpu --out=$run/endpoints.csv
      python tools/fcs_eval.py $run $SIM6_DATA --states=256 --paths=100 --stress=1 --kstudy=25,50,100,200 --device=cpu --out=$run/fcs.txt
      [ "$v" = "tb" ] || python tools/db_residual_profile.py $run $SIM6_DATA --trajectories=512 --out=$run/residual_profile.txt
    done; done
    python tools/auc_endpoints.py --compare 'runs/*_sim6_*/endpoints.csv' --metric=fcs --control=db | tee runs/sim6_comparison.txt ;;
  membench)
    d=$2; v=$3; dev=${4:-cuda}
    cfg=src/configs/network_cfgs/paper/DS${d}_$v.yaml
    splits=$(python -c "import yaml,sys; print(yaml.safe_load(open('$cfg'))['GFN']['TRAINING_DATA_LOADER']['MINI_BATCH_SPLITS'])")
    python tools/probe_rollout_memory.py $cfg dataset/benchmark_datasets/DS$d.pickle --device=$dev --batch=128 --splits=$splits --updates=10 2>&1 | tee membench_ds${d}_$v.txt ;;
  ds)
    d=$2; v=$3; dev=${4:-cuda}; seed=${5:-1}
    PYTHONUNBUFFERED=1 nohup python -u train_net.py src/configs/network_cfgs/paper/DS${d}_$v.yaml dataset/benchmark_datasets/DS$d.pickle runs/ds${d}_network_${v}_300ep --device=$dev --quiet --seed=$seed > logs_ds${d}_network_${v}_300ep.txt 2>&1 &
    echo "started DS$d $v seed $seed (pid $!)" ;;
  eval)
    run=$2; seqs=$3; dev=${4:-cuda}; every=${5:-10}
    python tools/auc_endpoints.py $run $seqs --states=128 --paths=50 --every=$every --device=$dev --out=$run/endpoints.csv
    python tools/fcs_eval.py $run $seqs --states=256 --paths=100 --stress=1 --kstudy=25,50,100,200,400 --device=$dev --out=$run/fcs.txt
    grep -q "NET_LOSS: TB" $run/config.yaml || python tools/db_residual_profile.py $run $seqs --trajectories=512 --device=$dev --out=$run/residual_profile.txt ;;
  *)
    sed -n 2,16p "$0" ;;
esac
