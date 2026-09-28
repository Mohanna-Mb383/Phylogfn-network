"""
Consistency check: with R_MAX = 0 PhyloGFN-Net must be a PhyloGFN tree sampler.
Both samplers are trained for the same budget on the same alignment and their
importance-weighted marginal log-likelihood estimates are compared.

    python tools/compare_tree_mode.py dataset/simulated/sim6_r1.pickle [epochs] [steps_per_epoch]
"""
import sys, os, random, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.configs.defaults import get_cfg_defaults
from src.utils.utils import correct_cfg_data, load_sequences
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_phylo import RolloutWorker
from src.gfn.training_data_loader import TrainingDataLoader
from src.gfn.gfn_evaluator import GFNEvaluator
from src.gfn.rollout_worker_net import RolloutWorkerNet
from src.gfn.training_data_loader_net import TrainingDataLoaderNet
from src.gfn.gfn_evaluator_net import GFNEvaluatorNet
from train_net import generate_exploration_spec, current_temperature

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src/configs/network_cfgs/cfg_net_small_cpu.yaml')


def run(mode, seqs, epochs, steps, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cfg = get_cfg_defaults(); cfg.merge_from_file(CFG)
    cfg.GFN.TRAINING_DATA_LOADER.EPOCHS_NUM = epochs
    cfg.GFN.TRAINING_DATA_LOADER.STEPS_PER_EPOCH = steps
    if mode == 'tree':
        cfg.ENV.ENVIRONMENT_TYPE = 'ONE_STEP_BINARY_TREE'
    else:
        cfg.ENV.NETWORK.R_MAX = 0
    cfg = correct_cfg_data(seqs, 1, cfg)
    env = build_env(cfg, seqs)
    gen = build_gfn(cfg, env, 'cpu', ddp=False)
    if mode == 'tree':
        rw = RolloutWorker(env)
        loader = TrainingDataLoader(cfg, env, rw, '/tmp/_best_trees_cmp.pt')
        evaluator = GFNEvaluator(cfg.GFN.MODEL.EVALUATION, rw, gen, verbose=False)
    else:
        rw = RolloutWorkerNet(env)
        loader = TrainingDataLoaderNet(cfg, env, rw, '/tmp/_best_nets_cmp.pt')
        evaluator = GFNEvaluatorNet(cfg.GFN.MODEL.EVALUATION, rw, gen, verbose=False)
    t_cfg = cfg.GFN.TRAINING_DATA_LOADER.TEMPERATURE_ANNEALING
    t0 = time.time()
    for epoch in range(epochs):
        temperature = current_temperature(t_cfg, epoch)
        if temperature != env.reward_fn.scale:
            z = gen.compute_log_Z().item() * (env.reward_fn.scale / temperature)
            torch.nn.init.constant_(gen._Z, z / 256)
            env.reward_fn.scale = temperature
        specs = generate_exploration_spec(cfg.GFN.TRAINING_DATA_LOADER.EXPLORATION, epoch)
        for (batch, trajs), rs in loader.build_epoch_iterator(gen, specs):
            gen.accumulate_loss(batch, 1)
            gen.update_model()
    mlls = []
    for _ in range(5):
        r = evaluator.evaluate_marginal_likelihood(1024)
        mlls.append(r[0] if isinstance(r, tuple) else r)
    return float(np.mean(mlls)), float(np.std(mlls)), time.time() - t0


if __name__ == '__main__':
    seqs = load_sequences(sys.argv[1])
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    for mode in ('tree', 'net'):
        m, s, dt = run(mode, seqs, epochs, steps)
        print(f'{mode:5s}: MLL {m:.2f} +- {s:.2f}   ({dt:.0f}s)')
