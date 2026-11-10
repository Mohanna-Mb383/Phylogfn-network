"""
probe_rollout_memory.py -- measure the memory of ONE training step of train_net.py before
launching a 300-epoch run (same config, same data loader, same generator; nothing is trained).

    python tools/probe_rollout_memory.py <cfg_path> <sequences_path> [--device=cpu|cuda] [--batch=128]
                                        [--splits=1] [--eval=256] [--eps=0.5] [--updates=0]

The probe follows the configuration's objective: with GFN.NET_LOSS = DB / WDB the rollout worker records the
per-transition quantities (record_steps) and the flow head is evaluated, with LOOKAHEAD.ENABLED the child
embeddings are formed, exactly as in train_net.py.  --updates=<k> (cuda) additionally performs k complete optimiser
updates with the real data loader (splits x rollouts, gradient accumulation, clip, Adam step) and reports the peak
memory and the seconds per update: the dry run to make before a DS7 / DS8 training.

On cuda it reports torch.cuda.max_memory_allocated() after the rollout (activations kept for
backward), after loss.backward(), and for one evaluation chunk, against the device's total memory.
On cpu it counts the bytes that autograd saves for backward during the rollout (unique storages),
which is the quantity that grows with the trajectory length and fills the GPU; the result is
scaled to the requested --batch so the probe can run with a small batch on a small machine.

--batch   total trajectories per rollout = GFN_BATCH_SIZE + BEST_STATE_BATCH_SIZE (128 in cfg_net_DS1_300ep.yaml)
--splits  MINI_BATCH_SPLITS: each rollout then has batch/splits trajectories and the gradient is
          accumulated over `splits` rollouts before one optimiser step (same 128-trajectory update).
"""
import os, sys, time, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from docopt import docopt

from src.configs.defaults import get_cfg_defaults
from src.utils.utils import correct_cfg_data, load_sequences
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_net import RolloutWorkerNet
from src.gfn.training_data_loader_net import TrainingDataLoaderNet
from src.gfn.gfn_evaluator_net import GFNEvaluatorNet

USAGE = """probe_rollout_memory

Usage:
    probe_rollout_memory.py <cfg_path> <sequences_path> [--device=<dev>] [--batch=<b>] [--splits=<s>] [--eval=<e>] [--eps=<p>] [--probe_batch=<pb>] [--updates=<k>]

Options:
    --device=<dev>      cpu or cuda [default: cpu]
    --batch=<b>         trajectories per optimiser update to report for [default: 128]
    --splits=<s>        MINI_BATCH_SPLITS to report for [default: 1]
    --eval=<e>          evaluation chunk size (GFNEvaluatorNet uses 256) [default: 256]
    --eps=<p>           random-action probability used for the probe rollout [default: 0.5]
    --probe_batch=<pb>  trajectories actually rolled out on cpu (scaled to --batch) [default: 8]
    --updates=<k>       cuda only: number of complete optimiser updates of the dry run after the probe [default: 0]
"""


def gib(x):
    return x / 1024 ** 3


def main():
    args = docopt(USAGE)
    dev = args['--device']
    B = int(args['--batch'])
    S = int(args['--splits'])
    E = int(args['--eval'])
    eps = float(args['--eps'])
    seqs = load_sequences(args['<sequences_path>'])
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args['<cfg_path>'])
    cfg = correct_cfg_data(seqs, 1, cfg)
    N, m = len(seqs), len(seqs[0])
    r_max = cfg.ENV.NETWORK.R_MAX
    l_max = N + r_max
    n_pairs = l_max * (l_max - 1) // 2
    steps_max = N - 1 + 2 * r_max
    print(f'dataset: N={N} taxa, m={m} sites, R_MAX={r_max} -> l_max={l_max}, pairs={n_pairs}, '
          f'longest trajectory={steps_max} steps, policy input per lineage 2mc+4={2 * m * 4 + 4}')
    if dev == 'cuda' and not torch.cuda.is_available():
        print('cuda requested but not available'); sys.exit(1)

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    env = build_env(cfg, seqs)
    # numpy >= 2 can return a complex (zero imaginary part) eigendecomposition of the JC matrix;
    # the probe keeps the real parts so that get_transition_matrices works as on numpy 1.x
    for name in ('D', 'U', 'U_inv'):
        t = getattr(env.evolution_model, name)
        if torch.is_complex(t):
            t.data = t.data.real.clone()
    env.to(dev)
    gen = build_gfn(cfg, env, dev, ddp=False)
    rw = RolloutWorkerNet(env, record_steps=(cfg.GFN.NET_LOSS != 'TB'), wdb_gamma=cfg.GFN.WDB_GAMMA)
    print(f'objective {cfg.GFN.NET_LOSS}' + (f' (gamma {cfg.GFN.WDB_GAMMA})' if cfg.GFN.NET_LOSS == 'WDB' else '')
          + f', look-ahead {"on" if cfg.GFN.MODEL.LOOKAHEAD.ENABLED else "off"}, MINI_BATCH_SPLITS in the file {cfg.GFN.TRAINING_DATA_LOADER.MINI_BATCH_SPLITS} (probe uses --splits {S})')
    feats_bytes = B * l_max * 2 * m * 4 * 8        # float64 features of one rollout
    x_bytes = B * l_max * (2 * m * 4 + 4) * 4       # float32 policy input of one step
    print(f'static per rollout at batch {B}: features float64 {gib(feats_bytes):.2f} GiB; policy input per step float32 {gib(x_bytes):.2f} GiB')

    if dev == 'cuda':
        rollout_batch = B // S
        loader_cfg = cfg.GFN.TRAINING_DATA_LOADER
        loader_cfg.MINI_BATCH_SPLITS = S
        import tempfile
        loader = TrainingDataLoaderNet(cfg, env, rw, os.path.join(tempfile.mkdtemp(prefix='probe_'), 'best_networks.pt'))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.time()
        data, trajs = loader.generate_batch(gen, {'random_action_prob': eps})
        torch.cuda.synchronize()
        peak_roll = torch.cuda.max_memory_allocated()
        n_steps = max(len(t.actions) for t in trajs)
        gen.accumulate_loss(data, S)
        torch.cuda.synchronize()
        peak_bwd = torch.cuda.max_memory_allocated()
        gen.opt.zero_grad()
        total = torch.cuda.get_device_properties(0).total_memory
        print(f'rollout of {rollout_batch} trajectories ({n_steps} steps, {time.time() - t0:.0f}s): '
              f'peak allocated {gib(peak_roll):.2f} GiB (baseline {gib(base):.2f}); after backward {gib(peak_bwd):.2f} GiB; '
              f'device total {gib(total):.2f} GiB')
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            rw.rollout(gen, E, generate_full_trajectories=False)
        torch.cuda.synchronize()
        print(f'evaluation chunk of {E} trajectories (no grad): peak allocated {gib(torch.cuda.max_memory_allocated()):.2f} GiB')
        if peak_bwd > 0.95 * total:
            print('VERDICT: does not fit (peak within 5% of device memory or above)')
        else:
            print(f'VERDICT: fits with {gib(total - peak_bwd):.2f} GiB to spare (the PyTorch cache adds up to a few GiB on top)')
        k = int(args['--updates'])
        if k > 0:
            # dry run: k complete updates exactly as train_net.py does them (rollout -> accumulate_loss over S splits -> step)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time(); losses = []
            for u in range(k):
                for _ in range(S):
                    data, trajs = loader.generate_batch(gen, {'random_action_prob': eps})
                    gen.accumulate_loss(data, S)
                info = gen.update_model(); losses.append(info['loss'])
            torch.cuda.synchronize()
            print(f'dry run: {k} updates of {B} trajectories ({S} rollouts of {rollout_batch} each): peak allocated '
                  f'{gib(torch.cuda.max_memory_allocated()):.2f} GiB, reserved {gib(torch.cuda.max_memory_reserved()):.2f} GiB, '
                  f'{(time.time() - t0) / k:.1f} s per update, losses {["%.3g" % l for l in losses]}')
        return

    # ---- cpu: count autograd-saved bytes during one rollout with a small batch, scale to B
    pb = int(args['--probe_batch'])
    saved = {}
    per_step = []
    current = {}

    def pack(t):
        try:
            st = t.untyped_storage()
            key = (st.data_ptr(), st.nbytes())
            if key not in saved:
                saved[key] = st.nbytes()
                current[key] = st.nbytes()
        except Exception:
            pass
        return t

    def unpack(t):
        return t

    orig_forward = gen.forward
    step_counter = [0]

    def counting_forward(input_dict):
        current.clear()
        out = orig_forward(input_dict)
        per_step.append(sum(current.values()))
        step_counter[0] += 1
        return out
    gen.forward = counting_forward
    t0 = time.time()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        data, trajs = rw.rollout(gen, pb, random_spec={'random_action_prob': eps}, generate_full_trajectories=False)
    n_steps = len(per_step)
    total_saved = sum(saved.values())
    param_bytes = sum(p.numel() * p.element_size() for p in gen.parameters())
    act = total_saved - min(total_saved, param_bytes)       # parameters are saved once; the rest scales with batch
    per_step_full = [b * B / pb for b in per_step]
    print(f'cpu probe with {pb} trajectories, {n_steps} steps, {time.time() - t0:.0f}s: autograd saved {gib(total_saved):.3f} GiB '
          f'(parameters {gib(param_bytes):.3f} GiB); first/median/last step {gib(per_step[0]):.4f}/{gib(sorted(per_step)[len(per_step)//2]):.4f}/{gib(per_step[-1]):.4f} GiB')
    est_act = act * B / pb
    print(f'scaled to batch {B} (splits {S} -> rollout of {B // S}): activations kept for backward '
          f'{gib(est_act / S):.1f} GiB (+ features {gib(feats_bytes / S):.1f} GiB + parameters/optimiser {gib(3 * param_bytes):.2f} GiB)')
    print(f'  = about {gib(est_act / S + feats_bytes / S + 3 * param_bytes):.1f} GiB live at the end of the first rollout; '
          f'per step at full batch: {gib(est_act / n_steps * 1.0 / S):.2f} GiB')
    print('  (steps in this probe: %d; a batch runs until its longest trajectory ends, at most %d steps)' % (n_steps, steps_max))


if __name__ == '__main__':
    main()
