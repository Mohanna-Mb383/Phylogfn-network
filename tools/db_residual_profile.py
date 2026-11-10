"""
db_residual_profile.py -- transition-level detailed-balance residuals of a DB / WDB checkpoint
(da Silva et al., ICLR 2025, Figure 3, for the network MDP).

    python tools/db_residual_profile.py <run_dir> <sequences.pickle> [--trajectories=256] [--eps=0.0]
                                        [--device=cpu|cuda] [--checkpoint=<path>] [--seed=0] [--out=<file>]

For every recorded transition s_t -> s_{t+1} of --trajectories rollouts (policy mixed with uniform random actions
with probability --eps, as during training):
    e_t = log F(s_t) + log P_F(s_{t+1}|s_t) - log F(s_{t+1}) - log P_B(s_t|s_{t+1}),   F(x) = R(x) at terminals,
and prints the mean of e_t^2 (and the gamma-weighted share of the WDB loss) by depth t, by action type
(MERGE of two closed lineages, MERGE with one open lineage, closing MERGE, RETICULATE), by the number of open
reticulations of s_t, by the number of lineages of s_{t+1}, and by the reticulation count R(x) of the trajectory.
Requires a run trained with GFN.NET_LOSS = DB or WDB (the flow head of a TB run is untrained: the tool refuses it).
"""
import os, sys, glob, random, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.configs.defaults import get_cfg_defaults
from src.utils.utils import correct_cfg_data, load_sequences
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_net import RolloutWorkerNet, ACTION_TYPES

KIND_NAMES = {v: k for k, v in ACTION_TYPES.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir'); ap.add_argument('sequences')
    ap.add_argument('--trajectories', type=int, default=256)
    ap.add_argument('--eps', type=float, default=0.0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    seqs = load_sequences(args.sequences)
    cfg = get_cfg_defaults(); cfg.merge_from_file(os.path.join(args.run_dir, 'config.yaml')); cfg = correct_cfg_data(seqs, 1, cfg)
    if cfg.GFN.NET_LOSS == 'TB':
        raise SystemExit('run was trained with TB: no trained flow head, transition residuals are undefined')
    env = build_env(cfg, seqs)
    for name in ('D', 'U', 'U_inv'):
        t = getattr(env.evolution_model, name)
        if torch.is_complex(t):
            t.data = t.data.real.clone()
    env.to(args.device)
    gen = build_gfn(cfg, env, args.device, ddp=False)
    ck = args.checkpoint or sorted(glob.glob(os.path.join(args.run_dir, 'checkpoints', 'checkpoint_*.pt')))[-1]
    gen.load(ck); gen.eval()
    env.reward_fn.scale = 1.0
    rw = RolloutWorkerNet(env, record_steps=True, wdb_gamma=cfg.GFN.WDB_GAMMA)
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    say('run %s  checkpoint %s  objective %s  gamma %s  N=%d  R_MAX=%d  trajectories %d  eps %.2f' % (
        args.run_dir, os.path.basename(ck), cfg.GFN.NET_LOSS, cfg.GFN.WDB_GAMMA, env.n_leaves, env.r_max, args.trajectories, args.eps))
    with torch.no_grad():
        data, _ = rw.rollout(gen, args.trajectories, random_spec={'random_action_prob': args.eps} if args.eps > 0 else None)
        e, mask, gamma = gen.transition_residuals(data)
    e2 = (e ** 2).cpu().numpy(); m = mask.cpu().numpy() > 0
    g = (data['step_gamma'].cpu().numpy() if cfg.GFN.NET_LOSS == 'WDB' else np.ones_like(e2)) * m
    gnorm = g / np.maximum(g.sum(1, keepdims=True), 1e-300)
    contrib = gnorm * e2                                   # per-transition share of the per-trajectory loss
    depth = data['step_depth'].numpy(); kind = data['step_kind'].numpy(); n_open = data['step_n_open'].numpy()
    n_lin = data['step_n_lin_child'].numpy(); R = data['n_ret']
    say('overall: mean e^2 over transitions %.4f   per-trajectory loss (mean) %.4f   trajectories by R %s' % (
        e2[m].mean(), contrib.sum(1).mean(), np.bincount(R, minlength=env.r_max + 1).tolist()))

    def table(name, key, values):
        say('by %s: value -> mean e^2 (n transitions) | mean share of the trajectory loss' % name)
        for v in values:
            sel = m & (key == v)
            if sel.sum() == 0:
                continue
            say('   %-14s %10.4f (%6d) | %.4f' % (str(v), e2[sel].mean(), sel.sum(), contrib[sel].sum() / len(e2)))
    table('depth t', depth, sorted(np.unique(depth[m])))
    table('action type', kind, sorted(np.unique(kind[m])))
    say('   action types: ' + ', '.join('%d=%s' % (k, v) for k, v in KIND_NAMES.items()))
    table('open reticulations of s_t', n_open, sorted(np.unique(n_open[m])))
    table('lineages of s_{t+1}', n_lin, sorted(np.unique(n_lin[m])))
    Rb = np.repeat(R[:, None], e2.shape[1], axis=1)
    table('R(x) of the trajectory', Rb, sorted(np.unique(R)))
    # root-side vs terminal-side thirds
    L = m.sum(1, keepdims=True)
    frac = (depth + 1) / np.maximum(L, 1)
    third = np.where(frac <= 1 / 3, 0, np.where(frac <= 2 / 3, 1, 2))
    table('trajectory third (0 root side, 2 terminal side)', third, [0, 1, 2])
    if args.out:
        open(args.out, 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
