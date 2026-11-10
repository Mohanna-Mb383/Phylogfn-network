"""
regression_vs_baseline.py -- byte-level regression test of the TB path against the unpatched repository.

    python tools/regression_vs_baseline.py [--baseline=<dir>] [--baseline_rev=54510ac] [--seed=0] [--n=6] [--m=20]
                                           [--episodes=16] [--keep]

The same worker (this file, --worker mode) is run twice as a subprocess with the SAME seed: once with the unpatched
repository on sys.path (a `git worktree` of --baseline_rev, created and removed by this script unless --baseline is
given), once with this repository.  Each worker, with GFN.NET_LOSS = TB and look-ahead off:
  1. seeds python random, numpy and torch, builds the environment on the same random alignment and the generator;
  2. saves the generator's state_dict (every parameter and buffer);
  3. runs a rollout of --episodes trajectories with epsilon = 0.3 exploration and saves the sampled actions
     (types, pairs / lineages, branch lengths, theta), log P_F, log P_B, log R, the logits of the first step and the
     TB loss;
  4. takes one optimiser step (clip + Adam) and saves the state_dict again, then a second rollout and 5 random
     environment trajectories (python `random` stream).
The parent compares the two outputs: every tensor present in the baseline must be bit-identical in the patched run
(torch.equal), the sampled actions must be identical, and the only parameters the patched run may add are the flow
head's (net_model.flow_head.*).  A difference anywhere means the patch changed the TB training path or its random
stream; the flow head is built on a forked generator precisely so that this test passes.
"""
import os, sys, subprocess, tempfile, shutil, argparse, random, math


def worker(repo, out, seed, n, m, episodes):
    sys.path.insert(0, repo)
    os.chdir(repo)
    import numpy as np
    import torch
    from src.configs.defaults import get_cfg_defaults
    from src.utils.utils import correct_cfg_data
    from src.env import build_env
    from src.gfn.build import build_gfn
    from src.gfn.rollout_worker_net import RolloutWorkerNet
    rng = np.random.default_rng(12345)                          # alignment: identical in both workers
    seqs = [''.join(rng.choice(list('ACGT'), m)) for _ in range(n)]
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(repo, 'src/configs/network_cfgs/cfg_net_small_cpu.yaml'))
    cfg.ENV.NETWORK.R_MAX = 2
    cfg.GFN.MODEL.TRANSFORMER.DEPTH = 2
    cfg = correct_cfg_data(seqs, 1, cfg)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = build_env(cfg, seqs)
    for name in ('D', 'U', 'U_inv'):                            # numpy >= 2: complex eigendecomposition guard (both workers)
        t = getattr(env.evolution_model, name)
        if torch.is_complex(t):
            t.data = t.data.real.clone()
    gen = build_gfn(cfg, env, 'cpu', ddp=False)
    rw = RolloutWorkerNet(env)
    res = {'repo': repo, 'state_dict_0': {k: v.detach().clone() for k, v in gen.state_dict().items()}}

    def record_rollout(tag):
        data, trajs = rw.rollout(gen, episodes, random_spec={'random_action_prob': 0.3}, generate_full_trajectories=True)
        acts = []
        for t in trajs:
            acts.append([(a['type'], int(a.get('pair', -1)), int(a.get('lineage', -1)),
                          tuple(float(x) for x in a['edge_action']) if isinstance(a['edge_action'], (tuple, list)) else float(a['edge_action']),
                          float(a.get('theta', -1.0))) for a in t.actions])
        res[tag] = {'actions': acts, 'log_pf': data['log_pf'].detach().clone(), 'log_pb': data['log_pb'].detach().clone(),
                    'log_rewards': data['log_rewards'].detach().clone(), 'n_ret': data['n_ret'].tolist()}
        return data
    s0 = env.get_initial_state()
    inp = env.prepare_rollout_inputs(env.initial_features(2), [s0, s0], None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        first = gen(inp)
    res['first_logits'] = first['logits'].detach().clone()
    res['first_log_pf'] = first['log_paths_pf'].detach().clone()
    data = record_rollout('rollout_1')
    loss = gen.get_loss_from_rollout_outputs(data)
    res['tb_loss'] = loss.detach().clone()
    gen.accumulate_loss(data, 1)
    info = gen.update_model()
    res['grad_norm'] = float(info['grad_norm']); res['param_norm'] = float(info['param_norm'])
    res['state_dict_1'] = {k: v.detach().clone() for k, v in gen.state_dict().items()}
    record_rollout('rollout_2')
    res['env_sample_scores'] = [float(t.log_reward) for t in env.sample(5, False)]
    res['log_Z'] = float(gen.log_Z())
    torch.save(res, out)


def compare(a, b):
    import torch
    problems = []
    for tag in ('state_dict_0', 'state_dict_1'):
        extra = [k for k in b[tag] if k not in a[tag]]
        bad_extra = [k for k in extra if not k.startswith('net_model.flow_head.')]
        if bad_extra:
            problems.append('%s: unexpected new tensors %s' % (tag, bad_extra))
        for k, v in a[tag].items():
            if k not in b[tag]:
                problems.append('%s: %s missing in the patched run' % (tag, k))
            elif not torch.equal(v, b[tag][k]):
                problems.append('%s: %s differs (max abs %.3e)' % (tag, k, (v.double() - b[tag][k].double()).abs().max().item()))
        print('%s: %d tensors compared, %d added by the patch (all flow head: %s)' % (tag, len(a[tag]), len(extra), not bad_extra))
    for key in ('first_logits', 'first_log_pf', 'tb_loss'):
        if not torch.equal(a[key], b[key]):
            problems.append('%s differs (max abs %.3e)' % (key, (a[key].double() - b[key].double()).abs().max().item()))
    for tag in ('rollout_1', 'rollout_2'):
        if a[tag]['actions'] != b[tag]['actions']:
            problems.append('%s: sampled actions differ' % tag)
        for key in ('log_pf', 'log_pb', 'log_rewards'):
            if not torch.equal(a[tag][key], b[tag][key]):
                problems.append('%s: %s differs (max abs %.3e)' % (tag, key, (a[tag][key].double() - b[tag][key].double()).abs().max().item()))
        if a[tag]['n_ret'] != b[tag]['n_ret']:
            problems.append('%s: n_ret differs' % tag)
        steps = sum(len(x) for x in a[tag]['actions'])
        print('%s: %d trajectories, %d actions, log R mean %.4f: %s' % (tag, len(a[tag]['actions']), steps, a[tag]['log_rewards'].mean().item(),
                                                                    'identical' if not [p for p in problems if p.startswith(tag)] else 'DIFFERENT'))
    for key in ('grad_norm', 'param_norm', 'log_Z'):
        if a[key] != b[key]:
            problems.append('%s differs: %r vs %r' % (key, a[key], b[key]))
    if a['env_sample_scores'] != b['env_sample_scores']:
        problems.append('env.sample (python random stream) differs')
    print('TB loss %.6f (baseline) vs %.6f (patched); grad norm %.6f vs %.6f; log Z %.6f vs %.6f' % (
        a['tb_loss'].item(), b['tb_loss'].item(), a['grad_norm'], b['grad_norm'], a['log_Z'], b['log_Z']))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', nargs=2, metavar=('REPO', 'OUT'), default=None)
    ap.add_argument('--baseline', default=None, help='directory with the unpatched repository (default: git worktree of --baseline_rev)')
    ap.add_argument('--baseline_rev', default='54510ac')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=6); ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--episodes', type=int, default=16)
    ap.add_argument('--keep', action='store_true', help='keep the worker outputs')
    args = ap.parse_args()
    if args.worker:
        worker(os.path.abspath(args.worker[0]), os.path.abspath(args.worker[1]), args.seed, args.n, args.m, args.episodes)
        return
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = tempfile.mkdtemp(prefix='regression_')
    made_worktree = False
    try:
        if args.baseline:
            base = os.path.abspath(args.baseline)
        else:
            base = os.path.join(tmp, 'baseline')
            subprocess.check_call(['git', '-C', here, 'worktree', 'add', '--detach', base, args.baseline_rev])
            made_worktree = True
        outs = {}
        for tag, repo in (('baseline', base), ('patched', here)):
            out = os.path.join(tmp, tag + '.pt')
            env = dict(os.environ); env['PYTHONPATH'] = repo; env['PYTHONHASHSEED'] = '0'
            subprocess.check_call([sys.executable, os.path.abspath(__file__), '--worker', repo, out, '--seed', str(args.seed),
                                   '--n', str(args.n), '--m', str(args.m), '--episodes', str(args.episodes)], cwd=repo, env=env)
            outs[tag] = out
        import torch
        a = torch.load(outs['baseline'], weights_only=False); b = torch.load(outs['patched'], weights_only=False)
        print('baseline %s\npatched  %s\nseed %d, N = %d, m = %d, %d episodes' % (a['repo'], b['repo'], args.seed, args.n, args.m, args.episodes))
        problems = compare(a, b)
        if problems:
            print('REGRESSION: %d differences' % len(problems))
            for p in problems:
                print('  ' + p)
            sys.exit(1)
        print('REGRESSION TEST PASSED: parameters, sampled trajectories, log P_F / log P_B / log R, TB loss, optimiser step and random '
              'streams are bit-identical to the unpatched repository')
    finally:
        if made_worktree:
            subprocess.call(['git', '-C', here, 'worktree', 'remove', '--force', base])
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
