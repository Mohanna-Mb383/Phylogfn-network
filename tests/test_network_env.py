"""
Tests for the PhyloGFN-Net environment.  Run:   python -m pytest tests -q
(or simply  python tests/test_network_env.py)

1. the MDP with the level-1 masks reaches EXACTLY the class of binary level-1
   networks: 603 networks for N = 4, R <= 2 and 11,460 for N = 5, R <= 2
   (Bouvel, Gambette & Mansouri 2020; enum_level1.py);
2. no reachable non-terminal state is a dead end;
3. the folded "mix at the gall" likelihood equals the explicit 2^R displayed-tree
   likelihood on random networks;
4. backward sampling from a network replays to the same network and the same score;
5. trajectory length = N - 1 + 2R;  |Pa(s)| agrees with the backward sampler.
"""
import os, sys, random, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.configs.defaults import get_cfg_defaults
from src.utils.utils import correct_cfg_data
from src.env.phylo_network_env import PhyloNetworkEnv, network_canonical_string, NetworkState
from src.utils.network_likelihood import network_log_score_explicit

EXPECTED = {(4, 2): 603, (5, 2): 11460, (4, 1): 243, (5, 1): 2910, (3, 2): 36}


def make_env(n, r_max, m=30, seed=0, lambda_r=0.7, prior_type='EXP_R'):
    rng = np.random.default_rng(seed)
    seqs = [''.join(rng.choice(list('ACGT'), m)) for _ in range(n)]
    cfg = get_cfg_defaults()
    cfg.ENV.ENVIRONMENT_TYPE = 'ONE_STEP_LEVEL1_NETWORK'
    cfg.ENV.SEQUENCE_TYPE = 'DNA_WITH_GAP'
    cfg.ENV.REWARD.RESHAPE_METHOD = 'EXPONENTIAL'
    cfg.ENV.REWARD.C = 0.0
    cfg.ENV.NETWORK.R_MAX = r_max
    cfg.ENV.NETWORK.LAMBDA_R = lambda_r
    cfg.ENV.NETWORK.PRIOR_TYPE = prior_type
    cfg.PARSIMONY_PROBLEM = False
    cfg.GFN.NORMALIZE_LIKELIHOOD = True
    cfg = correct_cfg_data(seqs, 1, cfg)
    return PhyloNetworkEnv(cfg, seqs)


def state_key(state):
    """canonical string of a partial state (order-free), for memoised search"""
    parts = []
    for l in state.lineages:
        if l.is_bare:
            parts.append('S%d#' % l.slot + network_canonical_string(l.node.children[0], 10 ** 6) + '@%d' % l.node.idx)
        else:
            parts.append(network_canonical_string(l.node, 10 ** 6))
    return '|'.join(sorted(parts)) + '|R%d' % state.n_ret


def enumerate_terminal(env):
    """DFS over the MDP ignoring continuous parameters; returns set of canonical terminal networks
    and checks that no reachable state is a dead end."""
    seen = set()
    terminals = set()
    stack = [env.get_initial_state()]
    dead_ends = 0
    while stack:
        s = stack.pop()
        k = state_key(s)
        if k in seen:
            continue
        seen.add(k)
        if s.is_done:
            terminals.add(network_canonical_string(s.lineages[0].node, env.n_leaves))
            continue
        ids = np.where(env.legal_actions(s))[0]
        if len(ids) == 0:
            dead_ends += 1
        for a in ids:
            action = env.action_from_id(int(a), s, random_params=True)
            ns, _, _ = env.apply_action(s, action, None)
            stack.append(ns)
    return terminals, dead_ends, len(seen)


def test_class_counts_and_no_dead_ends():
    for (n, r), expected in EXPECTED.items():
        env = make_env(n, r)
        terminals, dead_ends, n_states = enumerate_terminal(env)
        print(f'N={n} R_MAX={r}: reachable terminal networks {len(terminals)} (expected {expected}); '
              f'{n_states} distinct states; dead ends {dead_ends}')
        assert len(terminals) == expected
        assert dead_ends == 0


def test_likelihood_matches_explicit():
    random.seed(1); torch.manual_seed(1)
    worst = 0.0
    for n, r, pt in [(4, 1, 'EXP_R'), (5, 2, 'UNIFORM_R'), (6, 3, 'EXP_R'), (7, 2, 'UNIFORM_R')]:
        env = make_env(n, r, m=25, prior_type=pt)
        for _ in range(12):
            traj = env.sample(1, True)[0]
            net = traj.current_state.network
            fast = traj.current_state.log_score
            slow = network_log_score_explicit(net, env)
            worst = max(worst, abs(fast - slow))
            assert abs(fast - slow) < 1e-7 * max(1.0, abs(slow)), (n, r, fast, slow)
    print('max |fast - explicit| =', worst)


def test_backward_sampling_roundtrip():
    random.seed(2); torch.manual_seed(2)
    for n, r in [(5, 2), (6, 2), (8, 3)]:
        env = make_env(n, r, m=20)
        for _ in range(10):
            traj = env.sample(1, True)[0]
            net = traj.current_state.network
            actions, log_pb = env.sample_backward_from_network(net)
            assert len(actions) == n - 1 + 2 * net.num_reticulations()
            # replay forward: same network, same score, same |Pa| sequence
            state = env.get_initial_state()
            feats = env.initial_features(1)[0]
            pa = []
            for a in actions:
                state, feats, score = env.apply_action(state, a, feats)
                pa.append(state.num_parents())
            assert state.is_done
            assert network_canonical_string(state.lineages[0].node, n) == net.topology_id
            assert abs(score - net.log_score) < 1e-6
            assert np.allclose(-np.log(np.array(pa, dtype=float)), log_pb), (pa, log_pb)
    print('backward round trip ok')


def test_trajectory_length_and_reward():
    random.seed(3)
    env = make_env(6, 2, m=20)
    for traj in env.sample(20, True):
        R = traj.current_state.network.num_reticulations()
        assert len(traj.actions) == 6 - 1 + 2 * R
        assert R <= 2
        assert math.isfinite(traj.log_reward)
    print('trajectory lengths ok')


if __name__ == '__main__':
    test_class_counts_and_no_dead_ends()
    test_likelihood_matches_explicit()
    test_backward_sampling_roundtrip()
    test_trajectory_length_and_reward()
    print('ALL TESTS PASSED')
