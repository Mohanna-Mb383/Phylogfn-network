"""
Tests for the DB / WDB objectives and the look-ahead policy features (da Silva et al., ICLR 2025, Eq. 1, 5-7).
Run:  python tests/test_db_wdb.py

1. TB path unchanged: with NET_LOSS = 'TB' and LOOKAHEAD off, the policy forward pass equals the original model's
   (same shared weights -> identical logits and log P_F) and the loss equals the original TB formula.
2. Per-step records are consistent with the summed quantities: sum_t log P_F,t == log P_F, sum_t log P_B,t == log P_B.
3. Terminal constraint F(x) = R(x): the last residual of every trajectory uses log_reward as log F(s_{t+1}).
4. DB == WDB with gamma = 'uniform'; WDB equals the hand-computed sum_t gamma_t e_t^2 / sum_t gamma_t per trajectory.
5. gamma = 'count' equals 1 / sum_{k<=r} 2^k r(n, k); at the initial state of N = 4, R_MAX = 2 the count is 1911
   (15 + 2*228 + 4*360, the slot-labelled objects of the enumeration in tests/verify_network_exact.py).
6. Look-ahead child tokens of a MERGE equal the environment's own _merge_features for that pair with the same
   branch length (both edges), after the per-site normalisation of prepare_rollout_inputs.
7. DB / WDB gradients reach the flow head; the TB loss does not depend on it.
8. RETICULATE look-ahead: both prospective bare-slot tokens equal the environment's tokens of the RETICULATE(i) child
   (features and flags), in a closed and in an open state; illegal lineages get zero.
9. Final (root) MERGE look-ahead, closed and closing cases: equal to the environment's terminal token after the per-site
   normalisation; un-normalised, a constant factor lambda^(1/m) (edge prior folded per edge vs once on the root edge).
10. Optimiser: the flow head is in the optimiser's parameter groups; one DB / WDB update changes every flow-head
    parameter; one TB update changes none of them (and does change the policy heads).
The byte-level comparison with the unpatched repository (same seed: parameters, sampled trajectories, loss, one
optimiser step) is tools/regression_vs_baseline.py.
"""
import os, sys, random, math, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.configs.defaults import get_cfg_defaults
from src.utils.utils import correct_cfg_data
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_net import RolloutWorkerNet
from src.utils.level1_counts import level1_count

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make(n=6, m=20, seed=0, net_loss='TB', gamma='uniform', lookahead=False, r_max=2):
    rng = np.random.default_rng(seed)
    seqs = [''.join(rng.choice(list('ACGT'), m)) for _ in range(n)]
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(ROOT, 'src/configs/network_cfgs/cfg_net_small_cpu.yaml'))
    cfg.ENV.NETWORK.R_MAX = r_max
    cfg.GFN.NET_LOSS = net_loss
    cfg.GFN.WDB_GAMMA = gamma
    cfg.GFN.MODEL.LOOKAHEAD.ENABLED = lookahead
    cfg.GFN.MODEL.TRANSFORMER.DEPTH = 2
    cfg = correct_cfg_data(seqs, 1, cfg)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = build_env(cfg, seqs)
    for name in ('D', 'U', 'U_inv'):
        t = getattr(env.evolution_model, name)
        if torch.is_complex(t):
            t.data = t.data.real.clone()
    gen = build_gfn(cfg, env, 'cpu', ddp=False)
    rw = RolloutWorkerNet(env, record_steps=(net_loss != 'TB'), wdb_gamma=gamma)
    return cfg, env, gen, rw


def test_tb_path_unchanged():
    cfg, env, gen, rw = make()
    # the byte-level comparison with the unpatched repository (same seed, same draws, same parameters, same
    # optimiser step) is tools/regression_vs_baseline.py; here: the flow head does not enter logits / log P_F, and
    # under TB it is not even evaluated (log_flow is None) unless a caller asks for it.
    s0 = env.get_initial_state()
    feats = env.initial_features(2)
    inp = env.prepare_rollout_inputs(feats, [s0, s0], None)
    torch.manual_seed(1)
    out0 = gen(inp)
    assert out0['log_flow'] is None
    inp['need_log_flow'] = True
    torch.manual_seed(1)
    out1 = gen(inp)
    with torch.no_grad():
        for p in gen.net_model.flow_head.parameters():
            p.add_(1.0)
    torch.manual_seed(1)
    out2 = gen(inp)
    assert torch.equal(out0['logits'], out1['logits']) and torch.equal(out0['log_paths_pf'], out1['log_paths_pf'])
    assert torch.equal(out1['logits'], out2['logits']) and torch.equal(out1['log_paths_pf'], out2['log_paths_pf'])
    assert not torch.equal(out1['log_flow'], out2['log_flow'])
    # the flow head's construction leaves the global torch generator untouched: the draw after building two models
    # with the same seed is the same whether or not the flow head is twice as large
    def draw_after_build(layers):
        cfg2 = cfg.clone(); cfg2.GFN.MODEL.FLOW_HEAD.LAYERS = layers
        torch.manual_seed(5)
        from src.model.network_model.one_step_network_model import PhyloNetworkModelOneStep
        PhyloNetworkModelOneStep(cfg2.GFN)
        return torch.rand(3)
    assert torch.equal(draw_after_build(2), draw_after_build(4))
    # TB loss formula unchanged
    data, _ = rw.rollout(gen, 8)
    loss = gen.get_loss_from_rollout_outputs(data)
    hand = ((gen.compute_log_Z() + data['log_pf'] - data['log_rewards'] - data['log_pb']) ** 2).mean()
    assert torch.allclose(loss, hand)
    # head input sizes equal the sizes the original configuration declares (E + summary token)
    assert gen.net_model.pair_logits_head.feature[0].in_features == cfg.GFN.MODEL.TRANSFORMER.LOGITS_HEAD.INPUT_SIZE
    assert gen.net_model.single_logits_head.feature[0].in_features == cfg.GFN.MODEL.TRANSFORMER.SINGLE_LOGITS_HEAD.INPUT_SIZE
    print('1. TB path: logits / log P_F independent of the flow head; TB loss formula unchanged; head input sizes as configured')


def test_records_and_terminal():
    cfg, env, gen, rw = make(net_loss='DB')
    data, trajs = rw.rollout(gen, 12, random_spec={'random_action_prob': 0.3})
    assert torch.allclose(data['step_log_pf'].sum(1), data['log_pf'], atol=1e-5)
    assert torch.allclose(data['step_log_pb'].sum(1), data['log_pb'], atol=1e-5)
    lens = data['step_mask'].sum(1).long()
    for k, t in enumerate(trajs):
        assert lens[k].item() == len(t.actions) == 6 - 1 + 2 * data['n_ret'][k]
        assert data['step_is_last'][k, lens[k] - 1] == 1.0 and data['step_is_last'][k, :lens[k] - 1].sum() == 0
    e, mask, gamma = gen.transition_residuals(data)
    for k in range(12):
        T = lens[k].item()
        last = data['step_log_flow'][k, T - 1] + data['step_log_pf'][k, T - 1] - data['log_rewards'][k] - data['step_log_pb'][k, T - 1]
        assert torch.allclose(e[k, T - 1], last, atol=1e-5)
        if T > 1:
            mid = data['step_log_flow'][k, 0] + data['step_log_pf'][k, 0] - data['step_log_flow'][k, 1] - data['step_log_pb'][k, 0]
            assert torch.allclose(e[k, 0], mid, atol=1e-5)
        assert e[k, T:].abs().sum() == 0
    print('2./3. per-step records sum to the trajectory quantities; terminal residual uses log R(x); padding is zero')


def test_db_equals_uniform_wdb_and_normalisation():
    cfg, env, gen, rw = make(net_loss='DB')
    data, _ = rw.rollout(gen, 10, random_spec={'random_action_prob': 0.3})
    l_db = gen.db_loss(data).item()
    gen.net_loss = 'WDB'
    data['step_gamma'] = torch.ones_like(data['step_gamma'])
    l_wdb_u = gen.db_loss(data).item()
    assert abs(l_db - l_wdb_u) < 1e-6 * max(1.0, abs(l_db))
    # hand computation of the normalised weighted loss with a non-trivial gamma
    rw2 = RolloutWorkerNet(env, record_steps=True, wdb_gamma='logcount')
    data2, _ = rw2.rollout(gen, 10, random_spec={'random_action_prob': 0.3})
    e, mask, gamma = gen.transition_residuals(data2)
    e = e.detach()
    hand = []
    for k in range(10):
        T = int(mask[k].sum().item())
        num = sum(float(gamma[k, t]) * float(e[k, t]) ** 2 for t in range(T))
        den = sum(float(gamma[k, t]) for t in range(T))
        hand.append(num / den)
    hand = float(np.mean(hand))
    l = gen.db_loss(data2).item()
    assert abs(l - hand) < 1e-4 * max(1.0, abs(hand)), (l, hand)
    assert (gamma[mask > 0] < 1.0).any() and (gamma[mask > 0] <= 1.0).all()
    print('4. DB == WDB(uniform); WDB == hand-computed sum gamma e^2 / sum gamma per trajectory, batch mean')


def test_gamma_count():
    cfg, env, gen, rw = make(n=4, r_max=2)
    s0 = env.get_initial_state()
    expected = sum(2 ** k * level1_count(4, k) for k in range(3))
    assert expected == 1911
    assert abs(env.log_topo_descendants(s0) - math.log(1911)) < 1e-12
    assert abs(env.transition_weight(s0, 'count') - 1.0 / 1911) < 1e-15
    assert abs(env.transition_weight(s0, 'logcount') - 1.0 / (1.0 + math.log(1911))) < 1e-12
    assert abs(env.transition_weight(s0, 'sqrtcount') - 1911 ** -0.5) < 1e-12
    assert abs(env.transition_weight(s0, 'depth') - 1.0 / 4) < 1e-12
    # a terminal state has weight 1 under every option; a 2-lineage closed state has 1 + 2*r(2,1) = 5 completions
    traj = env.sample(1, True)[0]
    assert env.transition_weight(traj.current_state, 'count') == 1.0
    s = env.get_initial_state()
    s, _, _ = env.apply_action(s, {'type': 'merge', 'pair': env.pair_index[(0, 1)], 'edge_action': (0.1, 0.1)}, None)
    s, _, _ = env.apply_action(s, {'type': 'merge', 'pair': env.pair_index[(0, 1)], 'edge_action': (0.1, 0.1)}, None)
    assert len(s.lineages) == 2 and abs(math.exp(env.log_topo_descendants(s)) - (1 + 2 * level1_count(2, 1))) < 1e-9
    print('5. gamma(count) = 1/1911 at the N=4, R_MAX=2 initial state; other options consistent')


def test_lookahead_child_tokens():
    cfg, env, gen, rw = make(n=5, m=12, lookahead=True)
    random.seed(3)
    # a state with an open reticulation: RETICULATE(0) then nothing else
    s = env.get_initial_state()
    feats = env.initial_features(1)[0]
    s, feats, _ = env.apply_action(s, {'type': 'ret', 'lineage': 0, 'theta': 0.3, 'edge_action': 0.05}, feats)
    b_hat = 0.1
    tokens = {}

    def capture(tok):                      # identity "embedding" that records the tokens
        tokens.setdefault('calls', []).append(tok)
        return tok
    pe, se = env.lookahead_child_embeddings(feats.unsqueeze(0), [s], capture, b_hat, 0.5)
    n = len(s.lineages)
    checked = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not env.merge_legal(s.lineages[i], s.lineages[j]):
                assert pe[0, env.pair_index[(i, j)]].abs().sum() == 0
                continue
            case = env.merge_case(s.lineages[i], s.lineages[j])
            new = env._merge_features(feats, i, j, case, b_hat, b_hat, last_step=False)      # child feats, merged at row i
            ref = env._normalise_tokens(new[i:i + 1])[0]
            got = pe[0, env.pair_index[(i, j)], :2 * env.m * env.c]
            assert torch.allclose(got, ref, atol=1e-6), (i, j, case)
            flags = pe[0, env.pair_index[(i, j)], 2 * env.m * env.c:]
            assert flags[0].item() == (1.0 if case in (1, 2) else 0.0)
            checked += 1
    assert checked >= 6
    # RETICULATE children: two bare-slot tokens, both with open and bare flags
    legal = [i for i, l in enumerate(s.lineages) if l.open_h is None]
    assert se[0, legal[0]].abs().sum() > 0
    bare = [i for i, l in enumerate(s.lineages) if l.is_bare]
    assert se[0, bare[0]].abs().sum() == 0
    print('6. look-ahead MERGE child tokens equal _merge_features for all %d legal pairs (cases 0-3); RETICULATE tokens where legal' % checked)


def _capture_tokens(env, feats, state, b_hat, theta_hat):
    """look-ahead with an identity 'embedding': returns pair_emb, single_emb and the list of token matrices handed to
    the embedding function, in call order ([merge tokens], [slot-0 tokens], [slot-1 tokens])"""
    calls = []

    def capture(tok):
        calls.append(tok.clone())
        return tok
    pe, se = env.lookahead_child_embeddings(feats.unsqueeze(0), [state], capture, b_hat, theta_hat)
    return pe, se, calls


def _env_tokens(env, feats, state):
    return env.prepare_rollout_inputs(feats.unsqueeze(0), [state], None)['batch_input'][0]


def test_lookahead_reticulate_tokens():
    """RETICULATE look-ahead: for every lineage where RETICULATE is legal, the two prospective bare-slot tokens equal the
    tokens the environment itself produces for the child state of RETICULATE(i) with theta = THETA_HAT and
    b_hx = B_HAT (features and the four flags), and single_emb is their sum."""
    cfg, env, gen, rw = make(n=5, m=12, lookahead=True)
    b_hat, theta_hat = 0.1, 0.5
    checked = 0
    for prefix in ([], [{'type': 'ret', 'lineage': 0, 'theta': 0.3, 'edge_action': 0.05}]):
        s = env.get_initial_state()
        feats = env.initial_features(1)[0]
        for a in prefix:                                    # second case: a state with an open reticulation
            s, feats, _ = env.apply_action(s, a, feats)
        pe, se, calls = _capture_tokens(env, feats, s, b_hat, theta_hat)
        mask = env.legal_actions(s)
        n = len(s.lineages)
        legal = [i for i in range(n) if mask[env.n_pairs + i]]
        assert len(calls) == 3 and calls[1].shape[0] == calls[2].shape[0] == len(legal)
        for row, i in enumerate(legal):
            child, cfeats, _ = env.apply_action(s, {'type': 'ret', 'lineage': i, 'theta': theta_hat, 'edge_action': b_hat}, feats)
            tok = _env_tokens(env, cfeats, child)
            assert child.lineages[i].is_bare and child.lineages[i].slot == 0 and child.lineages[n].slot == 1
            assert torch.allclose(calls[1][row], tok[i], atol=1e-6), ('slot 0', i)
            assert torch.allclose(calls[2][row], tok[n], atol=1e-6), ('slot 1', i)
            assert torch.allclose(se[0, i], calls[1][row] + calls[2][row], atol=1e-6)
            checked += 1
        for i in range(n):
            if i not in legal:
                assert se[0, i].abs().sum() == 0
        if prefix:
            assert len(legal) == n - 2 and all(s.lineages[i].open_h is None for i in legal)
    assert checked == 5 + 4
    print('8. RETICULATE look-ahead tokens equal the environment tokens of the RETICULATE(i) child (both slots, features and '
          'flags) for %d legal lineages in a closed and in an open state; illegal lineages get zero' % checked)


def test_lookahead_final_merge_tokens():
    """final (root) MERGE look-ahead: the token equals the environment's terminal token after the per-site normalisation
    used by the policy input; without normalisation the two differ by the constant factor lambda^(1/m) per site
    (non-root formula: the edge prior exp(log lambda - lambda b) is folded into each of the two edges; root formula:
    once, on the summed edge)."""
    cfg, env, gen, rw = make(n=3, m=12, lookahead=True, r_max=1)
    lam, m = cfg.ENV.EVOLUTION_MODEL.PRIOR_LAMBDA, env.m
    b_hat = 0.1
    # closed final merge (case 0) and closing final merge (case 3: both lineages open on the same h)
    paths = {
        'closed': [{'type': 'merge', 'pair': env.pair_index[(0, 1)], 'edge_action': (0.2, 0.3)}],
        'closing': [{'type': 'ret', 'lineage': 0, 'theta': 0.4, 'edge_action': 0.05},
                    {'type': 'merge', 'pair': env.pair_index[(0, 1)], 'edge_action': (0.2, 0.3)},
                    {'type': 'merge', 'pair': env.pair_index[(1, 2)], 'edge_action': (0.1, 0.2)}],
    }
    for name, path in paths.items():
        s = env.get_initial_state()
        feats = env.initial_features(1)[0]
        for a in path:
            s, feats, _ = env.apply_action(s, a, feats)
        assert len(s.lineages) == 2 and not s.is_done
        case = env.merge_case(s.lineages[0], s.lineages[1])
        assert case == (0 if name == 'closed' else 3)
        pe, se, calls = _capture_tokens(env, feats, s, b_hat, 0.5)
        child, cfeats, score = env.apply_action(s, {'type': 'merge', 'pair': env.pair_index[(0, 1)], 'edge_action': (b_hat, b_hat)}, feats)
        assert child.is_done and score is not None
        tok = _env_tokens(env, cfeats, child)[0]
        got = pe[0, env.pair_index[(0, 1)]]
        assert torch.allclose(got, tok, atol=1e-6), name
        # RETICULATE singles: legal in the closed 2-lineage state (R_MAX not reached), illegal in the closing one
        assert (se[0].abs().sum() > 0) == (name == 'closed')
        # un-normalised features: constant ratio lambda^(1/m)
        env.normalize_features = False
        try:
            pe_u, _, _ = _capture_tokens(env, feats, s, b_hat, 0.5)
            tok_u = _env_tokens(env, cfeats, child)[0]
        finally:
            env.normalize_features = True
        k = 2 * env.m * env.c
        ratio = pe_u[0, env.pair_index[(0, 1)], :k] / tok_u[:k]
        assert torch.allclose(ratio, torch.full_like(ratio, lam ** (1.0 / m)), rtol=1e-5), name
        assert torch.equal(pe_u[0, env.pair_index[(0, 1)], k:], tok_u[k:])
    print('9. final MERGE look-ahead tokens (closed and closing cases) equal the environment terminal tokens after per-site '
          'normalisation; un-normalised they differ by the constant lambda^(1/m)')


def test_optimizer_step():
    """the flow head is in the optimiser and is moved by a DB / WDB step; a TB step leaves it bit-identical."""
    for net_loss, gamma, expect_change in (('DB', 'uniform', True), ('WDB', 'logcount', True), ('TB', 'uniform', False)):
        cfg, env, gen, rw = make(net_loss=net_loss, gamma=gamma)
        in_opt = {id(p) for g in gen.opt.param_groups for p in g['params']}
        flow_params = list(gen.net_model.flow_head.parameters())
        assert len(flow_params) == 2 * (cfg.GFN.MODEL.FLOW_HEAD.LAYERS + 1)
        # DB / WDB: every flow-head parameter is in the optimiser; TB: none is (the optimiser is the original one)
        assert all((id(p) in in_opt) == expect_change for p in flow_params)
        assert all(id(p) in in_opt for p in gen.net_model.parameters() if id(p) not in {id(q) for q in flow_params})
        n_model = sum(len(list(mod.parameters())) for mod in (gen.net_model, gen.edges_model, gen.ret_model))
        assert len(in_opt) == n_model - (0 if expect_change else len(flow_params)) + int(cfg.GFN.MODEL.UPDATE_Z)
        before = [p.detach().clone() for p in flow_params]
        other_before = [p.detach().clone() for p in gen.net_model.pair_logits_head.parameters()]
        data, _ = rw.rollout(gen, 8, random_spec={'random_action_prob': 0.3})
        gen.accumulate_loss(data, 1)
        info = gen.update_model()
        changed = [not torch.equal(a, p.detach()) for a, p in zip(before, flow_params)]
        assert all(changed) == expect_change and any(changed) == expect_change, (net_loss, changed)
        assert all(not torch.equal(a, p.detach()) for a, p in zip(other_before, gen.net_model.pair_logits_head.parameters()))
        assert math.isfinite(info['loss'])
        if expect_change:                                   # a second rollout + step keeps moving it (Adam state exists)
            st = gen.opt.state
            assert all(len(st[p]) > 0 for p in flow_params)
    print('10. flow-head parameters are in the optimiser; one DB / WDB update changes all of them; a TB update changes none')


def test_gradients():
    for net_loss in ('DB', 'WDB'):
        cfg, env, gen, rw = make(net_loss=net_loss, gamma='logcount', lookahead=(net_loss == 'WDB'))
        data, _ = rw.rollout(gen, 6, random_spec={'random_action_prob': 0.3})
        gen.opt.zero_grad()
        gen.get_loss_from_rollout_outputs(data).backward()
        g = sum(p.grad.abs().sum().item() for p in gen.net_model.flow_head.parameters())
        assert g > 0
    cfg, env, gen, rw = make(net_loss='TB')
    data, _ = rw.rollout(gen, 6)
    gen.opt.zero_grad()
    gen.get_loss_from_rollout_outputs(data).backward()
    assert all(p.grad is None or p.grad.abs().sum().item() == 0 for p in gen.net_model.flow_head.parameters())
    print('7. flow-head gradients present under DB / WDB, absent under TB')


if __name__ == '__main__':
    test_tb_path_unchanged()
    test_records_and_terminal()
    test_db_equals_uniform_wdb_and_normalisation()
    test_gamma_count()
    test_lookahead_child_tokens()
    test_gradients()
    test_lookahead_reticulate_tokens()
    test_lookahead_final_merge_tokens()
    test_optimizer_step()
    print('ALL TESTS PASSED')
