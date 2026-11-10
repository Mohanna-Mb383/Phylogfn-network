"""
reference_posterior_small_n.py -- enumerated-topology numerical reference posterior for a small problem
(N = 4 leaves, R_MAX = 2: the 603 rooted binary leaf-labelled level-1 networks with at most two reticulations
as graphs; N = 3: 36 at R_MAX = 2) and the total-variation distance of a trained sampler's topology
frequencies to it.  This is the validation of FCS as a TV proxy for networks (IDEAS v3, RQ1 item 5).

    python tools/reference_posterior_small_n.py <cfg.yaml> <sequences.pickle> [--draws=2000] [--seed=0]
        [--run_dir=<run>] [--checkpoint=<path>] [--samples=4096] [--floor_reps=20] [--device=cpu] [--out=<file>]
        [--save_reference=<file.pkl>] [--load_reference=<file.pkl>]

--save_reference stores the enumeration and the per-graph integration draws (so that tools/auc_endpoints.py and later
calls with --load_reference reuse the same numerical reference instead of recomputing it); --load_reference refuses a
file made for another problem (leaves, R_MAX, sites, priors, draws or seed differ).  A sampled network whose
canonical string is not in the enumerated class is an error (RuntimeError), never silently counted.

Reference: for every graph G, p(Y|G) = E_{b ~ Exp(PRIOR_LAMBDA) per edge (root edges: one draw for their sum),
theta ~ U(0,1)} [ L(Y|G, b, theta) ] estimated by --draws Monte-Carlo draws (relative standard error printed per
graph; the integrator is exact in the limit, the reference is NUMERICAL, not exact).  Posterior over graphs under
the stated prior exp(-LAMBDA_R R)/Z: pi_graph(G) ~ exp(-LAMBDA_R R(G)) p(Y|G); under the prior the sampler actually
targets (2^R slot labellings): pi_lab(G) ~ 2^R exp(-LAMBDA_R R) p(Y|G).
Sampler: --samples networks from the checkpoint, aggregated over slot labellings by the canonical string;
TV(frequencies, pi_lab); TV(frequencies reweighted by 2^-R, pi_graph).  Two error floors: finite-sample floor
(draw --samples from pi_lab itself, --floor_reps times) and integration floor (bootstrap of the per-graph draws).
"""
import os, sys, glob, math, random, argparse, copy, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.configs.defaults import get_cfg_defaults
from src.utils.utils import correct_cfg_data, load_sequences
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_net import RolloutWorkerNet
from src.env.phylo_network_env import PhyloNetwork, network_canonical_string


def slot_canon(v, via_slot=None, memo=None):
    memo = {} if memo is None else memo
    key = (v.idx, via_slot)
    if key in memo:
        return memo[key]
    if v.kind == 'leaf':
        s = 'L%d' % v.idx
    elif v.kind == 'ret':
        s = '#%s(%s)' % ('?' if via_slot is None else via_slot, slot_canon(v.children[0], None, memo))
    else:
        parts = []
        for c in v.children:
            vs = next(k for k in (0, 1) if c.parents[k] is v) if c.kind == 'ret' else None
            parts.append(slot_canon(c, vs, memo))
        s = '(' + ','.join(sorted(parts)) + ')'
    memo[key] = s
    return s


def state_key(state):
    parts, memo = [], {}
    for l in state.lineages:
        if l.is_bare:
            parts.append('S%d:%s' % (l.slot, slot_canon(l.node.children[0], None, memo)))
        else:
            s = slot_canon(l.node, None, memo)
            if l.open_h is not None:
                s = 'O[%s]' % slot_canon(l.open_h.children[0], None, memo) + s
            parts.append(s)
    return '|'.join(sorted(parts)) + '|R%d' % state.n_ret


def enumerate_graphs(env):
    """one representative terminal NetNode per graph (canonical string ignoring slot labels)"""
    seen, graphs = set(), {}
    s0 = env.get_initial_state()
    stack = [s0]; seen.add(state_key(s0))
    while stack:
        s = stack.pop()
        if s.is_done:
            root = s.lineages[0].node
            graphs.setdefault(network_canonical_string(root, env.n_leaves), root)
            continue
        for a in np.where(env.legal_actions(s))[0]:
            ns, _, _ = env.apply_action(copy.deepcopy(s), env.action_from_id(int(a), s, random_params=True), None)
            k = state_key(ns)
            if k not in seen:
                seen.add(k); stack.append(ns)
    return graphs


def n_ret_of(root):
    out, seen, stack = 0, set(), [root]
    while stack:
        v = stack.pop()
        if v.idx in seen:
            continue
        seen.add(v.idx); out += v.kind == 'ret'; stack.extend(v.children)
    return out


def log_lik_draws(env, actions, M, lam, rng):
    """M draws of log L(Y | G, b, theta) with b, theta from the prior, by replaying `actions` with fresh parameters"""
    out = np.empty(M)
    for i in range(M):
        acts, logprior = [], 0.0
        state = env.get_initial_state()
        n = len(state.lineages)
        for a in actions:
            a = dict(a)
            if a['type'] == 'merge':
                if n == 2:
                    t = rng.exponential(1.0 / lam); a['edge_action'] = (t / 2, t / 2); logprior += math.log(lam) - lam * t
                else:
                    b1, b2 = rng.exponential(1.0 / lam, 2); a['edge_action'] = (float(b1), float(b2)); logprior += 2 * math.log(lam) - lam * (b1 + b2)
                n -= 1
            else:
                b = rng.exponential(1.0 / lam); a['edge_action'] = float(b); a['theta'] = float(rng.uniform(0, 1)); logprior += math.log(lam) - lam * b
                n += 1
            acts.append(a)
        feats = env.initial_features(1)[0]
        state = env.get_initial_state()
        for a in acts:
            state, feats, score = env.apply_action(state, a, feats)
        R = state.n_ret
        out[i] = score - logprior - env.log_topology_prior_unnormalised(R)
    return out


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def build_problem_env(cfg_path, seqs):
    cfg = get_cfg_defaults(); cfg.merge_from_file(cfg_path); cfg = correct_cfg_data(seqs, 1, cfg)
    env = build_env(cfg, seqs)
    for name in ('D', 'U', 'U_inv'):                      # numpy >= 2 may return a complex JC eigendecomposition
        t = getattr(env.evolution_model, name)
        if torch.is_complex(t):
            t.data = t.data.real.clone()
    return cfg, env


def reference_identity(env, lam, draws, seed):
    return {'n_leaves': env.n_leaves, 'r_max': env.r_max, 'sites': env.m, 'prior_lambda': float(lam),
            'lambda_r': float(env.lambda_r), 'prior_type': env.prior_type, 'draws': int(draws), 'seed': int(seed)}


def posterior_over_graphs(logp, R, lambda_r, labelled):
    """pi(G) ~ exp(-lambda_r R) p(Y|G), times 2^R for the labelled target the sampler actually targets"""
    lw = logp - lambda_r * R + (R * math.log(2) if labelled else 0.0)
    w = np.exp(lw - lw.max()); return w / w.sum()


def compute_reference(env, lam, draws, seed, say=print):
    """enumerate the graphs and integrate p(Y|G) by Monte Carlo for each; returns a picklable dict"""
    rng = np.random.default_rng(seed)
    graphs = enumerate_graphs(env)
    keys = sorted(graphs)
    R = np.array([n_ret_of(graphs[k]) for k in keys])
    say('N=%d R_MAX=%d: %d graphs enumerated; by R %s; sites %d; PRIOR_LAMBDA %g; LAMBDA_R %g; draws per graph %d' % (
        env.n_leaves, env.r_max, len(keys), np.bincount(R, minlength=env.r_max + 1).tolist(), env.m, lam, env.lambda_r, draws))
    logp = np.empty(len(keys)); relse = np.empty(len(keys)); draws_all = []
    for g, k in enumerate(keys):
        acts, _ = env.sample_backward_from_network(PhyloNetwork(graphs[k], None, env.n_leaves))
        ll = log_lik_draws(env, acts, draws, lam, rng)
        draws_all.append(ll)
        mx = ll.max(); w = np.exp(ll - mx)
        logp[g] = mx + math.log(w.mean())
        relse[g] = w.std() / w.mean() / math.sqrt(len(ll))
    say('reference log p(Y|G): relative standard error per graph: median %.3f, max %.3f' % (np.median(relse), relse.max()))
    return {'identity': reference_identity(env, lam, draws, seed), 'keys': keys, 'R': R, 'logp': logp, 'relse': relse,
            'draws_all': np.stack(draws_all)}


def save_reference(ref, path):
    with open(path, 'wb') as f:
        pickle.dump(ref, f)


def load_reference(path, env, lam, draws=None, seed=None):
    """load a saved reference and refuse one made for a different problem (draws / seed are checked when given)"""
    with open(path, 'rb') as f:
        ref = pickle.load(f)
    want = reference_identity(env, lam, draws if draws is not None else ref['identity']['draws'],
                              seed if seed is not None else ref['identity']['seed'])
    if ref['identity'] != want:
        raise RuntimeError('reference %s was made for another problem: %s vs %s' % (path, ref['identity'], want))
    return ref


def reference_floors(ref, lambda_r, samples, floor_reps, seed, say=print):
    """integration floor (bootstrap of the per-graph draws) and finite-sample floor (draws from the reference itself)"""
    rng = np.random.default_rng(seed + 1)
    keys, R, logp, draws_all = ref['keys'], ref['R'], ref['logp'], ref['draws_all']
    pi_lab = posterior_over_graphs(logp, R, lambda_r, True)
    tvs = []
    for _ in range(floor_reps):
        lp_b = np.empty(len(keys))
        for g in range(len(keys)):
            d = draws_all[g][rng.integers(0, draws_all.shape[1], draws_all.shape[1])]
            mx = d.max(); lp_b[g] = mx + math.log(np.exp(d - mx).mean())
        tvs.append(tv(posterior_over_graphs(lp_b, R, lambda_r, True), pi_lab))
    say('integration floor: TV(bootstrap reference, reference) = %.4f +- %.4f over %d replicates' % (np.mean(tvs), np.std(tvs), floor_reps))
    int_floor = (float(np.mean(tvs)), float(np.std(tvs)))
    tvs = []
    for _ in range(floor_reps):
        c = rng.multinomial(samples, pi_lab) / samples
        tvs.append(tv(c, pi_lab))
    say('finite-sample floor: TV(%d draws from the reference, reference) = %.4f +- %.4f' % (samples, np.mean(tvs), np.std(tvs)))
    return int_floor, (float(np.mean(tvs)), float(np.std(tvs)))


def load_run_generator(run_dir, seqs, device, checkpoint=None):
    cfg_r = get_cfg_defaults(); cfg_r.merge_from_file(os.path.join(run_dir, 'config.yaml')); cfg_r = correct_cfg_data(seqs, 1, cfg_r)
    env_r = build_env(cfg_r, seqs)
    for name in ('D', 'U', 'U_inv'):
        t = getattr(env_r.evolution_model, name)
        if torch.is_complex(t):
            t.data = t.data.real.clone()
    env_r.to(device)
    gen = build_gfn(cfg_r, env_r, device, ddp=False)
    ck = checkpoint or sorted(glob.glob(os.path.join(run_dir, 'checkpoints', 'checkpoint_*.pt')))[-1]
    gen.load(ck); gen.eval(); env_r.reward_fn.scale = 1.0
    return cfg_r, env_r, gen, ck


def sampler_frequencies(env_r, gen, keys, n_leaves, samples, batch=512):
    """topology frequencies of `samples` networks drawn from the current policy, aggregated over slot labellings by the
    canonical string.  A network outside the enumerated class is an error: the enumeration (tests V4 / V5) and the
    canonical string are then inconsistent with the sampler's environment, and the TV would be wrong."""
    rw = RolloutWorkerNet(env_r)
    index = {k: i for i, k in enumerate(keys)}
    counts = np.zeros(len(keys))
    done = 0
    while done < samples:
        with torch.no_grad():
            _, trajs = rw.rollout(gen, min(batch, samples - done), generate_full_trajectories=True)
        for t in trajs:
            k = network_canonical_string(t.current_state.network.root, n_leaves)
            if k not in index:
                raise RuntimeError('sampled network outside the enumerated class (%d graphs): %s' % (len(keys), k))
            counts[index[k]] += 1
        done += len(trajs)
    return counts / samples


def sampler_tv(ref, lambda_r, f):
    """(TV to the labelled target, TV of the 2^-R-reweighted frequencies to the graph posterior, mass by R)"""
    R = ref['R']
    pi_lab = posterior_over_graphs(ref['logp'], R, lambda_r, True)
    pi_graph = posterior_over_graphs(ref['logp'], R, lambda_r, False)
    f_graph = f * 2.0 ** (-R); f_graph /= f_graph.sum()
    return tv(f, pi_lab), tv(f_graph, pi_graph), [float(f[R == r].sum()) for r in range(int(R.max()) + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cfg'); ap.add_argument('sequences')
    ap.add_argument('--draws', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--run_dir', default=None); ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--samples', type=int, default=4096)
    ap.add_argument('--floor_reps', type=int, default=20)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=None)
    ap.add_argument('--save_reference', default=None); ap.add_argument('--load_reference', default=None)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    seqs = load_sequences(args.sequences)
    cfg, env = build_problem_env(args.cfg, seqs)
    lam = cfg.ENV.EVOLUTION_MODEL.PRIOR_LAMBDA
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    if args.load_reference:
        ref = load_reference(args.load_reference, env, lam, args.draws, args.seed)
        say('reference loaded from %s: %d graphs, %d draws per graph, relative SE median %.3f, max %.3f' % (
            args.load_reference, len(ref['keys']), ref['identity']['draws'], np.median(ref['relse']), ref['relse'].max()))
    else:
        ref = compute_reference(env, lam, args.draws, args.seed, say)
        if args.save_reference:
            save_reference(ref, args.save_reference)
            say('reference saved to %s' % args.save_reference)
    keys, R, logp = ref['keys'], ref['R'], ref['logp']
    pi_lab, pi_graph = posterior_over_graphs(logp, R, env.lambda_r, True), posterior_over_graphs(logp, R, env.lambda_r, False)
    for name, pi in (('labelled target (2^R, what the sampler targets)', pi_lab), ('stated prior over graphs', pi_graph)):
        order = np.argsort(-pi)
        say('posterior under the %s: mass by R %s; top-5 graphs %s' % (name, [round(float(pi[R == r].sum()), 4) for r in range(env.r_max + 1)],
            ', '.join('%.3f' % pi[i] for i in order[:5])))
    reference_floors(ref, env.lambda_r, args.samples, args.floor_reps, args.seed, say)
    # ---- sampler
    if args.run_dir:
        cfg_r, env_r, gen, ck = load_run_generator(args.run_dir, seqs, args.device, args.checkpoint)
        f = sampler_frequencies(env_r, gen, keys, env.n_leaves, args.samples)
        tv_lab, tv_graph, mass = sampler_tv(ref, env.lambda_r, f)
        say('sampler %s (%s): %d samples, all inside the enumerated class' % (args.run_dir, os.path.basename(ck), args.samples))
        say('   TV(sampler frequencies, labelled target) = %.4f   TV(2^-R-reweighted frequencies, graph posterior) = %.4f' % (tv_lab, tv_graph))
        say('   sampler mass by R %s' % [round(x, 4) for x in mass])
    if args.out:
        open(args.out, 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
