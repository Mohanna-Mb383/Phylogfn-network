"""
Exact reference posterior over level-1 network topologies for small N (proposal target T3).

For every topology G of the class (enumerated with the sampler's own MDP, so the class is the
same), the marginal likelihood
      p(Y | G) = ∫ P(Y | G, b, θ) p(b) p(θ) db dθ
is estimated by importance sampling from the prior (branch lengths ~ Exp(PRIOR_LAMBDA) per edge,
the two root edges counted as ONE edge as in PhyloGFN's reward, θ ~ U(0, 1)), with S draws per
topology and a Monte-Carlo standard error.  The posterior is p(G | Y) ∝ p(G) p(Y | G) with the
model's own topology prior (EXP_R / UNIFORM_R, λ, w).  Everything is vectorised over the S draws:
one displayed-tree structure per topology, batched Felsenstein pruning with [S, 4, 4] transition
matrices.
"""
import copy
import itertools
import math
import numpy as np

from src.env.phylo_network_env import all_nodes, reticulation_nodes, network_canonical_string


# ------------------------------------------------------------------------------- symbolic displayed trees
def edge_table(root):
    """edges of the network with ids: list of (parent idx, child idx); the two root edges share one id
    (PhyloGFN root convention: a single root length split in halves)."""
    edges, eid = [], {}
    for v in all_nodes(root):
        for c in v.children:
            edges.append((v.idx, c.idx))
    for k, (p, c) in enumerate(edges):
        eid[(p, c)] = k
    root_edges = [(p, c) for (p, c) in edges if p == root.idx]
    if len(root_edges) == 2:
        eid[root_edges[1]] = eid[root_edges[0]]        # same id -> same length b_root/2 each (see below)
    return edges, eid, len(edges)                      # one (unused) column is left for the merged root edge


def displayed_tree_structures(root):
    """[(theta_choice tuple, ch)] with ch: node -> [(child, [edge ids whose lengths add up])]"""
    nodes = all_nodes(root)
    rets = reticulation_nodes(root)
    edges, eid, _ = edge_table(root)
    out = []
    for choice in itertools.product([0, 1], repeat=len(rets)):
        ch = {v.idx: [] for v in nodes}
        for v in nodes:
            if v.kind == 'ret':
                continue
            for c in v.children:
                ch[v.idx].append([c.idx, [eid[(v.idx, c.idx)]]])
        for h, k in zip(rets, choice):
            drop = h.parents[1 - k]
            ch[drop.idx] = [e for e in ch[drop.idx] if e[0] != h.idx]
            ch[h.idx] = [[h.children[0].idx, [eid[(h.idx, h.children[0].idx)]]]]
        parent_of = {c: p for p, lst in ch.items() for c, _ in lst}
        tree_root, changed = root.idx, True
        while changed:
            changed = False
            for v in list(ch.keys()):
                if len(ch[v]) == 1:
                    (c, ids) = ch[v][0]
                    if v == tree_root:
                        tree_root = c
                        del ch[v]
                        parent_of.pop(c, None)
                    else:
                        p = parent_of[v]
                        ch[p] = [[c if cc == v else cc, (dd + ids) if cc == v else dd] for cc, dd in ch[p]]
                        parent_of[c] = p
                        del ch[v]
                    changed = True
                    break
        out.append((choice, ch, tree_root))
    return rets, out


# ------------------------------------------------------------------------------- batched likelihood
def _jc_batch(b):
    """b [S] -> [S, 4, 4]"""
    p = 0.25 * (1.0 - np.exp(-4.0 * b / 3.0))
    M = np.repeat(p[:, None, None], 4, 1).repeat(4, 2)
    idx = np.arange(4)
    M[:, idx, idx] = (1.0 - 3.0 * p)[:, None]
    return M


def batched_log_likelihood(root, seq_arrays, B, theta, root_edge_ids):
    """B [S, n_ids] edge lengths (the shared root id holds b_root, used as b_root/2 on each root edge),
    theta [S, R] -> log P(Y | G, b_s, θ_s) [S]"""
    rets, structs = displayed_tree_structures(root)
    S = B.shape[0]
    m = seq_arrays.shape[1]
    site = np.zeros((S, m))
    for choice, ch, tree_root in structs:
        w = np.ones(S)
        for r, k in enumerate(choice):
            w = w * (theta[:, r] if k == 0 else 1.0 - theta[:, r])

        def rec(v):
            if v not in ch or len(ch[v]) == 0:
                return np.broadcast_to(seq_arrays[v][None], (S, m, 4))
            out = np.ones((S, m, 4))
            for c, ids in ch[v]:
                b = np.zeros(S)
                for e in ids:
                    b = b + (0.5 * B[:, e] if e in root_edge_ids else B[:, e])
                out = out * np.einsum('smv,svc->smc', rec(c), _jc_batch(b))
            return out
        L = rec(tree_root)
        site += w[:, None] * (L / 4.0).sum(-1)
    return np.sum(np.log(site), axis=1)


def marginal_loglik_is(root, seq_arrays, S, prior_lambda, rng):
    """log p̂(Y | G) by importance sampling from the prior, and its Monte-Carlo standard error (log scale)"""
    edges, eid, n_ids = edge_table(root)
    root_ids = {eid[(p, c)] for (p, c) in edges if p == root.idx}
    R = len(reticulation_nodes(root))
    B = rng.exponential(1.0 / prior_lambda, size=(S, n_ids))
    theta = rng.uniform(0.0, 1.0, size=(S, max(R, 1)))
    ll = batched_log_likelihood(root, seq_arrays, B, theta, root_ids)
    mx = ll.max()
    w = np.exp(ll - mx)
    mean = w.mean()
    se = w.std(ddof=1) / (mean * math.sqrt(S)) if S > 1 else float('nan')
    return float(mx + math.log(mean)), float(se)


# ------------------------------------------------------------------------------- enumeration
def enumerate_class(env):
    """one PhyloNetwork per topology of the class reachable by the MDP (continuous values random)"""
    from tests.test_network_env import state_key                     # canonical partial-state key
    seen, nets, stack = set(), {}, [(env.get_initial_state(), [])]
    while stack:
        s, actions = stack.pop()
        k = state_key(s)
        if k in seen:
            continue
        seen.add(k)
        if s.is_done:
            tid = network_canonical_string(s.lineages[0].node, env.n_leaves)
            if tid not in nets:
                # the search shares NetNode objects between branches (reticulation parent slots get
                # overwritten): rebuild this topology from its own action list to get private objects
                nets[tid] = env.actions_to_trajectory(actions).current_state.network
            continue
        for a in np.where(env.legal_actions(s))[0]:
            action = env.action_from_id(int(a), s, random_params=True)
            ns, _, _ = env.apply_action(s, action, None)
            stack.append((ns, actions + [action]))
    return nets


def exact_posterior(env, S=500, seed=0, verbose=True):
    """returns dict topology_id -> {'log_ml', 'se', 'R', 'log_post'} and the R posterior"""
    rng = np.random.default_rng(seed)
    seq = env.seq_arrays.detach().cpu().numpy()
    nets = enumerate_class(env)
    res = {}
    for i, (tid, net) in enumerate(nets.items()):
        log_ml, se = marginal_loglik_is(net.root, seq, S, env.evolution_model.prior_lambda, rng)
        R = net.num_reticulations()
        res[tid] = {'log_ml': log_ml, 'se': se, 'R': R, 'log_prior': env.log_topology_prior_unnormalised(R)}
        if verbose and (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(nets)} topologies')
    lp = np.array([v['log_ml'] + v['log_prior'] for v in res.values()])
    lz = np.logaddexp.reduce(lp)
    for v, l in zip(res.values(), lp):
        v['log_post'] = float(l - lz)
    post_R = {}
    for v in res.values():
        post_R[v['R']] = post_R.get(v['R'], 0.0) + math.exp(v['log_post'])
    return res, post_R
