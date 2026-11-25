"""
Simulate an alignment from a (G, b, θ) network object under Jukes-Cantor, and draw (G, b, θ) from the
model's own prior -- the two ingredients of simulation-based calibration (proposal target T7).

Displayed-tree model: every site independently keeps, at each reticulation h, the parent-0 edge with
probability θ_h (else the parent-1 edge) and evolves down the resulting displayed tree.
"""
import itertools
import math
import numpy as np

from src.env.phylo_network_env import all_nodes, reticulation_nodes
from src.utils.network_likelihood import displayed_trees

_BASES = 'ACGT'


def _jc_step(states, b, rng):
    """states [n] in 0..3 -> child states after a branch of length b"""
    p_change = 0.75 * (1.0 - math.exp(-4.0 * b / 3.0))
    change = rng.random(states.shape[0]) < p_change
    new = states.copy()
    k = change.sum()
    if k:
        new[change] = (states[change] + rng.integers(1, 4, size=k)) % 4      # uniform over the 3 other bases
    return new


def simulate_sites_from_network(root, m, rng):
    """returns [N, m] integer states (0..3) for the leaves 0..N-1"""
    rets = reticulation_nodes(root)
    R = len(rets)
    n_leaves = sum(1 for v in all_nodes(root) if v.kind == 'leaf')
    out = np.zeros((n_leaves, m), dtype=int)
    # per-site displayed tree = the choice vector at the reticulations
    choices = np.stack([(rng.random(m) >= h.theta).astype(int) for h in rets], axis=1) if R else np.zeros((m, 0), dtype=int)
    trees = {choice: (ch, tr) for choice, (_, ch, tr) in zip(itertools.product([0, 1], repeat=R), displayed_trees(root))} if R \
        else {(): displayed_trees(root).__next__()[1:]}
    for choice, (ch, tr) in trees.items():
        sites = np.where((choices == np.array(choice, dtype=int)).all(1))[0] if R else np.arange(m)
        if len(sites) == 0:
            continue
        root_states = rng.integers(0, 4, size=len(sites))

        def rec(v, states):
            if v not in ch or len(ch[v]) == 0:
                out[v, sites] = states
                return
            for c, d in ch[v]:
                rec(c, _jc_step(states, d, rng))
        rec(tr, root_states)
    return out


def states_to_strings(states):
    return [''.join(_BASES[x] for x in row) for row in states]


def draw_network_from_prior(env, class_nets, rng):
    """(G, b, θ) ~ p(G) p(b) p(θ): G from the model's topology prior over the enumerated class
    (class_nets: topology id -> PhyloNetwork), b ~ Exp(PRIOR_LAMBDA) per edge (root: one length split in
    halves), θ ~ U(0, 1).  Returns a deep copy with the drawn continuous values set."""
    import copy
    tids = list(class_nets)
    logw = np.array([env.log_topology_prior_unnormalised(class_nets[t].num_reticulations()) for t in tids])
    w = np.exp(logw - logw.max())
    w /= w.sum()
    net = copy.deepcopy(class_nets[tids[rng.choice(len(tids), p=w)]])
    lam = env.evolution_model.prior_lambda
    for v in all_nodes(net.root):
        if v is net.root and len(v.children) == 2:
            b = rng.exponential(1.0 / lam)
            v.child_dists = [b / 2, b / 2]
        else:
            v.child_dists = [float(rng.exponential(1.0 / lam)) for _ in v.children]
        for c, d in zip(v.children, v.child_dists):
            if c.kind == 'ret':
                slot = c.parents.index(v)
                c.parent_dists[slot] = d
        if v.kind == 'ret':
            v.theta = float(rng.uniform(0.0, 1.0))
            v.child_dist = v.child_dists[0]
    net.log_score = None
    return net
