"""
Reference (slow, explicit) likelihood of a level-1 network under the displayed-tree
model of Jin, Nakhleh, Snir & Tuller (2006):

    P(Y | N, b, theta) = prod_i  sum_{T in T(N)}  w(T | theta) P(Y_i | T, b)

with  w(T | theta) = prod_h theta_h^[T keeps slot 0 of h] (1 - theta_h)^[T keeps slot 1 of h].

It enumerates the 2^R displayed trees explicitly and runs Felsenstein pruning on each.
It exists ONLY to test the fast "mix at the top of the gall" computation of
PhyloNetworkEnv (they must agree to floating-point precision).
"""
import itertools
import numpy as np
from src.env.phylo_network_env import all_nodes, reticulation_nodes


def _jc_matrix(b):
    """Jukes-Cantor transition matrix for branch length b (same convention as decompJC)."""
    p = 0.25 * (1.0 - np.exp(-4.0 * b / 3.0))
    M = np.full((4, 4), p)
    np.fill_diagonal(M, 1.0 - 3.0 * p)
    return M


def displayed_trees(root):
    """yield (weight, children_dict, tree_root) for every displayed tree of the network.

    children_dict maps node idx -> list of (child idx, branch length) after deleting one
    parent edge per reticulation and suppressing degree-2 vertices (lengths are summed).
    """
    nodes = all_nodes(root)
    rets = reticulation_nodes(root)
    for choice in itertools.product([0, 1], repeat=len(rets)):
        weight = 1.0
        # build the raw graph: parent -> [(child, dist)]
        ch = {v.idx: [] for v in nodes}
        for v in nodes:
            if v.kind == 'ret':
                continue
            for c, d in zip(v.children, v.child_dists):
                ch[v.idx].append([c.idx, d])
        for h, k in zip(rets, choice):
            weight *= h.theta if k == 0 else (1.0 - h.theta)
            keep = h.parents[k]
            drop = h.parents[1 - k]
            # delete the edge drop -> h
            ch[drop.idx] = [e for e in ch[drop.idx] if e[0] != h.idx]
            # h keeps its single child
            ch[h.idx] = [[h.children[0].idx, h.child_dist]]
        # suppress every vertex with exactly one child (h itself and each `drop`),
        # summing lengths; the root may also lose a child -> new root is its only child
        parent_of = {}
        for p, lst in ch.items():
            for c, d in lst:
                parent_of[c] = p
        changed = True
        tree_root = root.idx
        while changed:
            changed = False
            for v in list(ch.keys()):
                if len(ch[v]) == 1:
                    (c, d) = ch[v][0]
                    if v == tree_root:
                        tree_root = c
                        del ch[v]
                        parent_of.pop(c, None)
                    else:
                        p = parent_of[v]
                        ch[p] = [[cc, dd + d] if cc == v else [cc, dd] for cc, dd in ch[p]]
                        ch[p] = [[c if cc == v else cc, dd] for cc, dd in ch[p]]
                        parent_of[c] = p
                        del ch[v]
                    changed = True
                    break
        yield weight, ch, tree_root


def tree_site_likelihoods(ch, tree_root, seq_arrays):
    """Felsenstein pruning; returns per-site likelihoods P(Y_i | T, b) (uniform root)."""
    n_leaves, m, c = seq_arrays.shape

    def rec(v):
        if v not in ch or len(ch[v]) == 0:
            return seq_arrays[v]                         # leaf: one-hot [m, 4]
        out = np.ones((m, c))
        for child, d in ch[v]:
            out = out * (rec(child) @ _jc_matrix(d))
        return out
    L = rec(tree_root)
    return (L / 4.0).sum(-1)                              # [m]


def network_log_likelihood_explicit(root, seq_arrays):
    """log prod_i sum_T w(T) P(Y_i | T)"""
    site = None
    for w, ch, tree_root in displayed_trees(root):
        s = w * tree_site_likelihoods(ch, tree_root, seq_arrays)
        site = s if site is None else site + s
    return float(np.sum(np.log(site)))


def network_log_prior_edges(root, prior_lambda):
    """exponential prior on every branch length, PhyloGFN root convention
    (the two root edges count as ONE edge of length b_l + b_r)."""
    logp = 0.0
    for v in all_nodes(root):
        if v is root:
            total = sum(v.child_dists)
            logp += np.log(prior_lambda) - prior_lambda * total
        else:
            for d in v.child_dists:
                logp += np.log(prior_lambda) - prior_lambda * d
    return float(logp)


def network_log_score_explicit(net, env):
    """the quantity the environment stores as `log_score` for a terminal network"""
    seq = env.seq_arrays.detach().cpu().numpy()
    ll = network_log_likelihood_explicit(net.root, seq)
    lp = network_log_prior_edges(net.root, env.evolution_model.prior_lambda)
    return ll + lp + env.log_topology_prior_unnormalised(net.num_reticulations())
