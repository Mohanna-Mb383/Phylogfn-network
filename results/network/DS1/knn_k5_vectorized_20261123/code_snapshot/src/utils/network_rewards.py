"""
Alternative rewards of the proposal (Reward slide, risks R3 / R4):

  softwired parsimony    min over the displayed trees of the Fitch parsimony score
                         (the standard "softwired" network parsimony); reward = -score,
                         the network counterpart of PhyloGFN's parsimony mode.
  quartet pseudo-likelihood
                         PL(Y | N) = (C(N,4) / Q) * sum_{q in Q sampled quartets} log P(Y_q | N|q),
                         where N|q is the network restricted to the four taxa of q and
                         P(Y_q | N|q) = sum_T w(T | theta) P(Y_q | T|q) is the displayed-tree mixture of
                         the restriction (weights inherited from the displayed trees of N).
                         Cost O(Q * 2^R * S * c^2), independent of N -- the proposal's training reward.

Both work on the terminal PhyloNetwork object (root NetNode) and the one-hot sequence arrays
[N, m, 4] (gaps = all-ones rows, treated as missing data).
"""
import itertools
import numpy as np
from src.utils.network_likelihood import displayed_trees, tree_site_likelihoods


# ------------------------------------------------------------------------------- Fitch on one displayed tree
def fitch_score(ch, tree_root, seq_arrays):
    """Fitch parsimony of a rooted tree given as ch: node -> [(child, dist)], leaves = seq rows."""
    def rec(v):
        if v not in ch or len(ch[v]) == 0:
            return seq_arrays[v] > 0, 0                      # [m, 4] boolean state sets
        sets, cost = [], 0
        for c, _ in ch[v]:
            s, k = rec(c)
            sets.append(s)
            cost += k
        inter = sets[0]
        union = sets[0]
        for s in sets[1:]:
            inter = inter & s
            union = union | s
        empty = ~inter.any(-1)                                  # sites where the intersection is empty
        out = np.where(empty[:, None], union, inter)
        return out, cost + int(empty.sum())
    _, score = rec(tree_root)
    return score


def softwired_parsimony(root, seq_arrays):
    return min(fitch_score(ch, tr, seq_arrays) for _, ch, tr in displayed_trees(root))


# ------------------------------------------------------------------------------- restriction of a tree to a taxon set
def restrict_tree(ch, tree_root, keep):
    """ch: node -> [(child, dist)]; returns (ch_q, root_q) of the tree induced on the leaves in `keep`
    (other leaves removed, degree-1 internal nodes pruned, degree-2 nodes suppressed with lengths summed)."""
    def rec(v):
        # returns list of (node, dist_from_v) surviving children after pruning, or None if the subtree is empty
        if v not in ch or len(ch[v]) == 0:
            return v if v in keep else None
        kept = []
        for c, d in ch[v]:
            sub = rec(c)
            if sub is None:
                continue
            if isinstance(sub, tuple):                 # suppressed chain: (node, extra length)
                kept.append((sub[0], d + sub[1]))
            else:
                kept.append((sub, d))
        if not kept:
            return None
        if len(kept) == 1:
            return (kept[0][0], kept[0][1])            # v is suppressed; pass the length up
        new_ch[v] = kept
        return v
    new_ch = {}
    r = rec(tree_root)
    if r is None:
        return {}, None
    if isinstance(r, tuple):                            # the root was suppressed: its single child is the new root
        return new_ch, r[0]
    return new_ch, r


def quartet_pseudo_loglik(root, seq_arrays, Q, seed=0):
    """(C(N,4)/Q) * sum over Q random quartets of log sum_T w(T) P(Y_q | T|q)"""
    n = seq_arrays.shape[0]
    all_q = list(itertools.combinations(range(n), 4))
    rng = np.random.default_rng(None if seed < 0 else seed)
    if Q >= len(all_q):
        quartets, scale = all_q, 1.0
    else:
        idx = rng.choice(len(all_q), size=Q, replace=False)
        quartets, scale = [all_q[i] for i in idx], len(all_q) / Q
    trees = list(displayed_trees(root))
    total = 0.0
    for q in quartets:
        keep = set(q)
        site = None
        for w, ch, tr in trees:
            ch_q, r_q = restrict_tree(ch, tr, keep)
            s = w * tree_site_likelihoods(ch_q, r_q, seq_arrays)
            site = s if site is None else site + s
        total += float(np.sum(np.log(site)))
    return scale * total
