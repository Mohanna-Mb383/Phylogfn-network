"""
Validation metrics of the proposal (targets T4 - T6, and the topology-posterior comparison of T3).

  hardwired clusters   for every edge u -> v the set of leaves below v; the hardwired-cluster
                       distance between two networks on the same taxa is the size of the symmetric
                       difference of their cluster sets (Huson et al.; 0 = identical cluster sets).
  reticulation events  (hybrid clade, {sister set 1, sister set 2}) per reticulation; used to match
                       the reticulations of a sampled network with those of the truth and to compute
                       the MAE of θ (T6).
  posterior mode of R  (T4) from a list of networks or an R histogram.
  total variation      between two distributions over topology ids (T3).

Networks are given either as PhyloNetwork objects (root NetNode) or as truth files written by
tools/simulate_network_data.py ("root r", "edge p c d", "reticulation h parents p0 p1 theta").
"""
import numpy as np


# ------------------------------------------------------------------------------- generic graph view
class SimpleNet(object):
    """children: idx -> [(child idx, length)]; parents: idx -> [idx]; leaves 0..N-1; rets: idx -> (p0, p1, theta)"""

    def __init__(self, children, root, n_leaves, rets):
        self.children = children
        self.root = root
        self.n_leaves = n_leaves
        self.rets = rets

    @staticmethod
    def from_phylo_network(net):
        from src.env.phylo_network_env import all_nodes
        children, rets = {}, {}
        for v in all_nodes(net.root):
            children[v.idx] = [(c.idx, d) for c, d in zip(v.children, v.child_dists)]
            if v.kind == 'ret':
                rets[v.idx] = (v.parents[0].idx, v.parents[1].idx, v.theta)
        return SimpleNet(children, net.root.idx, net.n_leaves, rets)

    @staticmethod
    def from_truth_file(path):
        children, rets, root = {}, {}, None
        for line in open(path):
            t = line.split()
            if not t:
                continue
            if t[0] == 'root':
                root = int(t[1])
            elif t[0] == 'edge':
                children.setdefault(int(t[1]), []).append((int(t[2]), float(t[3])))
            elif t[0] == 'reticulation':
                rets[int(t[1])] = (int(t[3]), int(t[4]), float(t[6]))   # 'reticulation h parents p0 p1 theta x'
        nodes = set(children) | {c for lst in children.values() for c, _ in lst}
        n_leaves = sum(1 for v in nodes if v not in children)
        return SimpleNet(children, root, n_leaves, rets)

    def leaves_below(self):
        memo = {}

        def rec(v):
            if v in memo:
                return memo[v]
            if not self.children.get(v):                 # a leaf: no children (absent key or empty list)
                memo[v] = frozenset([v])
            else:
                memo[v] = frozenset().union(*(rec(c) for c, _ in self.children[v]))
            return memo[v]
        for v in list(self.children) + list(range(self.n_leaves)):
            rec(v)
        return memo


# ------------------------------------------------------------------------------- clusters
def hardwired_clusters(snet):
    below = snet.leaves_below()
    clusters = set()
    for p, lst in snet.children.items():
        for c, _ in lst:
            s = below[c]
            if 1 < len(s) < snet.n_leaves:          # trivial clusters (single leaves, all leaves) carry no information
                clusters.add(s)
    return clusters


def hardwired_cluster_distance(a, b):
    ca, cb = hardwired_clusters(a), hardwired_clusters(b)
    return len(ca ^ cb)


# ------------------------------------------------------------------------------- reticulation events and theta
def reticulation_events(snet):
    """per reticulation: hybrid clade (leaves below h), the sister set at each parent (leaves below the
    parent's other child, without the hybrid clade), theta (inheritance of parent 0)"""
    below = snet.leaves_below()
    events = []
    for h, (p0, p1, theta) in snet.rets.items():
        hyb = below[h]
        sisters = []
        for p in (p0, p1):
            other = [c for c, _ in snet.children[p] if c != h]
            sisters.append(frozenset().union(*(below[c] for c in other)) - hyb if other else frozenset())
        events.append({'hybrid': hyb, 'sister0': sisters[0], 'sister1': sisters[1], 'theta': theta})
    return events


def theta_mae(sample_net, truth_net):
    """T6: mean absolute error of θ over the truth's reticulations, matched by hybrid clade.  θ is oriented
    by the sister set of the truth's parent 0 (if neither orientation of the sample matches, the better
    one is used); a truth reticulation without a match in the sample counts as an error of 1."""
    ev_s = reticulation_events(sample_net)
    ev_t = reticulation_events(truth_net)
    if not ev_t:
        return 0.0 if not ev_s else 1.0
    errs = []
    for t in ev_t:
        match = [s for s in ev_s if s['hybrid'] == t['hybrid']]
        if not match:
            errs.append(1.0)
            continue
        s = match[0]
        if s['sister0'] == t['sister0']:
            th = s['theta']
        elif s['sister1'] == t['sister0']:
            th = 1.0 - s['theta']
        else:
            th = min((s['theta'], 1.0 - s['theta']), key=lambda x: abs(x - t['theta']))
        errs.append(abs(th - t['theta']))
    return float(np.mean(errs))


# ------------------------------------------------------------------------------- posterior over R, TV
def posterior_mode_R(counts):
    """counts: list of R values or an R histogram"""
    counts = np.asarray(counts)
    if counts.ndim == 1 and counts.dtype.kind in 'iu' and len(counts) <= 16 and counts.sum() > len(counts):
        return int(np.argmax(counts))
    return int(np.bincount(counts.astype(int)).argmax())


def total_variation(p, q):
    """p, q: dict topology id -> probability"""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
