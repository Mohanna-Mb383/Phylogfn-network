"""
Explicit topology encoding of a batch of partial networks for the graph encoder
(proposal: "state = the partial network built so far -- its nodes, its edges, and the list of
lineages still open").

For every state the nodes reachable from its open lineages form a forest of partial networks.
Each node gets a feature row
    [leaf, tree, ret] one-hot | is the root of an open lineage | root lineage carries an open cycle
    | #leaves below / N | height / N | theta (ret nodes) | mean child branch length | log(1 + #children)
and the parent -> child adjacency.  A bare slot lineage (a reticulation waiting for a parent)
is mapped to the reticulation node itself.  Output tensors are padded to V_MAX nodes.
"""
import math
import numpy as np
import torch
from src.env.phylo_network_env import all_nodes

NODE_FEATURES = 10


def graph_batch(states, n_leaves, v_max, l_max, device):
    B = len(states)
    X = np.zeros((B, v_max, NODE_FEATURES), dtype=np.float32)
    A = np.zeros((B, v_max, v_max), dtype=np.float32)          # A[b, parent, child] = 1
    node_mask = np.zeros((B, v_max), dtype=bool)
    lineage_node = np.zeros((B, l_max), dtype=np.int64)         # position of each lineage's node
    for b, s in enumerate(states):
        nodes, seen = [], {}
        for l in s.lineages:
            for v in all_nodes(l.node):
                if v.idx not in seen:
                    seen[v.idx] = len(nodes)
                    nodes.append(v)
        assert len(nodes) <= v_max, (len(nodes), v_max)
        # leaves below / height (memoised over the DAG)
        below, height = {}, {}

        def rec(v):
            if v.idx in below:
                return
            if v.kind == 'leaf':
                below[v.idx], height[v.idx] = 1, 0
                return
            for c in v.children:
                rec(c)
            below[v.idx] = sum(below[c.idx] for c in v.children)
            height[v.idx] = 1 + max(height[c.idx] for c in v.children)
        for v in nodes:
            rec(v)
        open_roots = {l.node.idx: (l.open_h is not None) for l in s.lineages}
        for v in nodes:
            p = seen[v.idx]
            node_mask[b, p] = True
            X[b, p, {'leaf': 0, 'tree': 1, 'ret': 2}[v.kind]] = 1.0
            if v.idx in open_roots:
                X[b, p, 3] = 1.0
                X[b, p, 4] = 1.0 if open_roots[v.idx] else 0.0
            X[b, p, 5] = below[v.idx] / n_leaves
            X[b, p, 6] = height[v.idx] / n_leaves
            X[b, p, 7] = v.theta if v.kind == 'ret' else 0.0
            X[b, p, 8] = float(np.mean(v.child_dists)) if v.child_dists else 0.0
            X[b, p, 9] = math.log1p(len(v.children))
            for c in v.children:
                A[b, p, seen[c.idx]] = 1.0
        for i, l in enumerate(s.lineages):
            lineage_node[b, i] = seen[l.node.idx]
    return {
        'graph_x': torch.tensor(X, device=device),
        'graph_adj': torch.tensor(A, device=device),
        'graph_node_mask': torch.tensor(node_mask, device=device),
        'graph_lineage_node': torch.tensor(lineage_node, device=device),
    }
