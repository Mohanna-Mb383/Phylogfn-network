"""
PhyloGFN-Net: a level-1 phylogenetic NETWORK environment for PhyloGFN (version 3).

The MDP (unchanged from version 1; see lecture Part IV)
--------------------------------------------------------
state  = sorted list of open *lineages* (roots of partial networks) + bookkeeping.
actions
  MERGE(i, j)      join lineages i and j under a new tree node; sample two branch lengths;
  RETICULATE(i)    put a reticulation node h above lineage i; h gets two bare parent slots
                   p0, p1 (two new open lineages); sample theta_h and the length of h -> i.
terminal           one lineage left, carrying no open reticulation.

Changes in version 3 (every one is tested in tests/):
  1. Nodes are never mutated after creation.  A tree node w records in `child_slots` which
     slot of a reticulation child it fills; the reticulation's `parents` are derived from
     that when a terminal PhyloNetwork is built (a deep copy of the node graph).  Hence two
     actions applied to the same state no longer corrupt each other (v1 wrote
     `h.parents[slot] = w` into a node shared by both states).
  2. Features live in a fixed "slot" tensor [B, l_max, 2, m, c]; a lineage remembers its
     row (`Lineage.row`).  MERGE writes the merged feature into u's row and frees v's row;
     RETICULATE reuses x's row for slot 0 and takes a free row for slot 1.  Nothing is ever
     shifted or cloned, and the updates of a whole batch are done by at most 9 batched
     kernels per step (grouped by action type, merge case and last-step flag) instead of a
     Python loop with ~15 small kernels per trajectory.
  3. Pairs are enumerated column-major, k(i, j) = j (j - 1) / 2 + i, so the pairs among the
     first n lineages are exactly the first n (n - 1) / 2 action ids.  The policy only scores
     the C(n_max, 2) + n_max actions of the largest active state instead of
     C(l_max, 2) + l_max (DS7: 1830 pairs -> 435 on average).
  4. Cycle masks (ENV.NETWORK.CYCLE_MASK):
       NONE        the v1 masks, bit for bit;
       TRIANGLES   no 3-cycles;
       DEGENERATE  no 3-cycles and no 4-cycle closed at the root.
     Under the displayed-tree likelihood a cycle displays two trees; the two have the SAME
     unrooted topology exactly when the cycle has at most three pendant groups, i.e. it is
     a triangle or a root 4-cycle (theorem in the dossier, verified exhaustively for N <= 6).
     Such a cycle adds only a branch-length mixture and is not identifiable.
     Masking alone creates dead ends, so every candidate action is also checked for
     feasibility: an exact memoised search over the abstract state (number of free lineages,
     the cycle-size class of every open gall, reticulations left) decides whether a terminal
     network of the class is still reachable.  Reachable states therefore never dead-end
     (tests enumerate all of them for N <= 6).
  5. The reward divides by 2^R: every network with R reticulations is produced as 2^R
     distinct terminal objects (which parent fills slot 0 of each reticulation, with
     theta <-> 1 - theta).  Without the division the sampler targets p(G | Y) 2^R(G).

Likelihood without the 2^R blow-up ("mixing at the top of each gall") -- unchanged
------------------------------------------------------------------------------------
A lineage with an open reticulation h keeps TWO Felsenstein feature tensors:
  A : the displayed trees in which THIS side keeps its edge into h  (weight theta or 1-theta)
  B : the displayed trees in which this side's edge into h is deleted
      (for a bare slot B is the all-ones vector, "no data below").
At the closing merge the exact displayed-tree likelihood of the cycle is A_u (x) B_v + B_u (x) A_v
per site, and the result is a single feature again (Proposition 1 of the lecture; verified
against the explicit sum over the 2^R displayed trees to 1e-11).
"""
import math
import random
import copy
from functools import lru_cache
import numpy as np
import torch
from torch import nn

from src.env.trajectory import Trajectory, SimpleTrajectory
from src.utils.evolution_model_torch import EvolutionModelTorch
from src.env.binary_tree_env_one_step_likelihood import CHARACTERS_MAPS, PhyloTreeReward
from src.utils.level1_counts import log_class_count, CYCLE_MASKS


# ----------------------------------------------------------------------------------
#  Network data structures (replace ete3.TreeNode, which cannot hold two parents)
# ----------------------------------------------------------------------------------
class NetNode(object):
    """A node of a (partial) rooted phylogenetic network.  IMMUTABLE after the action that
    creates it (its own children lists are filled by that action only).

    kind: 'leaf' | 'tree' | 'ret'
      leaf : no children, one parent
      tree : two children (child_dists), one parent; child_slots[c] = slot of child c filled
             by this node if child c is a reticulation, else None
      ret  : one child (child_dist), two parents = two SLOTS; theta is the inheritance
             weight of slot 0, (1 - theta) of slot 1.  `parents` / `parent_dists` are only
             filled on the deep copy owned by a terminal PhyloNetwork (see build_network).
    """
    __slots__ = ('idx', 'kind', 'children', 'child_dists', 'child_slots', 'parents', 'parent_dists',
                 'theta', 'child_dist', 'ret_id', 'child_tokens', 'tokens')

    def __init__(self, idx, kind):
        self.idx = idx
        self.kind = kind
        self.children = []
        self.child_dists = []
        self.child_slots = []
        self.parents = [None, None]
        self.parent_dists = [None, None]
        self.theta = None
        self.child_dist = None
        self.ret_id = None
        self.child_tokens = None       # ret only: ordering tokens of the child lineage (slot 0's side owns them)

    def leaves(self):
        return {v.idx for v in all_nodes(self) if v.kind == 'leaf'}


class Lineage(object):
    """An entry of the state list.

      * a real lineage: `node` is a NetNode; `slot is None`;
      * a BARE PARENT SLOT of a reticulation h: `node is h`, `slot in {0, 1}`.

    open_h : the reticulation this lineage carries unclosed (None if closed).
    depth  : number of tree nodes between this lineage's root (inclusive) and open_h
             (exclusive) on this side of the future cycle; 0 for a bare slot.  The cycle
             closed by MERGE(u, v) has depth(u) + depth(v) internal nodes.
    tokens : ordering tokens (leaf indices + one fresh token N + ret_id for slot 1 of every
             reticulation).  Tokens partition a set, so keys = min(tokens) are unique and the
             sorted order is well defined (this lets the backward sampler recompute action
             indices, as PhyloGFN's min_seq_idx does).
    row    : row of this lineage in the feature slot tensor.
    """
    __slots__ = ('node', 'slot', 'open_h', 'tokens', 'key', 'depth', 'row')

    def __init__(self, node, slot=None, open_h=None, tokens=None, depth=0, row=None):
        self.node = node
        self.slot = slot
        self.open_h = open_h
        self.tokens = frozenset(tokens) if tokens is not None else frozenset()
        self.key = min(self.tokens)
        self.depth = depth
        self.row = row

    @property
    def is_bare(self):
        return self.slot is not None

    @property
    def is_leaf(self):
        return (not self.is_bare) and self.node.kind == 'leaf'


class NetworkState(object):
    """A GFlowNet state: sorted list of lineages + bookkeeping.  Immutable."""
    __slots__ = ('lineages', 'next_idx', 'n_ret', 'free_rows', 'log_reward', 'log_score', 'num_lineages', 'is_done', 'network')

    def __init__(self, lineages, next_idx, n_ret, free_rows, log_reward=None, log_score=None):
        self.lineages = lineages
        self.next_idx = next_idx
        self.n_ret = n_ret
        self.free_rows = free_rows            # tuple of unused feature rows
        self.log_reward = log_reward
        self.log_score = log_score
        self.num_lineages = len(lineages)
        self.is_done = (len(lineages) == 1 and lineages[0].open_h is None)
        self.network = None                   # terminal PhyloNetwork, attached by the rollout

    def galls(self):
        """{id(h): [positions of its two carriers]} for the open reticulations"""
        g = {}
        for i, l in enumerate(self.lineages):
            if l.open_h is not None:
                g.setdefault(id(l.open_h), []).append(i)
        return g

    def num_parents(self, cycle_mask='NONE'):
        """|Pa(s)| under the uniform backward policy:
             (#lineages whose root is a tree node: un-merge)
           + (#reticulations whose two slots are both bare: un-reticulate).
        CYCLE_MASK NONE: un-reticulating h is a parent only if RETICULATE was legal from
        it, i.e. the state has a free lineage other than h's child (the v1 no-dead-end rule).
        Other masks: RETICULATE is legal iff the result is feasible, and every reachable
        state is feasible, so every both-bare reticulation is a parent."""
        n = 0
        seen_ret = {}
        n_free = 0
        for l in self.lineages:
            if l.is_bare:
                seen_ret[l.node.idx] = seen_ret.get(l.node.idx, 0) + 1
            else:
                if l.open_h is None:
                    n_free += 1
                if l.node.kind == 'tree':
                    n += 1
        if cycle_mask != 'NONE' or n_free >= 1:
            n += sum(1 for c in seen_ret.values() if c == 2)
        return n


# ----------------------------------------------------------------------------------
#  Terminal networks
# ----------------------------------------------------------------------------------
def all_nodes(root):
    out, seen, stack = [], set(), [root]
    while stack:
        v = stack.pop()
        if v.idx in seen:
            continue
        seen.add(v.idx)
        out.append(v)
        stack.extend(v.children)
    return out


def reticulation_nodes(root):
    return [v for v in all_nodes(root) if v.kind == 'ret']


def build_network(root):
    """Deep copy of the node graph below `root` with `parents` / `parent_dists` of every
    reticulation filled in from the parents' child_slots.  The MDP nodes stay untouched."""
    nodes = all_nodes(root)
    new = {}
    for v in nodes:
        w = NetNode(v.idx, v.kind)
        w.theta, w.child_dist, w.ret_id = v.theta, v.child_dist, v.ret_id
        w.child_dists = list(v.child_dists)
        w.child_slots = list(v.child_slots)
        new[v.idx] = w
    for v in nodes:
        w = new[v.idx]
        w.children = [new[c.idx] for c in v.children]
        for c, d, s in zip(w.children, w.child_dists, w.child_slots):
            if s is not None:
                assert c.kind == 'ret' and c.parents[s] is None
                c.parents[s] = w
                c.parent_dists[s] = d
    for v in new.values():
        if v.kind == 'ret':
            assert v.parents[0] is not None and v.parents[1] is not None, 'unfilled slot in a terminal network'
    return new[root.idx]


def cycle_geometry(net_root):
    """For every reticulation h of a terminal network: (h, a, b, top_is_root) where a and b
    are the numbers of tree nodes strictly between the top node of the cycle and h on the
    two sides (cycle length = a + b + 2)."""
    parent = {}
    for v in all_nodes(net_root):
        for c in v.children:
            if c.kind != 'ret':
                parent[c.idx] = v
    out = []
    for h in reticulation_nodes(net_root):
        paths = []
        for p in h.parents:
            path = [p]
            while path[-1].idx in parent:
                path.append(parent[path[-1].idx])
            paths.append(path)
        ids1 = {v.idx: k for k, v in enumerate(paths[1])}
        for a, v in enumerate(paths[0]):
            if v.idx in ids1:
                b = ids1[v.idx]
                out.append((h, a, b, v is net_root))
                break
        else:
            raise AssertionError('parents of a reticulation have no common ancestor')
    return out


def is_degenerate_cycle(a, b, top_is_root):
    """the two displayed trees of the cycle have the same unrooted topology"""
    return (a + b == 1) or (top_is_root and a + b == 2)


def count_triangles(net_root):
    return sum(1 for (_, a, b, _) in cycle_geometry(net_root) if a + b == 1)


def count_degenerate_cycles(net_root):
    return sum(1 for (_, a, b, r) in cycle_geometry(net_root) if is_degenerate_cycle(a, b, r))


def network_canonical_string(root, n_leaves):
    """Leaf-labelled canonical form (injective on level-1 networks, ignores slot labels):
    sorted children, a reticulation subtree written '#<subtree>' at both parent positions."""
    memo = {}

    def rec(v):
        if v.idx in memo:
            return memo[v.idx]
        if v.kind == 'leaf':
            s = 'L%d' % v.idx
        elif v.kind == 'ret':
            s = '#' + rec(v.children[0])
        else:
            s = '(' + ','.join(sorted(rec(c) for c in v.children)) + ')'
        memo[v.idx] = s
        return s
    return rec(root)


def network_labelled_string(root):
    """canonical form that ALSO records which slot each parent of a reticulation fills;
    the MDP produces 2^R labelled objects per network"""
    def rec(v):
        if v.kind == 'leaf':
            return 'L%d' % v.idx
        if v.kind == 'ret':
            return '#' + rec(v.children[0])
        parts = []
        for c, s in zip(v.children, v.child_slots):
            parts.append(('s%d:' % s if s is not None else '') + rec(c))
        return '(' + ','.join(sorted(parts)) + ')'
    return rec(root)


class PhyloNetwork(object):
    """A terminal object (G, b, theta): owns a deep copy of the node graph (parents filled)."""

    def __init__(self, root, log_score, n_leaves, generate_signature=False, copy_nodes=True):
        self.root = build_network(root) if copy_nodes else root
        self.log_score = log_score
        self.n_leaves = n_leaves
        if generate_signature:
            self.topology_id = network_canonical_string(self.root, n_leaves)
            self.signature = self.topology_id + ('_{:.3f}'.format(log_score) if log_score is not None else '_noscore')
        else:
            self.signature = 'partial_network'

    def update_log_score(self, log_score):
        self.log_score = log_score
        if self.signature != 'partial_network':
            self.signature = self.topology_id + '_{:.3f}'.format(log_score)

    def num_reticulations(self):
        return len(reticulation_nodes(self.root))

    def num_triangles(self):
        return count_triangles(self.root)

    def num_degenerate_cycles(self):
        return count_degenerate_cycles(self.root)

    # ordering for the replay buffer heap (same semantics as PhylogeneticTree)
    def __lt__(self, obj):
        return (obj.log_score - self.log_score) > 0.0001

    def __eq__(self, obj):
        return abs(self.log_score - obj.log_score) < 0.0001

    def edges(self):
        """list of (parent_idx, child_idx, length)"""
        out = []
        for v in all_nodes(self.root):
            for c, d in zip(v.children, v.child_dists):
                out.append((v.idx, c.idx, d))
        return out


# ----------------------------------------------------------------------------------
#  Exact feasibility of an abstract state (cycle masks TRIANGLES / DEGENERATE)
# ----------------------------------------------------------------------------------
_MODE = {'NONE': 0, 'TRIANGLES': 1, 'DEGENERATE': 2}
_S_CAP = 3


def _closing_ok(s, last_step, mode):
    """may a gall whose cycle would have s internal nodes be closed now?
    s = 0 : the two bare slots (parallel edges), never;
    mode TRIANGLES: s >= 2;  mode DEGENERATE: s >= 2, and s >= 3 at the root (last merge)."""
    if s < 1:
        return False
    if mode >= 1 and s < 2:
        return False
    if mode == 2 and last_step and s < 3:
        return False
    return True


@lru_cache(maxsize=None)
def _feasible(n_free, galls, rets_left, mode):
    """Is a terminal network of the class reachable from the abstract state
    (n_free free lineages, galls = sorted tuple of the internal-node counts s (capped at 3)
    of the open galls, rets_left reticulations still allowed)?
    Exact: the legality of every move depends only on this abstraction.
    Terminates: each move lowers  (#lineages + 2 rets_left)  by one."""
    n_galls = len(galls)
    n_lin = n_free + 2 * n_galls
    if n_lin == 1:
        return n_galls == 0
    if n_free >= 2 and _feasible(n_free - 1, galls, rets_left, mode):
        return True
    seen = set()
    for g, s in enumerate(galls):
        if s in seen:
            continue
        seen.add(s)
        rest = galls[:g] + galls[g + 1:]
        if n_free >= 1:
            ns = tuple(sorted(rest + (min(s + 1, _S_CAP),)))
            if _feasible(n_free - 1, ns, rets_left, mode):
                return True
        if _closing_ok(s, n_lin == 2, mode) and _feasible(n_free + 1, rest, rets_left, mode):
            return True
    if n_free >= 1 and rets_left >= 1:
        ns = tuple(sorted(galls + (0,)))
        if _feasible(n_free - 1, ns, rets_left - 1, mode):
            return True
    return False


# ----------------------------------------------------------------------------------
#  The environment
# ----------------------------------------------------------------------------------
class PhyloNetworkEnv(nn.Module):

    def __init__(self, cfg, sequences):
        super().__init__()
        self.cfg = cfg
        self.sequences = sequences
        self.n_leaves = len(sequences)
        net_cfg = cfg.ENV.NETWORK
        self.r_max = net_cfg.R_MAX
        self.lambda_r = net_cfg.LAMBDA_R
        self.prior_type = net_cfg.PRIOR_TYPE
        cycle_mask = net_cfg.CYCLE_MASK
        if cycle_mask == 'NONE' and getattr(net_cfg, 'BLOCK_TRIANGLES', False):
            cycle_mask = 'TRIANGLES'                                 # v2 flag, kept for old configs
        assert cycle_mask in CYCLE_MASKS, cycle_mask
        self.cycle_mask = cycle_mask
        self.mode = _MODE[cycle_mask]
        self.slot_label_correction = bool(net_cfg.SLOT_LABEL_CORRECTION)
        # log r(N, R): size of the topology class with R reticulations (UNIFORM_R prior)
        self.log_class_size = [log_class_count(self.n_leaves, k, cycle_mask) for k in range(self.r_max + 1)]
        self.reward_fn = PhyloTreeReward(cfg.ENV.REWARD)
        self.chars_dict = CHARACTERS_MAPS[cfg.ENV.SEQUENCE_TYPE]
        seq_arrays = np.array([self.seq2array(seq) for seq in self.sequences])
        self.seq_arrays = torch.nn.Parameter(torch.tensor(seq_arrays), requires_grad=False)   # [N, m, c] float64
        self.evolution_model = EvolutionModelTorch(cfg.ENV.EVOLUTION_MODEL)
        assert abs(self.evolution_model.rateM.sum().item()) < 1e-9 and self.seq_arrays.shape[2] == 4, \
            'PhyloGFN-Net assumes the Jukes-Cantor model with uniform root frequencies (1/4)'
        self.m = self.seq_arrays.shape[1]
        self.c = self.seq_arrays.shape[2]
        self.l_max = self.n_leaves + self.r_max            # max number of lineages (each RETICULATE adds one)
        # column-major pair enumeration: k(i, j) = j (j - 1) / 2 + i  (i < j)
        rows, cols = [], []
        for j in range(1, self.l_max):
            for i in range(j):
                rows.append(i)
                cols.append(j)
        self.pair_rows = np.array(rows, dtype=np.int64)
        self.pair_cols = np.array(cols, dtype=np.int64)
        self.pair_index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(self.pair_rows, self.pair_cols))}
        self.n_pairs = len(self.pair_rows)
        self.n_actions = self.n_pairs + self.l_max          # pairs (MERGE) followed by singles (RETICULATE)
        self.parsimony_problem = False
        self.normalize_features = cfg.GFN.NORMALIZE_LIKELIHOOD
        self.type = cfg.ENV.ENVIRONMENT_TYPE
        self._pair_rows_t = None

    # ------------------------------------------------------------------ basics
    def seq2array(self, seq):
        return np.array([self.chars_dict[x] for x in seq])

    @staticmethod
    def n_pairs_of(n):
        return n * (n - 1) // 2

    def get_initial_state(self):
        lineages = []
        for idx in range(self.n_leaves):
            lineages.append(Lineage(NetNode(idx, 'leaf'), tokens=[idx], row=idx))
        return NetworkState(lineages, self.n_leaves, 0, tuple(range(self.n_leaves, self.l_max)))

    def initial_features(self, batch):
        """slot tensor [batch, l_max, 2, m, c]: version A and version B (B = A for closed lineages)"""
        feats = torch.ones(batch, self.l_max, 2, self.m, self.c, dtype=self.seq_arrays.dtype,
                           device=self.seq_arrays.device)
        feats[:, :self.n_leaves, 0] = self.seq_arrays
        feats[:, :self.n_leaves, 1] = self.seq_arrays
        return feats

    # ------------------------------------------------------------------ masks
    @staticmethod
    def merge_legal(u, v):
        """the v1 (level-1) rule; cycle masks add conditions on top of it"""
        if u.open_h is not None and v.open_h is not None:
            if u.open_h is not v.open_h:
                return False                 # two different open cycles: level-2, forbidden
            if u.is_bare and v.is_bare:
                return False                 # the two bare slots of one h: parallel edges
        return True

    @staticmethod
    def merge_case(u, v):
        """0: both closed; 1: only u open; 2: only v open; 3: closing merge (same h)"""
        if u.open_h is None and v.open_h is None:
            return 0
        if v.open_h is None:
            return 1
        if u.open_h is None:
            return 2
        return 3

    def legal_actions(self, state):
        """boolean mask over the padded action space [n_pairs + l_max]"""
        L = state.lineages
        n = len(L)
        mask = np.zeros(self.n_actions, dtype=bool)
        if state.is_done:
            return mask
        P = self.n_pairs_of(n)
        rr, cc = self.pair_rows[:P], self.pair_cols[:P]
        gid = np.array([id(l.open_h) if l.open_h is not None else 0 for l in L], dtype=np.int64)
        bare = np.array([l.is_bare for l in L], dtype=bool)
        gu, gv = gid[rr], gid[cc]
        n_free = int((gid == 0).sum())
        if self.mode == 0:
            both_open = (gu != 0) & (gv != 0)
            bad = both_open & ((gu != gv) | (bare[rr] & bare[cc]))
            mask[:P] = ~bad
            if state.n_ret < self.r_max and n_free >= 2:
                mask[self.n_pairs:self.n_pairs + n] = (gid == 0)
            return mask
        # ---- cycle masks: closing rules + exact feasibility of the successor
        galls = {}
        for i, l in enumerate(L):
            if l.open_h is not None:
                galls.setdefault(gid[i], []).append(l.depth)
        s_of = {g: min(sum(d), _S_CAP) for g, d in galls.items()}
        base = tuple(sorted(s_of.values()))
        rets_left = self.r_max - state.n_ret

        def without(g):
            lst = list(base)
            lst.remove(s_of[g])
            return lst
        ok_ff = n_free >= 2 and _feasible(n_free - 1, base, rets_left, self.mode)
        ok_fc = {g: n_free >= 1 and _feasible(n_free - 1, tuple(sorted(without(g) + [min(s_of[g] + 1, _S_CAP)])), rets_left, self.mode)
                 for g in galls}
        ok_close = {g: _closing_ok(sum(galls[g]), n == 2, self.mode) and
                    _feasible(n_free + 1, tuple(sorted(without(g))), rets_left, self.mode) for g in galls}
        ok_pair = np.zeros(P, dtype=bool)
        ff = (gu == 0) & (gv == 0)
        ok_pair[ff] = ok_ff
        for g in galls:
            fc = ((gu == 0) & (gv == g)) | ((gu == g) & (gv == 0))
            ok_pair[fc] = ok_fc[g]
            ok_pair[(gu == g) & (gv == g)] = ok_close[g]
        mask[:P] = ok_pair
        if rets_left >= 1 and n_free >= 1 and _feasible(n_free - 1, tuple(sorted(base + (0,))), rets_left - 1, self.mode):
            mask[self.n_pairs:self.n_pairs + n] = (gid == 0)
        return mask

    def local_mask(self, state, n_max):
        """the legal mask restricted to the C(n_max, 2) + n_max actions the policy scores"""
        m = self.legal_actions(state)
        P = self.n_pairs_of(n_max)
        return np.concatenate([m[:P], m[self.n_pairs:self.n_pairs + n_max]])

    # ------------------------------------------------------------------ transitions (state part)
    def _next_state(self, state, action):
        """state after `action`; also returns the row bookkeeping needed by the feature kernels:
             merge     -> ('merge', row_u, row_v, case, last_step, bl, br)
             reticulate-> ('ret', row_x, row_new, theta, b_hx)"""
        L = state.lineages
        n = len(L)
        if action['type'] == 'merge':
            k = action['pair']
            i, j = int(self.pair_rows[k]), int(self.pair_cols[k])
            assert j < n and self.merge_legal(L[i], L[j]), 'illegal merge'
            u, v = L[i], L[j]
            bl, br = float(action['edge_action'][0]), float(action['edge_action'][1])
            case = self.merge_case(u, v)
            w = NetNode(state.next_idx, 'tree')
            for lin, d in ((u, bl), (v, br)):
                w.children.append(lin.node)
                w.child_dists.append(d)
                w.child_slots.append(lin.slot)
            open_h = None if case in (0, 3) else (u.open_h if case == 1 else v.open_h)
            depth = 0 if case in (0, 3) else (u.depth + 1 if case == 1 else v.depth + 1)
            new_lin = Lineage(w, open_h=open_h, tokens=u.tokens | v.tokens, depth=depth, row=u.row)
            new_L = L[:i] + [new_lin] + L[i + 1:j] + L[j + 1:]
            new_state = NetworkState(new_L, state.next_idx + 1, state.n_ret, state.free_rows + (v.row,))
            return new_state, ('merge', u.row, v.row, case, n == 2, bl, br)
        else:
            i = int(action['lineage'])
            x = L[i]
            assert x.open_h is None and state.n_ret < self.r_max, 'illegal reticulate'
            theta = float(action['theta'])
            b_hx = float(action['edge_action'])
            h = NetNode(state.next_idx, 'ret')
            h.ret_id = state.n_ret
            h.children = [x.node]
            h.child_dists = [b_hx]
            h.child_slots = [None]
            h.child_dist = b_hx
            h.theta = theta
            h.child_tokens = x.tokens
            fresh = self.n_leaves + state.n_ret                # ordering token of slot 1
            row_new = state.free_rows[0]
            p0 = Lineage(h, slot=0, open_h=h, tokens=x.tokens, depth=0, row=x.row)
            p1 = Lineage(h, slot=1, open_h=h, tokens=[fresh], depth=0, row=row_new)
            new_L = L[:i] + [p0] + L[i + 1:] + [p1]
            new_state = NetworkState(new_L, state.next_idx + 1, state.n_ret + 1, state.free_rows[1:])
            return new_state, ('ret', x.row, row_new, theta, b_hx)

    # ------------------------------------------------------------------ feature kernels (batched)
    def _prior_factor(self, b):
        """exp(log p(b) / m): the exponential edge prior spread over the sites, as PhyloGFN does"""
        return torch.exp(self.evolution_model.compute_log_prior_p(b) / self.m)

    def _merge_kernel(self, feats, b_idx, ru, rv, case, last_step, bl, br):
        """batched MERGE update for trajectories b_idx (same case and last_step flag):
        writes the merged (A, B) into rows ru.  bl, br: tensors [g] on feats.device."""
        uA, uB = feats[b_idx, ru, 0], feats[b_idx, ru, 1]           # [g, m, c] (copies)
        vA, vB = feats[b_idx, rv, 0], feats[b_idx, rv, 1]
        M = self.evolution_model.get_transition_matrices(torch.cat([bl, br]))   # [2g, c, c]
        g = bl.shape[0]
        Mu, Mv = M[:g], M[g:]
        if last_step:
            # PhyloGFN root convention: the prior is put on the single root edge b_l + b_r
            # (pulley principle: the likelihood only depends on the sum)
            pu = torch.ones_like(bl)
            pv = torch.ones_like(br)
            prior = self._prior_factor(bl + br).reshape(-1, 1, 1)
        else:
            pu = self._prior_factor(bl)
            pv = self._prior_factor(br)
            prior = 1.0
        tu = lambda f: torch.einsum('gmv,gvc->gmc', f, Mu) * pu.reshape(-1, 1, 1)
        tv = lambda f: torch.einsum('gmv,gvc->gmc', f, Mv) * pv.reshape(-1, 1, 1)
        if case == 0:
            wA = tu(uA) * tv(vA) * prior
            wB = wA
        elif case == 1:            # u carries the open reticulation
            wA = tu(uA) * tv(vA) * prior
            wB = tu(uB) * tv(vA) * prior
        elif case == 2:            # v carries it
            wA = tu(uA) * tv(vA) * prior
            wB = tu(uA) * tv(vB) * prior
        else:                      # closing merge: mix the two displayed choices
            wA = (tu(uA) * tv(vB) + tu(uB) * tv(vA)) * prior
            wB = wA
        feats[b_idx, ru, 0] = wA
        feats[b_idx, ru, 1] = wB

    def _reticulate_kernel(self, feats, b_idx, rx, rnew, theta, b_hx):
        xA = feats[b_idx, rx, 0]                                               # [g, m, c]
        M = self.evolution_model.get_transition_matrices(b_hx)                 # [g, c, c]
        Lh = torch.einsum('gmv,gvc->gmc', xA, M) * self._prior_factor(b_hx).reshape(-1, 1, 1)
        th = theta.reshape(-1, 1, 1)
        feats[b_idx, rx, 0] = th * Lh
        feats[b_idx, rx, 1] = 1.0
        feats[b_idx, rnew, 0] = (1.0 - th) * Lh
        feats[b_idx, rnew, 1] = 1.0

    def log_topology_prior_unnormalised(self, n_ret):
        """log p(G) up to the normaliser over the class, MINUS the slot-label multiplicity:
             EXP_R     : -lambda_r R - R log 2
             UNIFORM_R : -lambda_r R - log r(N, R) - R log 2
        Every network with R reticulations is reached as 2^R terminal objects of the MDP
        (which parent fills slot 0 of each reticulation, with theta <-> 1 - theta); the two
        labellings have the same reward, so dividing by 2^R makes the sum over terminal
        objects equal to the sum over networks.  Enabled by ENV.NETWORK.SLOT_LABEL_CORRECTION
        (off by default: the v1 target)."""
        base = -self.lambda_r * n_ret
        if self.slot_label_correction:
            base -= n_ret * math.log(2.0)
        if self.prior_type == 'UNIFORM_R':
            base -= self.log_class_size[n_ret]
        return base

    def terminal_log_scores(self, root_feats, n_rets):
        """root_feats [g, m, c] -> log P(Y | G, b, theta) + log P(b) + log p(G) (unnormalised) [g]"""
        log_p = torch.sum(torch.log(torch.sum(root_feats / 4.0, -1)), -1)
        prior = torch.tensor([self.log_topology_prior_unnormalised(int(r)) for r in n_rets],
                             dtype=log_p.dtype, device=log_p.device)
        return log_p + prior

    # ------------------------------------------------------------------ batched step
    def batch_apply_actions(self, actions, feats, batch_idx, states):
        """Apply one action per trajectory.
        actions   : list of action dicts (one per entry of batch_idx)
        feats     : the slot tensor [B, l_max, 2, m, c] of the whole batch, updated IN PLACE
                    (None: only the states are updated, no likelihood)
        batch_idx : indices (into feats) of the trajectories the actions belong to
        states    : their current states
        returns new_states, log_scores [g] (nan unless terminal), log_rewards [g] (nan unless terminal)
        """
        new_states, ops = [], []
        for s, a in zip(states, actions):
            ns, op = self._next_state(s, a)
            new_states.append(ns)
            ops.append(op)
        g = len(actions)
        log_scores = torch.full((g,), float('nan'), dtype=torch.float64)
        log_rewards = torch.full((g,), float('nan'), dtype=torch.float64)
        if feats is None:
            return new_states, log_scores, log_rewards
        dev, dt = feats.device, feats.dtype
        batch_idx = np.asarray(batch_idx)
        groups = {}
        for t, op in enumerate(ops):
            key = ('ret',) if op[0] == 'ret' else ('merge', op[3], op[4])
            groups.setdefault(key, []).append(t)
        for key, ts in groups.items():
            ts_np = np.array(ts)
            b_idx = torch.as_tensor(batch_idx[ts_np], device=dev)
            if key[0] == 'ret':
                rx = torch.as_tensor([ops[t][1] for t in ts], device=dev)
                rn = torch.as_tensor([ops[t][2] for t in ts], device=dev)
                th = torch.tensor([ops[t][3] for t in ts], dtype=dt, device=dev)
                bh = torch.tensor([ops[t][4] for t in ts], dtype=dt, device=dev)
                self._reticulate_kernel(feats, b_idx, rx, rn, th, bh)
            else:
                ru = torch.as_tensor([ops[t][1] for t in ts], device=dev)
                rv = torch.as_tensor([ops[t][2] for t in ts], device=dev)
                bl = torch.tensor([ops[t][5] for t in ts], dtype=dt, device=dev)
                br = torch.tensor([ops[t][6] for t in ts], dtype=dt, device=dev)
                self._merge_kernel(feats, b_idx, ru, rv, key[1], key[2], bl, br)
        done = [t for t, ns in enumerate(new_states) if ns.is_done]
        if done:
            b_idx = torch.as_tensor(batch_idx[np.array(done)], device=dev)
            rows = torch.as_tensor([new_states[t].lineages[0].row for t in done], device=dev)
            ls = self.terminal_log_scores(feats[b_idx, rows, 0], [new_states[t].n_ret for t in done])
            ls = ls.detach().cpu().double()
            lr = self.reward_fn(ls)
            for q, t in enumerate(done):
                new_states[t].log_score = float(ls[q])
                new_states[t].log_reward = float(lr[q])
                log_scores[t] = float(ls[q])
                log_rewards[t] = float(lr[q])
        return new_states, log_scores, log_rewards

    def apply_action(self, state, action, feats=None):
        """single-trajectory convenience wrapper.  feats: slot tensor [l_max, 2, m, c] of this
        trajectory, updated IN PLACE (and returned).  returns new_state, feats, log_score"""
        f = feats.unsqueeze(0) if feats is not None else None
        new_states, ls, _ = self.batch_apply_actions([action], f, [0], [state])
        ns = new_states[0]
        return ns, feats, (float(ls[0]) if ns.is_done and feats is not None else None)

    # ------------------------------------------------------------------ policy inputs
    def prepare_rollout_inputs(self, feats, batch_idx, states, random_spec, input_actions=None):
        """Input dict of the policy for the active trajectories `batch_idx` (states in the
        same order).  Only the rows of the n_max = max #lineages largest state are gathered:
        the policy scores C(n_max, 2) + n_max actions (local ids; see TBGFlowNetNet)."""
        b = len(states)
        n_max = max(len(s.lineages) for s in states)
        P = self.n_pairs_of(n_max)
        dev = feats.device
        rows = np.zeros((b, n_max), dtype=np.int64)
        nb = np.zeros(b, dtype=np.int64)
        flags = np.zeros((b, n_max, 4), dtype=np.float32)
        masks = np.zeros((b, P + n_max), dtype=bool)
        for k, s in enumerate(states):
            L = s.lineages
            nb[k] = len(L)
            masks[k] = self.local_mask(s, n_max)
            for i, l in enumerate(L):
                rows[k, i] = l.row
                if l.open_h is not None:
                    # which side of h this lineage is on (slot 0 <-> theta, slot 1 <-> 1 - theta):
                    # slot 0's side owns the tokens of h's child, slot 1's side never does
                    side = l.slot if l.is_bare else (0 if l.open_h.child_tokens <= l.tokens else 1)
                    flags[k, i] = (1.0, 1.0 if l.is_bare else 0.0, float(side),
                                   l.open_h.theta if side == 0 else 1.0 - l.open_h.theta)
        rows_t = torch.as_tensor(rows, device=dev)
        bidx_t = torch.as_tensor(np.asarray(batch_idx), device=dev)
        x = feats[bidx_t.unsqueeze(1), rows_t]                                   # [b, n_max, 2, m, c]
        if self.normalize_features:
            x = x / x.sum(-1, keepdim=True)
            x = x.float()
            x[torch.isnan(x)] = 1.0 / self.c
        else:
            x = x.float()
        x = x.reshape(b, n_max, -1)
        x = torch.cat([x, torch.as_tensor(flags, device=dev)], dim=-1)           # [b, n_max, 2mc + 4]
        nb_t = torch.as_tensor(nb, device=dev)
        pad = torch.arange(n_max, device=dev).unsqueeze(0) >= nb_t.unsqueeze(1)  # True = padding
        if self._pair_rows_t is None or self._pair_rows_t.device != dev:
            self._pair_rows_t = torch.as_tensor(self.pair_rows, device=dev)
            self._pair_cols_t = torch.as_tensor(self.pair_cols, device=dev)
        input_dict = {
            'batch_input': x,
            'batch_nb_seq': nb_t,
            'padding_mask': pad,
            'action_mask': torch.as_tensor(masks, device=dev),                   # True = legal, local ids
            'pair_rows': self._pair_rows_t[:P],
            'pair_cols': self._pair_cols_t[:P],
            'n_pairs': P,                                                         # LOCAL number of pairs
            'n_max': n_max,
            'random_spec': random_spec,
            'batch_size': b,
            'last_step': torch.as_tensor(nb == 2, device=dev),
        }
        if input_actions is not None:
            input_dict['input_actions'] = input_actions
        return input_dict

    # ------------------------------------------------------------------ action helpers
    def action_from_id(self, a, state, random_params=False, params=None):
        n = len(state.lineages)
        if a < self.n_pairs:
            if random_params:
                if n == 2:
                    b = math.exp(random.uniform(-10, 0))
                    edge = (b / 2, b / 2)
                else:
                    edge = (math.exp(random.uniform(-10, 0)), math.exp(random.uniform(-10, 0)))
            else:
                edge = params['edge_action']
            return {'type': 'merge', 'pair': int(a), 'edge_action': edge}
        else:
            i = int(a - self.n_pairs)
            if random_params:
                theta = random.uniform(0.02, 0.98)
                b_hx = math.exp(random.uniform(-10, 0))
            else:
                theta, b_hx = params['theta'], params['edge_action']
            return {'type': 'ret', 'lineage': i, 'theta': theta, 'edge_action': b_hx}

    @staticmethod
    def action_id(action, n_pairs):
        return action['pair'] if action['type'] == 'merge' else n_pairs + action['lineage']

    def random_action(self, state):
        ids = np.where(self.legal_actions(state))[0]
        assert len(ids) > 0, 'dead end'
        return self.action_from_id(int(random.choice(ids)), state, random_params=True)

    # ------------------------------------------------------------------ random rollouts
    def sample(self, num_trajs, generate_full_trajectory):
        """random rollouts (uniform over legal actions, random parameters) -- Z init and buffers"""
        trajectories = []
        for _ in range(num_trajs):
            state = self.get_initial_state()
            feats = self.initial_features(1)[0]
            traj = Trajectory(state) if generate_full_trajectory else SimpleTrajectory()
            while not state.is_done:
                a = self.random_action(state)
                new_state, feats, log_score = self.apply_action(state, a, feats)
                lr = new_state.log_reward if new_state.log_reward is not None else 0.0
                if generate_full_trajectory:
                    traj.update(new_state, a, lr, new_state.is_done)
                else:
                    traj.update(a, lr)
                state = new_state
            if generate_full_trajectory:
                traj.current_state.network = self.state_to_network(state)
            trajectories.append(traj)
        return trajectories

    def state_to_network(self, state):
        assert state.is_done
        return PhyloNetwork(state.lineages[0].node, state.log_score, self.n_leaves, generate_signature=True)

    def actions_to_trajectory(self, actions):
        state = self.get_initial_state()
        feats = self.initial_features(1)[0]
        traj = Trajectory(state)
        for a in actions:
            new_state, feats, _ = self.apply_action(state, a, feats)
            lr = new_state.log_reward if new_state.log_reward is not None else 0.0
            traj.update(new_state, a, lr, new_state.is_done)
            state = new_state
        traj.current_state.network = self.state_to_network(state)
        return traj

    def batch_actions_to_networks(self, batch_actions, batch_log_scores):
        nets = []
        for actions, score in zip(batch_actions, batch_log_scores):
            state = self.get_initial_state()
            for a in actions:
                state, _, _ = self.apply_action(state, a, None)
            net = self.state_to_network(state)
            net.update_log_score(float(score))
            nets.append(net)
        return nets

    # ------------------------------------------------------------------ backward sampling
    def sample_backward_from_network(self, net):
        """Uniform backward policy on a terminal PhyloNetwork: returns (forward action list,
        log P_B of the path).  Backward moves: un-merge any tree-root lineage, or
        un-reticulate any reticulation whose two slots are both bare lineages
        (CYCLE_MASK NONE: only when a free lineage exists).  Their number is |Pa(s)|."""
        cur = [(net.root, None)]                      # (node, slot)
        moves, parents_num = [], []

        def is_free(entry):
            v, slot = entry
            if slot is not None:
                return False
            sub = {u.idx for u in all_nodes(v)}
            return all(all(p is not None and p.idx in sub for p in h.parents) for h in reticulation_nodes(v))

        while True:
            unmerge = [k for k, (v, slot) in enumerate(cur) if slot is None and v.kind == 'tree']
            bare_count = {}
            for k, (v, slot) in enumerate(cur):
                if slot is not None:
                    bare_count.setdefault(v.idx, []).append(k)
            n_free = sum(1 for e in cur if is_free(e))
            unret = [ks for ks in bare_count.values() if len(ks) == 2] if (self.mode != 0 or n_free >= 1) else []
            options = [('unmerge', k) for k in unmerge] + [('unret', ks) for ks in unret]
            if not options:
                break
            parents_num.append(len(options))
            mv = random.choice(options)
            if mv[0] == 'unmerge':
                k = mv[1]
                w = cur[k][0]
                new_entries = [(c, s) for c, s in zip(w.children, w.child_slots)]
                cur = cur[:k] + cur[k + 1:] + new_entries
                moves.append(('merge', w))
            else:
                ks = mv[1]
                h = cur[ks[0]][0]
                cur = [e for t, e in enumerate(cur) if t not in ks] + [(h.children[0], None)]
                moves.append(('ret', h))
        moves = moves[::-1]
        parents_num = parents_num[::-1]
        # ---- replay forward to obtain action indices in the sorted-state convention
        state = self.get_initial_state()
        new_of = {l.node.idx: l.node for l in state.lineages}       # original idx -> replayed node
        actions = []
        for mv in moves:
            L = state.lineages
            if mv[0] == 'merge':
                w = mv[1]
                pos, dist_of = [], {}
                for child, dist, slot in zip(w.children, w.child_dists, w.child_slots):
                    if slot is not None:
                        p = next(t for t, l in enumerate(L) if l.is_bare and l.node is new_of[child.idx] and l.slot == slot)
                    else:
                        p = next(t for t, l in enumerate(L) if (not l.is_bare) and l.node is new_of[child.idx])
                    pos.append(p)
                    dist_of[p] = dist
                i, j = sorted(pos)
                a = {'type': 'merge', 'pair': self.pair_index[(i, j)], 'edge_action': (dist_of[i], dist_of[j])}
                state, _, _ = self.apply_action(state, a, None)
                new_of[w.idx] = state.lineages[i].node
            else:
                h = mv[1]
                x = h.children[0]
                i = next(t for t, l in enumerate(L) if (not l.is_bare) and l.node is new_of[x.idx])
                a = {'type': 'ret', 'lineage': i, 'theta': h.theta, 'edge_action': h.child_dist}
                state, _, _ = self.apply_action(state, a, None)
                new_of[h.idx] = state.lineages[i].node
            actions.append(a)
        assert state.is_done
        return actions, -np.log(np.array(parents_num, dtype=float))

    def compute_network_log_score(self, net):
        """terminal log score of a PhyloNetwork, by replaying one backward-sampled path"""
        actions, _ = self.sample_backward_from_network(net)
        state = self.get_initial_state()
        feats = self.initial_features(1)[0]
        log_score = None
        for a in actions:
            state, feats, log_score = self.apply_action(state, a, feats)
        return log_score
