"""
PhyloGFN-Net: a level-1 phylogenetic NETWORK environment for PhyloGFN.

This file is the network counterpart of `binary_tree_env_one_step_likelihood.py`.
Everything that the original environment does for trees is done here for level-1
networks (networks in which every cycle contains exactly one reticulation node and
cycles share no edges).

The MDP (see the lecture, Part III)
-----------------------------------
state  = list of open *lineages* (roots of partial networks), kept sorted by a key.
actions
  MERGE(i, j)      join lineages i and j under a new tree node; sample the two new
                   branch lengths (exactly the PhyloGFN action);
  RETICULATE(i)    put a new reticulation node h above lineage i; h receives two
                   *bare parent slots* p1, p2 which become two new open lineages;
                   sample theta_h (inheritance weight of slot p1) and the length of
                   the edge h -> i.
terminal state     one lineage left, carrying no unclosed reticulation.

A lineage carries at most one "open" reticulation h (a reticulation whose two parent
slots are not yet both below the same lineage).  Level-1 masks (Build slide):
  MERGE(u, v)      legal iff |open(u) U open(v)| <= 1 and (u, v) are not the two bare
                   slots of the same reticulation;
  RETICULATE(x)    legal iff open(x) is empty, another lineage y != x with open(y)
                   empty exists (no dead ends), and fewer than R_MAX reticulations
                   were created.

Likelihood without the 2^R blow-up ("mixing at the top of each gall")
---------------------------------------------------------------------
A lineage with an open reticulation h keeps TWO Felsenstein feature tensors:
  A : the displayed trees in which THIS side keeps its edge into h  (weight theta or 1-theta)
  B : the displayed trees in which this side's edge into h is deleted (the slot is
      suppressed; for a bare slot B is the all-ones vector, "no data below").
When the two carriers of h finally merge (the closing merge), the exact displayed-tree
likelihood of the closed cycle is   A_u (x) B_v  +  B_u (x) A_v   (per site), and the
result is a single feature again.  Because Felsenstein's recursion is linear in the
child features and the cycles of a level-1 network are edge-disjoint, this is exactly
sum_T w(T|theta) P(Y_i | T, b) for every site i  (verified in tests/test_network_env.py
against an explicit enumeration of the 2^R displayed trees).
"""
import math
import random
import itertools
import numpy as np
import torch
from torch import nn

from src.env.trajectory import Trajectory, SimpleTrajectory
from src.utils.evolution_model_torch import EvolutionModelTorch
from src.env.binary_tree_env_one_step_likelihood import CHARACTERS_MAPS, PhyloTreeReward


# ----------------------------------------------------------------------------------
#  Network data structures (replace ete3.TreeNode, which cannot hold two parents)
# ----------------------------------------------------------------------------------
class NetNode(object):
    """A node of a (partial) rooted phylogenetic network.

    kind: 'leaf' | 'tree' | 'ret'
      leaf : no children, one parent
      tree : two children (with branch lengths), one parent
      ret  : one child (branch length `child_dist`), two parents = two SLOTS.
             slot k (k = 0, 1) is filled by `parents[k]` with length `parent_dists[k]`;
             theta is the inheritance weight of slot 0, (1 - theta) of slot 1.
    """
    __slots__ = ('idx', 'kind', 'children', 'child_dists', 'parents', 'parent_dists',
                 'theta', 'child_dist', 'tokens', 'ret_id')

    def __init__(self, idx, kind):
        self.idx = idx                 # unique integer id (leaves: 0..N-1)
        self.kind = kind
        self.children = []             # tree: [left, right]; ret: [child]; leaf: []
        self.child_dists = []          # branch lengths of the edges to the children
        self.parents = [None, None]    # ret only: the two parent nodes (slots)
        self.parent_dists = [None, None]
        self.theta = None              # ret only
        self.child_dist = None         # ret only: length of the edge h -> child
        self.tokens = frozenset()      # ordering tokens owned by this node (see Lineage)
        self.ret_id = None             # ret only: creation order 0, 1, 2, ...

    # ---- helpers used by tests / signatures -----------------------------------
    def leaves(self):
        out = set()
        stack = [self]
        seen = set()
        while stack:
            v = stack.pop()
            if v.idx in seen:
                continue
            seen.add(v.idx)
            if v.kind == 'leaf':
                out.add(v.idx)
            stack.extend(v.children)
        return out


class Lineage(object):
    """An entry of the state list.

    Two flavours:
      * a real lineage: `node` is a NetNode (leaf / tree / ret whose cycle is closed
        or still open below it);  `slot is None`.
      * a BARE PARENT SLOT of a reticulation h: `node is h`, `slot in {0, 1}`.  It has
        no data of its own; it is the "hook" waiting for a parent.

    open_h : the reticulation node this lineage carries unclosed (None if closed).
    key    : sorting key = min of the ownership tokens.  Tokens are the leaf indices
             0..N-1 plus one fresh token N + ret_id for slot 1 of every reticulation;
             the leaves below h are owned by slot-0's side.  Tokens form a partition of
             a set, therefore keys are unique and the sorted order is well defined
             (this is what lets the backward sampler recompute action indices, exactly
             like PhyloGFN's min_seq_idx trick).
    """
    __slots__ = ('node', 'slot', 'open_h', 'tokens', 'key')

    def __init__(self, node, slot=None, open_h=None, tokens=None):
        self.node = node
        self.slot = slot
        self.open_h = open_h
        self.tokens = frozenset(tokens) if tokens is not None else frozenset()
        self.key = min(self.tokens)

    @property
    def is_bare(self):
        return self.slot is not None

    @property
    def is_leaf(self):
        return (not self.is_bare) and self.node.kind == 'leaf'


class NetworkState(object):
    """A GFlowNet state: sorted list of lineages + bookkeeping."""

    def __init__(self, lineages, next_idx, n_ret, log_reward=None, log_score=None):
        self.lineages = lineages          # sorted by key
        self.next_idx = next_idx          # next free node id
        self.n_ret = n_ret                # reticulations created so far
        self.log_reward = log_reward
        self.log_score = log_score
        self.num_lineages = len(lineages)
        self.is_done = (len(lineages) == 1 and lineages[0].open_h is None)

    # number of parent states under the uniform backward policy
    def num_parents(self):
        """|Pa(s)| = (#tree-root lineages: un-merge) + (#reticulations whose two slots are
        both bare: un-reticulate).  Un-reticulating h is only a parent if the forward
        RETICULATE would have been legal from that parent, i.e. if the state has a free
        lineage other than h's child (the no-dead-end mask)."""
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
                    n += 1                                 # un-merge
        if n_free >= 1:
            n += sum(1 for c in seen_ret.values() if c == 2)   # un-reticulate
        return n


class PhyloNetwork(object):
    """A terminal object (G, b, theta): wraps the root NetNode, mirrors PhylogeneticTree."""

    def __init__(self, root, log_score, n_leaves, generate_signature=False):
        self.root = root
        self.log_score = log_score
        self.n_leaves = n_leaves
        if generate_signature:
            self.topology_id = network_canonical_string(root, n_leaves)
            self.signature = self.topology_id + ('_{:.3f}'.format(log_score) if log_score is not None else '_noscore')
        else:
            self.signature = 'partial_network'

    def update_log_score(self, log_score):
        self.log_score = log_score
        if self.signature != 'partial_network':
            self.signature = self.topology_id + '_{:.3f}'.format(log_score)

    def num_reticulations(self):
        return len(reticulation_nodes(self.root))

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


def network_canonical_string(root, n_leaves):
    """Leaf-labelled canonical form (injective on level-1 networks): sorted children,
    a reticulation subtree written '#<subtree>' at both parent positions."""
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
        self.r_max = net_cfg.R_MAX                  # at most R_MAX reticulations
        self.lambda_r = net_cfg.LAMBDA_R            # prior p(G) ∝ exp(-lambda_r * R(G))
        self.prior_type = net_cfg.PRIOR_TYPE        # 'EXP_R' or 'UNIFORM_R' (see defaults.py)
        from src.utils.level1_counts import level1_count
        # log r(N, R): number of level-1 networks with R reticulations (used by the UNIFORM_R prior)
        self.log_class_size = [math.log(level1_count(self.n_leaves, k)) for k in range(net_cfg.R_MAX + 1)]
        self.reward_fn = PhyloTreeReward(cfg.ENV.REWARD)
        self.chars_dict = CHARACTERS_MAPS[cfg.ENV.SEQUENCE_TYPE]
        seq_arrays = np.array([self.seq2array(seq) for seq in self.sequences])
        self.seq_arrays = torch.nn.Parameter(torch.tensor(seq_arrays), requires_grad=False)   # [N, m, c] float64
        self.evolution_model = EvolutionModelTorch(cfg.ENV.EVOLUTION_MODEL)
        self.m = self.seq_arrays.shape[1]
        self.c = self.seq_arrays.shape[2]
        # maximum number of lineages ever present: N + R_MAX (each RETICULATE adds one)
        self.l_max = self.n_leaves + self.r_max
        rows, cols = torch.triu_indices(self.l_max, self.l_max, offset=1)
        self.pair_rows = rows.numpy()
        self.pair_cols = cols.numpy()
        self.pair_index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(self.pair_rows, self.pair_cols))}
        self.n_pairs = len(self.pair_rows)
        self.n_actions = self.n_pairs + self.l_max     # pairs (MERGE) followed by singles (RETICULATE)
        self.parsimony_problem = False
        self.normalize_features = cfg.GFN.NORMALIZE_LIKELIHOOD
        self.type = cfg.ENV.ENVIRONMENT_TYPE

    # ------------------------------------------------------------------ basics
    def seq2array(self, seq):
        return np.array([self.chars_dict[x] for x in seq])

    def get_initial_state(self):
        lineages = []
        for idx in range(self.n_leaves):
            node = NetNode(idx, 'leaf')
            node.tokens = frozenset([idx])
            lineages.append(Lineage(node, tokens=[idx]))
        return NetworkState(lineages, self.n_leaves, 0)

    def initial_features(self, batch):
        """feature tensor [batch, l_max, 2, m, c]: version A and version B (B = A for closed lineages)"""
        feats = torch.ones(batch, self.l_max, 2, self.m, self.c, dtype=self.seq_arrays.dtype,
                           device=self.seq_arrays.device)
        feats[:, :self.n_leaves, 0] = self.seq_arrays
        feats[:, :self.n_leaves, 1] = self.seq_arrays
        return feats

    # ------------------------------------------------------------------ masks
    def legal_actions(self, state):
        """boolean mask over the padded action space [n_pairs + l_max]."""
        L = state.lineages
        n = len(L)
        mask = np.zeros(self.n_actions, dtype=bool)
        if state.is_done:
            return mask
        for i in range(n):
            for j in range(i + 1, n):
                mask[self.pair_index[(i, j)]] = self.merge_legal(L[i], L[j])
        n_free = sum(1 for l in L if l.open_h is None)
        if state.n_ret < self.r_max:
            for i in range(n):
                if L[i].open_h is None and n_free >= 2:
                    mask[self.n_pairs + i] = True
        return mask

    @staticmethod
    def merge_legal(u, v):
        if u.open_h is not None and v.open_h is not None:
            if u.open_h is not v.open_h:
                return False                 # two different open cycles: level-2, forbidden
            if u.is_bare and v.is_bare:
                return False                 # the two bare slots of one h: parallel edges
        return True

    # ------------------------------------------------------------------ transitions
    def merge_case(self, u, v):
        """0: both closed; 1: only u open; 2: only v open; 3: closing merge (same h)"""
        if u.open_h is None and v.open_h is None:
            return 0
        if v.open_h is None:
            return 1
        if u.open_h is None:
            return 2
        return 3

    def apply_action(self, state, action, feats=None):
        """Apply one action to one state.

        action: {'type': 'merge', 'pair': k, 'edge_action': (b_left, b_right)}
                {'type': 'ret',   'lineage': i, 'theta': t, 'edge_action': b_hx}
        feats : optional tensor [l_max, 2, m, c] of this trajectory, updated in place-out.
        returns new_state, new_feats, log_score (None unless terminal)
        """
        L = list(state.lineages)
        n = len(L)
        if action['type'] == 'merge':
            i, j = int(self.pair_rows[action['pair']]), int(self.pair_cols[action['pair']])
            assert j < n and self.merge_legal(L[i], L[j]), 'illegal merge'
            u, v = L[i], L[j]
            bl, br = float(action['edge_action'][0]), float(action['edge_action'][1])
            last_step = (n == 2)
            w = NetNode(state.next_idx, 'tree')
            case = self.merge_case(u, v)
            # --- attach children (a bare slot attaches the reticulation node itself)
            for lin, d, slot_side in ((u, bl, 0), (v, br, 1)):
                child = lin.node
                w.children.append(child)
                w.child_dists.append(d)
                if lin.is_bare:
                    child.parents[lin.slot] = w
                    child.parent_dists[lin.slot] = d
            w.tokens = u.tokens | v.tokens
            open_h = None if case in (0, 3) else (u.open_h if case == 1 else v.open_h)
            new_lin = Lineage(w, open_h=open_h, tokens=u.tokens | v.tokens)
            new_L = L[:i] + [new_lin] + L[i + 1:j] + L[j + 1:]
            new_feats = None
            if feats is not None:
                new_feats = self._merge_features(feats, i, j, case, bl, br, last_step)
            new_state = NetworkState(new_L, state.next_idx + 1, state.n_ret)
            log_score = None
            if new_state.is_done and feats is not None:
                log_score = self._terminal_log_score(new_feats[0, 0], new_state.n_ret)
                new_state.log_score = log_score
                new_state.log_reward = float(self.reward_fn(log_score))
            return new_state, new_feats, log_score
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
            h.child_dist = b_hx
            h.theta = theta
            h.tokens = x.tokens
            fresh = self.n_leaves + state.n_ret          # ordering token of slot 1
            p1 = Lineage(h, slot=0, open_h=h, tokens=x.tokens)
            p2 = Lineage(h, slot=1, open_h=h, tokens=[fresh])
            new_L = L[:i] + [p1] + L[i + 1:] + [p2]
            new_feats = None
            if feats is not None:
                new_feats = self._reticulate_features(feats, i, n, theta, b_hx)
            new_state = NetworkState(new_L, state.next_idx + 1, state.n_ret + 1)
            return new_state, new_feats, None

    # ------------------------------------------------------------------ features
    def _transport(self, f, b):
        """f: [k, m, c] child features, b: [k] branch lengths -> M(b) f (with the edge prior folded in)."""
        M = self.evolution_model.get_transition_matrices(b)                       # [k, c, c]
        out = torch.einsum('bmv,bvc->bmc', f, M)
        log_prior = self.evolution_model.compute_log_prior_p(b)                    # [k]
        return out * torch.exp(log_prior / self.m).reshape(-1, 1, 1)

    def _merge_features(self, feats, i, j, case, bl, br, last_step):
        """feats: [l_max, 2, m, c] -> new feats with lineage i replaced by the merge, j removed."""
        dt, dev = feats.dtype, feats.device
        uA, uB = feats[i, 0], feats[i, 1]
        vA, vB = feats[j, 0], feats[j, 1]
        if last_step:
            # PhyloGFN root convention: the prior is put on the single edge b_l + b_r
            # (pulley principle: the likelihood only depends on the sum).
            b = torch.tensor([bl, br], dtype=dt, device=dev)
            M = self.evolution_model.get_transition_matrices(b)
            tu = lambda f: torch.einsum('mv,vc->mc', f, M[0])
            tv = lambda f: torch.einsum('mv,vc->mc', f, M[1])
            prior = torch.exp(self.evolution_model.compute_log_prior_p(b[0] + b[1]) / self.m)
        else:
            b = torch.tensor([bl, br], dtype=dt, device=dev)
            M = self.evolution_model.get_transition_matrices(b)
            pl = torch.exp(self.evolution_model.compute_log_prior_p(b) / self.m)
            tu = lambda f: torch.einsum('mv,vc->mc', f, M[0]) * pl[0]
            tv = lambda f: torch.einsum('mv,vc->mc', f, M[1]) * pl[1]
            prior = 1.0
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
        keep = [k for k in range(feats.shape[0]) if k != j]
        new = feats[keep].clone()
        new = torch.cat([new, torch.ones(1, 2, self.m, self.c, dtype=dt, device=dev)], dim=0)   # keep l_max rows
        new[i, 0] = wA
        new[i, 1] = wB
        return new

    def _reticulate_features(self, feats, i, n, theta, b_hx):
        dt, dev = feats.dtype, feats.device
        xA = feats[i, 0]
        b = torch.tensor([b_hx], dtype=dt, device=dev)
        Lh = self._transport(xA.unsqueeze(0), b)[0]          # feature at h (edge prior folded in)
        ones = torch.ones_like(Lh)
        new = feats.clone()
        new[i, 0] = theta * Lh
        new[i, 1] = ones
        new[n, 0] = (1.0 - theta) * Lh                        # slot 1 appended at position n
        new[n, 1] = ones
        return new

    def log_topology_prior_unnormalised(self, n_ret):
        """log p(G) up to the normaliser over the class:
             EXP_R     : -lambda_r * R                      (every network of the class weighted by exp(-λR))
             UNIFORM_R : -lambda_r * R - log r(N, R)        (prior on R, then uniform over the r(N,R) topologies)"""
        if self.prior_type == 'UNIFORM_R':
            return -self.lambda_r * n_ret - self.log_class_size[n_ret]
        return -self.lambda_r * n_ret

    def _terminal_log_score(self, root_feat, n_ret):
        """root_feat [m, c] -> log P(Y | G, b, theta) + log P(b) + log p(G) (unnormalised)"""
        log_p = torch.sum(torch.log(torch.sum(root_feat / 4.0, -1)), -1)
        return float(log_p) + self.log_topology_prior_unnormalised(n_ret)

    # ------------------------------------------------------------------ batched step
    def batch_apply_actions(self, actions, feats, states):
        """actions: list (one per trajectory); feats: [b, l_max, 2, m, c] or None; states: list or None."""
        new_states, new_feats, log_scores, log_rewards = [], [], [], []
        for k, a in enumerate(actions):
            f = feats[k] if feats is not None else None
            s = states[k] if states is not None else None
            if s is None:
                # states are needed to know cases; the rollout always keeps them
                raise ValueError('states are required')
            ns, nf, ls = self.apply_action(s, a, f)
            new_states.append(ns)
            new_feats.append(nf)
            log_scores.append(ls if ls is not None else float('nan'))
            log_rewards.append(ns.log_reward if ns.log_reward is not None else float('nan'))
        if feats is not None:
            new_feats = torch.stack(new_feats)
        else:
            new_feats = None
        return new_states, new_feats, torch.tensor(log_scores, dtype=torch.float64), torch.tensor(log_rewards, dtype=torch.float64)

    # ------------------------------------------------------------------ policy inputs
    def prepare_rollout_inputs(self, feats, states, random_spec, input_actions=None):
        """Build the input dict of the policy for the ACTIVE trajectories.

        feats : [b, l_max, 2, m, c];  states : list of NetworkState (same order)
        """
        b = feats.shape[0]
        if self.normalize_features:
            x = feats / feats.sum(-1, keepdim=True)
            x = x.float()
            x[torch.isnan(x)] = 1.0 / self.c
        else:
            x = feats.float()
        x = x.reshape(b, self.l_max, -1)                                         # [b, l_max, 2mc]
        flags = torch.zeros(b, self.l_max, 4, dtype=x.dtype, device=x.device)
        nb = torch.zeros(b, dtype=torch.long)
        masks = np.zeros((b, self.n_actions), dtype=bool)
        for k, s in enumerate(states):
            nb[k] = len(s.lineages)
            masks[k] = self.legal_actions(s)
            for i, l in enumerate(s.lineages):
                if l.open_h is not None:
                    flags[k, i, 0] = 1.0
                    flags[k, i, 1] = 1.0 if l.is_bare else 0.0
                    # which side of h this lineage owns (slot 0 <-> theta, slot 1 <-> 1 - theta)
                    side = l.slot if l.is_bare else (0 if l.open_h.tokens <= l.tokens else 1)
                    flags[k, i, 2] = float(side)
                    flags[k, i, 3] = l.open_h.theta if side == 0 else 1.0 - l.open_h.theta
        x = torch.cat([x, flags], dim=-1)
        pad = torch.arange(self.l_max).unsqueeze(0) >= nb.unsqueeze(1)             # True = padding
        input_dict = {
            'batch_input': x,
            'batch_nb_seq': nb.to(x.device),
            'padding_mask': pad.to(x.device),
            'action_mask': torch.tensor(masks).to(x.device),                          # True = legal
            'pair_rows': torch.tensor(self.pair_rows).to(x.device),
            'pair_cols': torch.tensor(self.pair_cols).to(x.device),
            'n_pairs': self.n_pairs,
            'random_spec': random_spec,
            'batch_size': b,
            'last_step': torch.tensor([len(s.lineages) == 2 for s in states]).to(x.device),
        }
        if input_actions is not None:
            input_dict['input_actions'] = input_actions
        return input_dict

    # ------------------------------------------------------------------ random rollouts
    def random_action(self, state, ret_prob=None):
        mask = self.legal_actions(state)
        ids = np.where(mask)[0]
        a = int(random.choice(ids))
        return self.action_from_id(a, state, random_params=True)

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
            return {'type': 'merge', 'pair': a, 'edge_action': edge}
        else:
            i = a - self.n_pairs
            if random_params:
                theta = random.uniform(0.02, 0.98)
                b_hx = math.exp(random.uniform(-10, 0))
            else:
                theta, b_hx = params['theta'], params['edge_action']
            return {'type': 'ret', 'lineage': i, 'theta': theta, 'edge_action': b_hx}

    @staticmethod
    def action_id(action, n_pairs):
        return action['pair'] if action['type'] == 'merge' else n_pairs + action['lineage']

    def sample(self, num_trajs, generate_full_trajectory):
        """random rollouts (uniform over legal actions, random parameters) -- used for Z init and buffers"""
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
        """Uniform backward policy: returns (forward action list, log P_B of the path).

        Backward moves from a state: un-merge any tree-root lineage, or un-reticulate any
        reticulation whose two slots are both bare lineages.  Their number is |Pa(s)|.
        """
        # current "lineage set" as a list of (node, slot)
        cur = [(net.root, None)]
        moves = []            # backward order
        parents_num = []

        def is_free(entry):
            """a real node whose subtree contains no reticulation with a parent outside it"""
            v, slot = entry
            if slot is not None:
                return False
            sub = {u.idx for u in all_nodes(v)}
            for h in reticulation_nodes(v):
                if any(p is None or p.idx not in sub for p in h.parents):
                    return False
            return True

        while True:
            unmerge = [k for k, (v, slot) in enumerate(cur) if slot is None and v.kind == 'tree']
            bare_count = {}
            for k, (v, slot) in enumerate(cur):
                if slot is not None:
                    bare_count.setdefault(v.idx, []).append(k)
            n_free = sum(1 for e in cur if is_free(e))
            unret = [ks for ks in bare_count.values() if len(ks) == 2] if n_free >= 1 else []
            options = [('unmerge', k) for k in unmerge] + [('unret', ks) for ks in unret]
            if not options:
                break
            parents_num.append(len(options))
            mv = random.choice(options)
            if mv[0] == 'unmerge':
                k = mv[1]
                w = cur[k][0]
                new_entries = []
                for child in w.children:
                    if child.kind == 'ret' and w in child.parents:
                        # w filled a slot of this reticulation: the slot becomes bare again
                        slot = child.parents.index(w)
                        new_entries.append((child, slot))
                    else:
                        new_entries.append((child, None))
                # a tree node whose both children are the same reticulation cannot exist (masked)
                cur = cur[:k] + cur[k + 1:] + new_entries
                moves.append(('merge', w))
            else:
                ks = mv[1]
                h = cur[ks[0]][0]
                cur = [e for t, e in enumerate(cur) if t not in ks] + [(h.children[0], None)]
                moves.append(('ret', h))
        moves = moves[::-1]
        parents_num = parents_num[::-1]
        # ---- replay forward to obtain action indices in the sorted-state convention.
        # The replay creates NEW node objects, so original nodes are matched through `new_of`.
        state = self.get_initial_state()
        new_of = {l.node.idx: l.node for l in state.lineages}       # original idx -> replayed node
        actions = []
        for mv in moves:
            L = state.lineages
            if mv[0] == 'merge':
                w = mv[1]
                pos = []
                for child in w.children:
                    if child.kind == 'ret' and w in child.parents:
                        slot = child.parents.index(w)
                        p = next(t for t, l in enumerate(L) if l.is_bare and l.node is new_of[child.idx] and l.slot == slot)
                    else:
                        p = next(t for t, l in enumerate(L) if (not l.is_bare) and l.node is new_of[child.idx])
                    pos.append(p)
                (i, j) = sorted(pos)
                # edge lengths in (left = position i, right = position j) order
                d = {}
                for child, dist in zip(w.children, w.child_dists):
                    if child.kind == 'ret' and w in child.parents:
                        d[('s', child.idx, child.parents.index(w))] = dist
                    else:
                        d[('n', child.idx)] = dist
                orig_idx = {id(v): k for k, v in new_of.items()}

                def key_of(l):
                    return ('s', orig_idx[id(l.node)], l.slot) if l.is_bare else ('n', orig_idx[id(l.node)])
                edge = (d[key_of(L[i])], d[key_of(L[j])])
                a = {'type': 'merge', 'pair': self.pair_index[(i, j)], 'edge_action': edge}
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
        log_paths_pb = -np.log(np.array(parents_num, dtype=float))
        return actions, log_paths_pb

    # ------------------------------------------------------------------ scoring a given network
    def compute_network_log_score(self, net):
        """Recompute the terminal log score of a PhyloNetwork by replaying a backward-sampled path."""
        actions, _ = self.sample_backward_from_network(net)
        state = self.get_initial_state()
        feats = self.initial_features(1)[0]
        for a in actions:
            state, feats, log_score = self.apply_action(state, a, feats)
        return log_score
