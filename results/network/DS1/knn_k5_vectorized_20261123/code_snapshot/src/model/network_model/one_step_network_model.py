"""
Policy network for PhyloGFN-Net (network counterpart of tree_topologies_model/one_step_model.py).

Differences from PhyloTreeModelOneStep:
  * the input set may contain up to l_max = N + R_MAX lineages and a per-trajectory
    padding mask (trajectories in a batch have different numbers of lineages);
  * every lineage token carries TWO Felsenstein features (versions A and B, see the
    environment) plus 4 flags (open? bare? side? inheritance share);
  * the action space is  [ all pairs (MERGE) | all singles (RETICULATE) | STOP ];  a second
    small MLP scores the singles, a scalar head the STOP;  illegal actions are masked to -inf
    with the mask computed by the environment (level-1 masks, R_MAX, no dead ends).

Proposal additions (each switched on by configuration, off = the PhyloGFN-Net behaviour):
  * k-NN candidate filtering (KNN.ENABLED): the pair MLP scores ONLY the candidate pairs
    (k nearest lineages of every open lineage, computed by the environment), O(l k) instead of
    O(l^2) pair scorings; the forward policy is the proposal's mixture
        P_F(a|s) = (1 - ε) P_kNN(a|s) + ε U(a|s),
    P_kNN = softmax over the candidates, U = uniform over the legal actions, so that every legal
    action keeps a non-zero probability (full support) and log P_F is finite for replayed actions.
  * λ-conditioned policy (LAMBDA_CONDITIONING): an embedding of the trajectory's λ is added
    to every token, so one network represents the family of posteriors p(G, b, θ | Y, λ).
  * graph encoder (GRAPH_ENCODER.ENABLED): message passing over the partial network's nodes and
    edges, added to the lineage tokens (explicit topology encoding).
  * state-flow head (SubTB) and backward head (learned P_B over the parents of a state).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.model.mlp import MLP
from src.model.weight_init import trunc_normal_
from src.model.transformer import TransformerEncoder
from src.model.network_model.graph_encoder import GraphEncoder
from src.env.network_graph_batch import NODE_FEATURES


class PhyloNetworkModelOneStep(nn.Module):

    def __init__(self, gfn_cfg, env_cfg=None):
        super().__init__()
        transformer_cfg = gfn_cfg.MODEL.TRANSFORMER
        self.concatenate_summary_token = transformer_cfg.PART1_HEAD.CONCATENATE_SUMMARY_TOKEN
        self.encoder = TransformerEncoder(transformer_cfg)
        self.seq_emb = MLP(transformer_cfg.SEQ_EMB)
        E = embedding_size = transformer_cfg.SEQ_EMB.OUTPUT_SIZE
        self.summary_token = nn.Parameter(torch.zeros(1, 1, embedding_size), requires_grad=True)
        trunc_normal_(self.summary_token, std=0.1)
        self.pair_logits_head = MLP(transformer_cfg.LOGITS_HEAD)             # MERGE(i, j)
        self.single_logits_head = MLP(transformer_cfg.SINGLE_LOGITS_HEAD)    # RETICULATE(i)
        self.stop_head = nn.Linear(E, 1)                                     # STOP (forced when legal)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        # ---- proposal additions ----
        net_cfg = env_cfg.NETWORK if env_cfg is not None else None
        self.knn = bool(net_cfg.KNN.ENABLED) if net_cfg is not None else False
        self.lambda_conditioning = bool(net_cfg.LAMBDA_CONDITIONING) if net_cfg is not None else False
        self.lambda_emb = MLP(gfn_cfg.MODEL.LAMBDA_EMB) if self.lambda_conditioning else None
        g_cfg = gfn_cfg.MODEL.GRAPH_ENCODER
        self.graph_encoder = GraphEncoder(NODE_FEATURES, g_cfg.HIDDEN_SIZE, g_cfg.LAYERS, E) if g_cfg.ENABLED else None
        self.flow_head = MLP(gfn_cfg.MODEL.STATE_FLOW_HEAD)                  # log F(s) for SubTB
        self.backward_head = MLP(gfn_cfg.MODEL.BACKWARD_HEAD)                # learned P_B over parents

    def model_params(self):
        return list(self.parameters())

    # ------------------------------------------------------------------ sampling
    def sample(self, logits, action_mask, random_spec):
        """logits already masked (-inf on illegal).  epsilon-greedy picks a uniformly random LEGAL action."""
        if random_spec is None:
            random_spec = {'random_action_prob': 0.0}
        if 'random_action_prob' in random_spec:
            actions = Categorical(logits=logits).sample()
            p = random_spec['random_action_prob']
            if p > 0:
                B = logits.shape[0]
                rand_flag = (torch.empty(B).uniform_(0, 1) <= p).to(logits.device)
                if rand_flag.any():
                    uniform_logits = torch.where(action_mask, torch.zeros_like(logits), torch.full_like(logits, float('-inf')))
                    rand_actions = Categorical(logits=uniform_logits).sample()
                    actions = torch.where(rand_flag, rand_actions, actions)
        else:
            T = random_spec['T']
            actions = Categorical(logits=logits / T).sample()
        return actions

    # ------------------------------------------------------------------ encoder
    def encode(self, kwargs):
        """tokens -> (summary [B, E], reps [B, l_max, E])"""
        x_in = kwargs['batch_input']                      # [B, l_max, 2mc + 4]
        pad = kwargs['padding_mask']                      # [B, l_max]  True = padding
        B, l_max, _ = x_in.shape
        x = self.seq_emb(x_in)                                             # [B, l_max, E]
        if self.graph_encoder is not None:
            g = self.graph_encoder(kwargs['graph_x'], kwargs['graph_adj'], kwargs['graph_node_mask'])   # [B, V, E]
            idx = kwargs['graph_lineage_node'].unsqueeze(-1).expand(-1, -1, g.shape[-1])
            x = x + torch.gather(g, 1, idx)                                # explicit topology information per lineage
        summary = self.summary_token.expand(B, -1, -1)
        if self.lambda_conditioning:
            lam = self.lambda_emb(kwargs['lambda'].reshape(B, 1)).unsqueeze(1)   # [B, 1, E]
            x = x + lam
            summary = summary + lam
        x = torch.cat((summary, x), dim=1)
        key_padding_mask = F.pad(pad, (1, 0), 'constant', False)
        x = self.encoder(x, key_padding_mask)
        return x[:, :1], x[:, 1:]

    # ------------------------------------------------------------------ action logits
    def action_logits(self, summary_token, reps, kwargs):
        """logits over [pairs | singles | STOP], -inf on illegal actions.  With k-NN filtering the
        pair MLP runs on the candidate pairs only."""
        action_mask = kwargs['action_mask']               # [B, n_actions]  True = legal
        rows, cols = kwargs['pair_rows'], kwargs['pair_cols']
        n_pairs = kwargs['n_pairs']
        B, l_max, E = reps.shape
        dev = reps.device
        pair_mask = action_mask[:, :n_pairs]
        if self.knn:
            pair_mask = pair_mask & kwargs['candidate_mask'][:, :n_pairs]
        b_idx, p_idx = torch.where(pair_mask)                              # candidate (and legal) pairs only
        feats = reps[b_idx, rows[p_idx]] + reps[b_idx, cols[p_idx]]       # order-invariant pair feature e_i + e_j
        if self.concatenate_summary_token:
            feats = torch.cat([feats, summary_token[b_idx, 0]], dim=1)
        pair_logits = torch.full((B, n_pairs), float('-inf'), device=dev)
        if feats.shape[0] > 0:
            pair_logits[b_idx, p_idx] = self.pair_logits_head(feats).squeeze(-1)
        singles = reps
        if self.concatenate_summary_token:
            singles = torch.cat([singles, summary_token.expand(-1, l_max, -1)], dim=2)
        single_logits = self.single_logits_head(singles).squeeze(-1)       # [B, l_max]
        parts = [pair_logits, single_logits]
        if kwargs.get('stop_id', -1) >= 0:
            parts.append(self.stop_head(summary_token[:, 0]))              # [B, 1]
        logits = torch.cat(parts, dim=1)
        return logits.masked_fill(~action_mask, float('-inf'))

    # ------------------------------------------------------------------ forward
    def forward(self, **kwargs):
        action_mask = kwargs['action_mask']
        random_spec = kwargs.get('random_spec', None)
        input_actions = kwargs.get('input_action_ids', None)
        summary_token, reps = self.encode(kwargs)
        B = reps.shape[0]
        logits = self.action_logits(summary_token, reps, kwargs)           # candidate-restricted, legal-masked

        if self.knn:
            # proposal sampling rule: with probability ε a uniform legal action, else P_kNN
            eps = float(kwargs['eps'])
            n_legal = action_mask.sum(1).clamp(min=1).to(logits.dtype)
            log_uniform = torch.where(action_mask, -torch.log(n_legal).unsqueeze(1).expand_as(logits),
                                      torch.full_like(logits, float('-inf')))
            log_knn = self.logsoftmax(logits)                              # -inf outside the candidates
            # finite floor instead of -inf: logaddexp(-inf, -inf) has a NaN gradient that 0 * NaN would spread
            log_p = torch.logaddexp(math.log(1.0 - eps) + log_knn.clamp(min=-1e4),
                                    math.log(eps) + log_uniform.clamp(min=-1e4))
            log_p = log_p.masked_fill(~action_mask, float('-inf'))
            if random_spec is not None and 'T' in random_spec:
                actions = Categorical(logits=logits / random_spec['T']).sample()
            else:
                use_uniform = (torch.empty(B).uniform_(0, 1) < eps).to(logits.device)
                # ---- diagnostic for late-training NaN / empty k-NN support ----
                score_mask = kwargs['candidate_mask'] & action_mask

                bad_scored = (~torch.isfinite(logits)) & score_mask
                empty_rows = ~((torch.isfinite(logits) & score_mask).any(dim=1))

                if bad_scored.any() or empty_rows.any():
                    print("\n[NaN-DIAG]")
                    print("raw logits contain NaN:", torch.isnan(logits).any().item())
                    print("raw logits contain +Inf:", torch.isposinf(logits).any().item())
                    print("scored non-finite rows:",
                          torch.where(bad_scored.any(dim=1))[0].detach().cpu().tolist())
                    print("empty candidate rows:",
                          torch.where(empty_rows)[0].detach().cpu().tolist())
                    print("legal actions per row:",
                          action_mask.sum(dim=1).detach().cpu().tolist()[:20])
                    print("candidate legal actions per row:",
                          score_mask.sum(dim=1).detach().cpu().tolist()[:20])
                    print("summary finite:", torch.isfinite(summary_token).all().item())
                    print("reps finite:", torch.isfinite(reps).all().item())

                    bad_params = [
                        n for n, p in self.named_parameters()
                        if p.is_floating_point() and not torch.isfinite(p).all()
                    ]
                    print("non-finite parameters:", bad_params[:20])
                    raise RuntimeError("NAN_DIAGNOSTIC_TRIGGERED")

                a_knn = Categorical(logits=logits).sample()
                a_uni = Categorical(logits=log_uniform).sample()
                actions = torch.where(use_uniform, a_uni, a_knn)
        else:
            actions = self.sample(logits, action_mask, random_spec)
            log_p = self.logsoftmax(logits)
        if input_actions is not None:
            # replay-buffer trajectories: the action is given, we only need its log-probability
            given = input_actions >= 0
            actions = torch.where(given, input_actions, actions)
        log_paths_pf = log_p[torch.arange(B), actions]
        return {
            'logits': logits,
            'actions': actions,
            'log_paths_pf': log_paths_pf,
            'summary_reps': summary_token[:, 0],
            'lineage_reps': reps,
        }

    # ------------------------------------------------------------------ SubTB: state flow
    def log_flow(self, summary_reps):
        return self.flow_head(summary_reps).squeeze(-1)                    # [B]

    # ------------------------------------------------------------------ learned backward policy
    def backward_log_prob(self, summary_reps, reps, options, taken):
        """options: list (per batch element) of (kind, position) parents; taken: list of the index of the
        parent actually visited.  Returns log P_B(taken | state) [B] under a softmax over the parents.
        kind 'merge' -> un-merge the tree-root lineage at `position`; 'ret' -> un-reticulate the
        reticulation whose slot-0 lineage is at `position`; 'stop' -> the single parent of a stopped state."""
        B, l_max, E = reps.shape
        dev = reps.device
        n_opt = max(1, max(len(o) for o in options))
        pos = torch.zeros(B, n_opt, dtype=torch.long, device=dev)
        kind = torch.zeros(B, n_opt, 2, device=dev)
        valid = torch.zeros(B, n_opt, dtype=torch.bool, device=dev)
        for b, opts in enumerate(options):
            for t, (k, p) in enumerate(opts):
                pos[b, t] = p
                kind[b, t, 0 if k == 'merge' else 1] = 1.0
                valid[b, t] = True
        gathered = torch.gather(reps, 1, pos.unsqueeze(-1).expand(-1, -1, E))                 # [B, n_opt, E]
        feats = torch.cat([gathered, summary_reps.unsqueeze(1).expand(-1, n_opt, -1), kind], dim=-1)
        logits = self.backward_head(feats).squeeze(-1).masked_fill(~valid, float('-inf'))
        log_p = F.log_softmax(logits, dim=1)
        idx = torch.tensor(taken, dtype=torch.long, device=dev)
        return log_p[torch.arange(B, device=dev), idx]
