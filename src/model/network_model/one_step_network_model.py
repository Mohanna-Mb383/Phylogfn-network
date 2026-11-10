"""
Policy network for PhyloGFN-Net (network counterpart of tree_topologies_model/one_step_model.py).

Differences from PhyloTreeModelOneStep:
  * the input set may contain up to l_max = N + R_MAX lineages and a per-trajectory
    padding mask (trajectories in a batch have different numbers of lineages);
  * every lineage token carries TWO Felsenstein features (versions A and B, see the
    environment) plus 4 flags (open? bare? side? inheritance share);
  * the action space is  [ all pairs (MERGE) | all singles (RETICULATE) ];  a second
    small MLP scores the singles;  illegal actions are masked to -inf with the mask
    computed by the environment (level-1 masks, R_MAX, no dead ends).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.model.mlp import MLP
from src.model.weight_init import trunc_normal_
from src.model.transformer import TransformerEncoder


class PhyloNetworkModelOneStep(nn.Module):

    def __init__(self, gfn_cfg):
        super().__init__()
        transformer_cfg = gfn_cfg.MODEL.TRANSFORMER
        self.concatenate_summary_token = transformer_cfg.PART1_HEAD.CONCATENATE_SUMMARY_TOKEN
        self.encoder = TransformerEncoder(transformer_cfg)
        self.seq_emb = MLP(transformer_cfg.SEQ_EMB)
        embedding_size = transformer_cfg.SEQ_EMB.OUTPUT_SIZE
        self.summary_token = nn.Parameter(torch.zeros(1, 1, embedding_size), requires_grad=True)
        trunc_normal_(self.summary_token, std=0.1)
        # input sizes of the action heads: pair / single embedding (E) [+ summary token (E)] [+ look-ahead: child
        # lineage embedding (E) + child aggregate embedding (E)]
        E = embedding_size
        self.lookahead = bool(gfn_cfg.MODEL.LOOKAHEAD.ENABLED)
        extra = (E if self.concatenate_summary_token else 0) + (2 * E if self.lookahead else 0)
        self.pair_logits_head = MLP(transformer_cfg.LOGITS_HEAD, input_size=E + extra)             # MERGE(i, j)
        self.single_logits_head = MLP(transformer_cfg.SINGLE_LOGITS_HEAD, input_size=E + extra)    # RETICULATE(i)
        # log F(s) head (DB / WDB).  Built on a forked CPU generator so that it consumes NO draws of the global
        # torch generator: the edge models, the log Z initialisation rollouts and the sampling that follow in
        # TBGFlowNetNet.__init__ see exactly the random stream of the original code (tools/regression_vs_baseline.py
        # checks this against the unpatched repository with the same seed).  Its own initialisation is derived from the
        # global seed, so different --seed values still give different flow heads.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((torch.initial_seed() + 0x466C6F77) % (2 ** 63 - 1))
            self.flow_head = MLP(gfn_cfg.MODEL.FLOW_HEAD, input_size=E)                            # log F(s) (DB / WDB)
        # the flow head is evaluated only when an objective needs it (or when a caller asks with need_log_flow=True):
        # under TB the forward pass does exactly the original computation
        self.need_log_flow = gfn_cfg.NET_LOSS != 'TB'
        self.logsoftmax = nn.LogSoftmax(dim=1)

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

    # ------------------------------------------------------------------ forward
    def forward(self, **kwargs):
        x_in = kwargs['batch_input']                      # [B, l_max, 2mc + 4]
        pad = kwargs['padding_mask']                      # [B, l_max]  True = padding
        action_mask = kwargs['action_mask']               # [B, n_pairs + l_max]  True = legal
        rows, cols = kwargs['pair_rows'], kwargs['pair_cols']
        random_spec = kwargs.get('random_spec', None)
        input_actions = kwargs.get('input_action_ids', None)

        B, l_max, _ = x_in.shape
        x0 = self.seq_emb(x_in)                                            # [B, l_max, E] token embeddings
        x = x0
        summary = self.summary_token.expand(B, -1, -1)
        x = torch.cat((summary, x), dim=1)
        key_padding_mask = F.pad(pad, (1, 0), 'constant', False)
        x = self.encoder(x, key_padding_mask)
        summary_token = x[:, :1]
        reps = x[:, 1:]                                                    # [B, l_max, E]
        log_flow = None                                                    # [B]  log F(s) (DB / WDB only)
        if kwargs.get('need_log_flow', self.need_log_flow):
            log_flow = self.flow_head(summary_token[:, 0]).squeeze(-1)

        # pairwise features e_i + e_j  (order-invariant, as in PhyloGFN)
        pairs = reps[:, rows] + reps[:, cols]                              # [B, n_pairs, E]
        singles = reps                                                     # [B, l_max, E]
        if self.concatenate_summary_token:
            pairs = torch.cat([pairs, summary_token.expand(-1, pairs.shape[1], -1)], dim=2)
            singles = torch.cat([singles, summary_token.expand(-1, l_max, -1)], dim=2)
        if self.lookahead:
            # look-ahead (da Silva et al. 2025, Eq. 7): embedding of the prospective child's new lineage(s) and of the
            # child's aggregated token set  A - e_i - e_j + e_child  (A = sum of the parent's token embeddings)
            child_pair = kwargs['child_pair_emb']                          # [B, n_pairs, E]
            child_single = kwargs['child_single_emb']                      # [B, l_max, E]
            x0d = x0.detach() if kwargs.get('lookahead_detach', True) else x0
            x0d = x0d.masked_fill(pad.unsqueeze(-1), 0.0)
            A = x0d.sum(1, keepdim=True)                                   # [B, 1, E]
            agg_pair = A - x0d[:, rows] - x0d[:, cols] + child_pair
            agg_single = A - x0d + child_single
            pairs = torch.cat([pairs, child_pair, agg_pair], dim=2)
            singles = torch.cat([singles, child_single, agg_single], dim=2)
        pair_logits = self.pair_logits_head(pairs).squeeze(-1)             # [B, n_pairs]
        single_logits = self.single_logits_head(singles).squeeze(-1)       # [B, l_max]
        logits = torch.cat([pair_logits, single_logits], dim=1)            # [B, n_pairs + l_max]
        logits = logits.masked_fill(~action_mask, float('-inf'))

        actions = self.sample(logits, action_mask, random_spec)
        if input_actions is not None:
            # replay-buffer trajectories: the action is given, we only need its log-probability
            given = input_actions >= 0
            actions = torch.where(given, input_actions, actions)
        log_p = self.logsoftmax(logits)
        log_paths_pf = log_p[torch.arange(B), actions]
        return {
            'logits': logits,
            'actions': actions,
            'log_paths_pf': log_paths_pf,
            'summary_reps': summary_token[:, 0],
            'lineage_reps': reps,
            'log_flow': log_flow,
        }

    def embed_tokens(self, tokens):
        """token embedding used for look-ahead child tokens (same seq_emb as the state tokens)"""
        return self.seq_emb(tokens)
