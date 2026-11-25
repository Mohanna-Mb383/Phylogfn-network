"""
Continuous heads of PhyloGFN-Net.

MergeEdgeModel        : the PhyloGFN edge MLP (Gaussian mixture over log branch lengths),
                        but able to serve, in ONE batch, both intermediate merges (two
                        edges) and last-step merges (one root edge) -- necessary because
                        trajectories of a batch no longer share the same step index.
ReticulationModel     : new head for the RETICULATE action.  It samples the pair
                        (log b_hx, logit theta_h) from a 2-D diagonal Gaussian mixture,
                        with the change-of-variable corrections
                           log p(b)     = log p(log b)   - log b
                           log p(theta) = log p(logit theta) - log theta - log(1 - theta)
"""
import copy
import torch
import torch.nn as nn
from torch.distributions import Normal, MixtureSameFamily, Independent, Categorical
from src.model.mlp import MLP
from src.model.edges_model.continuous.continuous_mixture import MixtureGaussianModel


class MergeEdgeModel(nn.Module):

    def __init__(self, mixture_cfg):
        super().__init__()
        self.root_edge_model = MixtureGaussianModel(mixture_cfg, True)     # one edge (b_l + b_r)
        self.lr_model = MixtureGaussianModel(mixture_cfg, False)           # two independent edges

    def forward(self, summary_reps, left_reps, right_reps, last_step, input_edge_actions=None):
        """returns edge_actions [B, 2] (branch lengths, l/r) and log_paths_pf [B]"""
        B = summary_reps.shape[0]
        dev = summary_reps.device
        rep = torch.cat([summary_reps, left_reps, right_reps], dim=1)
        edge_actions = torch.zeros(B, 2, device=dev)
        log_pf = torch.zeros(B, device=dev)
        for mode in (True, False):
            sel = (last_step == mode)
            if sel.sum() == 0:
                continue
            model = self.root_edge_model if mode else self.lr_model
            dist = model(rep[sel])['dist']
            log_action = dist.sample().clip(-10, 0)                     # log branch length(s)
            if input_edge_actions is not None:
                given = input_edge_actions[sel]
                if mode:
                    forced = torch.log(given.sum(dim=-1))                    # root: total length
                else:
                    forced = torch.log(given)
                has = ~torch.isnan(given[:, 0])
                log_action = torch.where(has.unsqueeze(-1) if not mode else has, forced, log_action)
            lp = dist.log_prob(log_action)
            if mode:
                lp = lp - log_action                                      # change of variable b = exp(.)
                lengths = torch.exp(log_action)
                edge_actions[sel] = torch.stack([lengths / 2, lengths / 2], dim=1)
            else:
                lp = lp - log_action.sum(dim=1)
                edge_actions[sel] = torch.exp(log_action)
            log_pf[sel] = lp
        return edge_actions, log_pf


class ReticulationModel(nn.Module):

    def __init__(self, ret_cfg):
        super().__init__()
        ret_cfg = copy.deepcopy(ret_cfg)
        self.K = ret_cfg.NB_COMPONENTS
        self.logit_range = ret_cfg.THETA_LOGIT_RANGE
        ret_cfg.HEAD.OUTPUT_SIZE = self.K * 5        # [mix logits K] [means 2K] [log vars 2K]
        self.model = MLP(ret_cfg.HEAD)

    def distribution(self, rep):
        out = self.model(rep)
        K = self.K
        mix = out[:, :K]
        mean = out[:, K:3 * K].reshape(-1, K, 2)
        mean_b = (torch.tanh(mean[..., 0]) + 1) / 2 * 10 - 10                    # log b in [-10, 0]
        mean_t = torch.tanh(mean[..., 1]) * self.logit_range                     # logit theta in [-r, r]
        mean = torch.stack([mean_b, mean_t], dim=-1)
        std = out[:, 3 * K:].exp().reshape(-1, K, 2) + 1e-3
        return MixtureSameFamily(Categorical(logits=mix), Independent(Normal(mean, std), 1))

    def forward(self, summary_reps, lineage_reps, input_ret_actions=None):
        """returns b_hx [B], theta [B], log_paths_pf [B]"""
        rep = torch.cat([summary_reps, lineage_reps], dim=1)
        dist = self.distribution(rep)
        z = dist.sample()
        z = torch.stack([z[:, 0].clip(-10, 0), z[:, 1].clip(-self.logit_range - 2, self.logit_range + 2)], dim=1)
        if input_ret_actions is not None:
            given = input_ret_actions                                            # [B, 2] = (b_hx, theta)
            has = ~torch.isnan(given[:, 0])
            forced = torch.stack([torch.log(given[:, 0]), torch.logit(given[:, 1].clamp(1e-6, 1 - 1e-6))], dim=1)
            z = torch.where(has.unsqueeze(-1), forced, z)
        lp = dist.log_prob(z)
        b_hx = torch.exp(z[:, 0])
        theta = torch.sigmoid(z[:, 1])
        lp = lp - z[:, 0] - torch.log(theta) - torch.log(1 - theta)
        return b_hx, theta, lp
