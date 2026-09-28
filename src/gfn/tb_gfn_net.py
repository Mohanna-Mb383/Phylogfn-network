"""
Trajectory-balance GFlowNet for PhyloGFN-Net (network counterpart of tb_gfn_phylo.py).

The objective is unchanged (Malkin et al. 2022; PhyloGFN eq. 3):
    L_TB(tau) = ( log Z + sum_t log P_F(s_{t+1}|s_t)  -  log R(x) - sum_t log P_B(s_t|s_{t+1}) )^2
with a uniform backward policy P_B(s|s') = 1/|Pa(s')|.  What changes is only what a
step is: a step is now MERGE (pair + two lengths) or RETICULATE (lineage + theta + one
length), and trajectories have different lengths, so the per-trajectory sums are
accumulated by the rollout worker instead of being stacked into a fixed table.
"""
import math
import numpy as np
import torch
from src.model.network_model.one_step_network_model import PhyloNetworkModelOneStep
from src.model.edges_model.continuous.network_heads import MergeEdgeModel, ReticulationModel
from src.utils.lr_schedulers.build import build_scheduler

LOSS_FN = {'MSE': torch.nn.MSELoss(), 'HUBER': torch.nn.HuberLoss(delta=1.0)}


class TBGFlowNetNet(torch.nn.Module):

    def __init__(self, gfn_cfg, env, device, ddp=False):
        super().__init__()
        assert gfn_cfg.MODEL.EDGES_MODELING.DISTRIBUTION == 'MIXTURE', 'PhyloGFN-Net implements the continuous variant'
        self.gfn_model_cfg = gfn_model_cfg = gfn_cfg.MODEL
        self.env = env
        self.device = device
        self.parsimony_problem = False
        self.condition_on_scale = False

        self.net_model = PhyloNetworkModelOneStep(gfn_cfg)
        self.edges_model = MergeEdgeModel(gfn_model_cfg.EDGES_MODELING.MIXTURE)
        self.ret_model = ReticulationModel(gfn_model_cfg.RETICULATION_MODELING)

        # log Z initialised from random rollouts (as PhyloGFN does)
        trajs = env.sample(200, generate_full_trajectory=False)
        self.max_reward_seen = float(np.max([x.log_reward for x in trajs]))
        init_Z = self.max_reward_seen if gfn_model_cfg.Z_PARTITION_INIT == -1 else gfn_model_cfg.Z_PARTITION_INIT
        self._Z = torch.nn.Parameter(torch.ones(256, device=device) * init_Z / 256, requires_grad=gfn_model_cfg.UPDATE_Z)
        self.to(device)

        model_params = list(self.net_model.parameters()) + list(self.edges_model.parameters()) + list(self.ret_model.parameters())
        params = [{'params': model_params, 'lr': gfn_model_cfg.LR_MODEL}]
        if gfn_model_cfg.UPDATE_Z:
            params.append({'params': [self._Z], 'lr': gfn_model_cfg.LR_Z})
        self.gradient_clipping_params = model_params
        self.grad_clip = gfn_model_cfg.GRAD_CLIP
        self.opt = torch.optim.Adam(params, weight_decay=gfn_model_cfg.L2_REG, betas=(0.9, 0.999), amsgrad=True)
        self.scheduler = build_scheduler(self.opt, gfn_cfg.MODEL.LR_SCHEDULER) if gfn_cfg.MODEL.USE_LR_SCHEDULER else None
        self.loss_fn = LOSS_FN[gfn_model_cfg.LOSS_FN]
        self.loss = 0

    # ------------------------------------------------------------------ Z
    def compute_log_Z(self, scale_key=None):
        return self._Z.sum()

    def log_Z(self, scale_key=None):
        with torch.no_grad():
            return self.compute_log_Z().item()

    def grad_norm(self, *_):
        return math.sqrt(sum(p.grad.norm().item() ** 2 for p in self.gradient_clipping_params if p.grad is not None))

    def param_norm(self, *_):
        return math.sqrt(sum(p.norm().item() ** 2 for p in self.gradient_clipping_params))

    def save(self, path):
        torch.save({'generator_state_dict': self.state_dict(), 'opt_state_dict': self.opt.state_dict()}, path)

    def load(self, path):
        d = torch.load(path, map_location='cpu')
        self.load_state_dict(d['generator_state_dict'])
        self.opt.load_state_dict(d['opt_state_dict'])

    # ------------------------------------------------------------------ one policy step
    def forward(self, input_dict):
        """One forward step for the active trajectories of a batch.

        returns dict with 'actions' (list of env action dicts) and 'log_paths_pf' [B]
        (discrete choice + continuous parameters)."""
        env = self.env
        B = input_dict['batch_size']
        dev = input_dict['batch_input'].device
        input_actions = input_dict.get('input_actions', None)
        ids = torch.full((B,), -1, dtype=torch.long, device=dev)
        edge_given = torch.full((B, 2), float('nan'), device=dev)
        ret_given = torch.full((B, 2), float('nan'), device=dev)
        if input_actions is not None:
            for k, a in enumerate(input_actions):
                if a is None:
                    continue
                ids[k] = env.action_id(a, env.n_pairs)
                if a['type'] == 'merge':
                    edge_given[k] = torch.tensor(a['edge_action'], device=dev)
                else:
                    ret_given[k] = torch.tensor([a['edge_action'], a['theta']], device=dev)
        input_dict = dict(input_dict)
        input_dict['input_action_ids'] = ids
        ret = self.net_model(**input_dict)
        actions = ret['actions']
        log_pf = ret['log_paths_pf']
        reps = ret['lineage_reps']
        summary = ret['summary_reps']
        is_merge = actions < env.n_pairs
        out_actions = [None] * B

        if is_merge.any():
            sel = torch.where(is_merge)[0]
            pair = actions[sel]
            rows = input_dict['pair_rows'][pair]
            cols = input_dict['pair_cols'][pair]
            left = reps[sel, rows]
            right = reps[sel, cols]
            last = input_dict['last_step'][sel]
            edges, lp = self.edges_model(summary[sel], left, right, last, edge_given[sel])
            log_pf = log_pf.index_add(0, sel, lp)
            edges_np = edges.detach().cpu().numpy()
            for t, k in enumerate(sel.tolist()):
                out_actions[k] = {'type': 'merge', 'pair': int(pair[t]), 'edge_action': (float(edges_np[t, 0]), float(edges_np[t, 1]))}
        if (~is_merge).any():
            sel = torch.where(~is_merge)[0]
            lineage = actions[sel] - env.n_pairs
            x_rep = reps[sel, lineage]
            b_hx, theta, lp = self.ret_model(summary[sel], x_rep, ret_given[sel])
            log_pf = log_pf.index_add(0, sel, lp)
            b_np, t_np = b_hx.detach().cpu().numpy(), theta.detach().cpu().numpy()
            for t, k in enumerate(sel.tolist()):
                out_actions[k] = {'type': 'ret', 'lineage': int(lineage[t]), 'theta': float(t_np[t]), 'edge_action': float(b_np[t])}
        return {'actions': out_actions, 'log_paths_pf': log_pf, 'logits': ret['logits']}

    # ------------------------------------------------------------------ TB loss
    def get_loss_from_rollout_outputs(self, rollout_outputs):
        log_pf = rollout_outputs['log_pf']              # [n_traj] already summed over steps
        log_pb = rollout_outputs['log_pb']
        log_rewards = rollout_outputs['log_rewards']
        log_z = self.compute_log_Z().reshape(-1).to(log_pf)
        return self.loss_fn(log_z + log_pf, log_rewards + log_pb)

    def accumulate_loss(self, rollout_outputs, factor=1.0):
        loss = self.get_loss_from_rollout_outputs(rollout_outputs) / factor
        loss.backward()
        self.loss += loss

    def update_model(self):
        info = {'grad_norm': self.grad_norm(), 'param_norm': self.param_norm(),
                'loss': self.loss.detach().cpu().numpy().tolist()}
        torch.nn.utils.clip_grad_norm_(self.gradient_clipping_params, self.grad_clip)
        self.opt.step()
        self.opt.zero_grad()
        self.loss = 0
        return info
