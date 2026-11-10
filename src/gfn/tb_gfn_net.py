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

        # objectives and policy features after da Silva et al. (ICLR 2025); 'TB' reproduces the original behaviour
        self.net_loss = gfn_cfg.NET_LOSS
        assert self.net_loss in ('TB', 'DB', 'WDB'), self.net_loss
        self.wdb_gamma = gfn_cfg.WDB_GAMMA
        self.lookahead_cfg = gfn_model_cfg.LOOKAHEAD

        # the flow head (log F(s)) is trained by DB / WDB only; under TB it is left out of the optimiser and of the
        # gradient-clipping list, so the optimiser state, the clipping and the reported parameter norm are those of the
        # original code (tools/regression_vs_baseline.py)
        flow_ids = {id(p) for p in self.net_model.flow_head.parameters()}
        net_params = [p for p in self.net_model.parameters() if self.net_loss != 'TB' or id(p) not in flow_ids]
        model_params = net_params + list(self.edges_model.parameters()) + list(self.ret_model.parameters())
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
        if self.net_loss == 'TB':
            with torch.no_grad():
                return self.compute_log_Z().item()
        # DB / WDB: the flow of the initial state plays the role of Z.  The forward pass samples an action, so it is
        # run on forked generators: a progress-bar read-out must not change the training random stream.
        devices = [self.device] if str(self.device).startswith('cuda') else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            env = self.env
            s0 = env.get_initial_state()
            feats = env.initial_features(1)
            inp = env.prepare_rollout_inputs(feats, [s0], None)
            inp['need_log_flow'] = True
            return self.forward(inp)['log_flow'][0].item()

    def grad_norm(self, *_):
        return math.sqrt(sum(p.grad.norm().item() ** 2 for p in self.gradient_clipping_params if p.grad is not None))

    def param_norm(self, *_):
        return math.sqrt(sum(p.norm().item() ** 2 for p in self.gradient_clipping_params))

    def save(self, path):
        torch.save({'generator_state_dict': self.state_dict(), 'opt_state_dict': self.opt.state_dict()}, path)

    def load(self, path):
        d = torch.load(path, map_location='cpu')
        # checkpoints written before the flow head existed (TB runs of the baseline code) lack its parameters:
        # accept them (the flow head is untrained and unused by TB); refuse any other mismatch
        missing, unexpected = self.load_state_dict(d['generator_state_dict'], strict=False)
        bad = [k for k in missing if not k.startswith('net_model.flow_head.')] + list(unexpected)
        if bad:
            raise RuntimeError('checkpoint does not match the model: %s' % bad)
        try:
            self.opt.load_state_dict(d['opt_state_dict'])
        except (ValueError, KeyError) as exc:          # optimiser state of a model without the flow head
            if not missing:
                raise
            print('warning: optimiser state not loaded (%s); fine for evaluation, not for resuming training' % exc)

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
        if self.lookahead_cfg.ENABLED:
            la = self.lookahead_cfg
            if la.DETACH:
                with torch.no_grad():
                    pe, se = env.lookahead_child_embeddings(input_dict['feats'], input_dict['states'], self.net_model.embed_tokens,
                                                            la.B_HAT, la.THETA_HAT)
            else:
                pe, se = env.lookahead_child_embeddings(input_dict['feats'], input_dict['states'], self.net_model.embed_tokens,
                                                        la.B_HAT, la.THETA_HAT)
            input_dict['child_pair_emb'] = pe
            input_dict['child_single_emb'] = se
            input_dict['lookahead_detach'] = bool(la.DETACH)
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
        log_flow = ret['log_flow']                       # None unless the flow head was evaluated (DB / WDB, or need_log_flow)
        if log_flow is not None:
            log_flow = log_flow + self.compute_log_Z().to(log_flow)
        return {'actions': out_actions, 'log_paths_pf': log_pf, 'logits': ret['logits'], 'log_flow': log_flow}

    # ------------------------------------------------------------------ TB loss
    def get_loss_from_rollout_outputs(self, rollout_outputs):
        if self.net_loss == 'TB':
            log_pf = rollout_outputs['log_pf']              # [n_traj] already summed over steps
            log_pb = rollout_outputs['log_pb']
            log_rewards = rollout_outputs['log_rewards']
            log_z = self.compute_log_Z().reshape(-1).to(log_pf)
            return self.loss_fn(log_z + log_pf, log_rewards + log_pb)
        return self.db_loss(rollout_outputs)

    def transition_residuals(self, rollout_outputs):
        """e_t = log F(s_t) + log P_F(s_{t+1}|s_t) - log F(s_{t+1}) - log P_B(s_t|s_{t+1}) for every recorded transition,
        with F(x) = R(x) hard-wired at terminal states (log_rewards).  returns e [B, T], mask [B, T], gamma [B, T]."""
        lpf = rollout_outputs['step_log_pf']            # [B, T]  log P_F(s_{t+1} | s_t), with grad
        lF = rollout_outputs['step_log_flow']           # [B, T]  log F(s_t), with grad
        lpb = rollout_outputs['step_log_pb'].to(lpf)    # [B, T]  log P_B(s_t | s_{t+1})
        mask = rollout_outputs['step_mask'].to(lpf)     # [B, T]  1 for recorded transitions
        is_last = rollout_outputs['step_is_last'].to(lpf)
        log_r = rollout_outputs['log_rewards'].to(lpf)  # [B]
        lF_next = torch.cat([lF[:, 1:], torch.zeros_like(lF[:, :1])], dim=1)
        lF_next = torch.where(is_last > 0, log_r.unsqueeze(1).expand_as(lF_next), lF_next)
        e = (lF + lpf - lF_next - lpb) * mask
        # gamma is kept in float64: the 'count' weights reach exp(-90) (DS1) .. exp(-259) (DS8) at the root, below the
        # float32 subnormal range (1e-45) for N >= 29; in float64 they are representable (min normal 2e-308).  Their
        # share of the normalised loss is still 1e-40 .. 1e-113: a negligible gradient, which is why 'count' is not
        # the default at DS scale (see src/configs/defaults.py).
        gamma = rollout_outputs['step_gamma'].to(lpf.device, torch.float64) * mask.double() if self.net_loss == 'WDB' else mask.double()
        return e, mask, gamma

    def db_loss(self, rollout_outputs):
        """DB  (paper Eq. 1): per-trajectory mean of e_t^2;  WDB (paper Eq. 5-6): per-trajectory sum_t gamma_t e_t^2 / sum_t gamma_t
        (accumulated in float64), then the mean over the batch, returned in the model's dtype."""
        e, mask, gamma = self.transition_residuals(rollout_outputs)
        per_traj = (gamma * e.double() ** 2).sum(1) / gamma.sum(1).clamp_min(1e-300)
        return per_traj.mean().to(e.dtype)

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
