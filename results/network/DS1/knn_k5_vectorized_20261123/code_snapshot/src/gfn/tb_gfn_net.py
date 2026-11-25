"""
GFlowNet generator for PhyloGFN-Net (network counterpart of tb_gfn_phylo.py).

Objectives (GFN.LOSS_TYPE):
  TB     Trajectory balance (Malkin et al. 2022; PhyloGFN eq. 3), unchanged in form:
             L_TB(tau) = ( log Z + sum_t log P_F(s_{t+1}|s_t)  -  log R(x) - sum_t log P_B(s_t|s_{t+1}) )^2
  SUBTB  Subtrajectory balance (Madan et al. 2023), the proposal's training objective (RQ4):
         flow consistency on every sub-trajectory s_i -> ... -> s_j, weighted by λ_SubTB^(j-i),
             L = sum_{i<j} λ^(j-i) ( log F(s_i) + sum log P_F - log F(s_j) - sum log P_B )^2 / sum_{i<j} λ^(j-i)
         with log F(s_0) = log Z, log F(s_T) = log R(x) and a learned state-flow head for the
         intermediate states (which carry no reward: "the reward exists only at (G, b, θ)").

Backward policy (GFN.BACKWARD_MODELING):
  UNIFORM  P_B(s|s') = 1/|Pa(s')| (PhyloGFN);
  LEARNED  a softmax over the parents of s' scored from the state representation (proposal §3.3).

λ-conditioned policy (ENV.NETWORK.LAMBDA_CONDITIONING): the strength λ of the reticulation prior is
sampled per trajectory during training, enters the policy as an input and log Z(λ) = log Z + z(λ)
with a small head z; at inference λ is swept (proposal: "the data, not a hard cap, set the
reticulation complexity").
"""
import math
import numpy as np
import torch
from src.model.mlp import MLP
from src.model.network_model.one_step_network_model import PhyloNetworkModelOneStep
from src.model.edges_model.continuous.network_heads import MergeEdgeModel, ReticulationModel
from src.utils.lr_schedulers.build import build_scheduler

LOSS_FN = {'MSE': torch.nn.MSELoss(), 'HUBER': torch.nn.HuberLoss(delta=1.0)}


class TBGFlowNetNet(torch.nn.Module):

    def __init__(self, gfn_cfg, env, device, ddp=False):
        super().__init__()
        assert gfn_cfg.MODEL.EDGES_MODELING.DISTRIBUTION == 'MIXTURE', 'PhyloGFN-Net implements the continuous variant'
        assert gfn_cfg.LOSS_TYPE in ('TB', 'SUBTB')
        assert gfn_cfg.BACKWARD_MODELING in ('UNIFORM', 'LEARNED')
        self.gfn_model_cfg = gfn_model_cfg = gfn_cfg.MODEL
        self.env = env
        self.device = device
        self.parsimony_problem = False
        self.condition_on_scale = False
        self.loss_type = gfn_cfg.LOSS_TYPE
        self.learned_pb = gfn_cfg.BACKWARD_MODELING == 'LEARNED'
        self.subtb_lambda = float(gfn_model_cfg.SUBTB_LAMBDA)
        env.graph_inputs = bool(gfn_model_cfg.GRAPH_ENCODER.ENABLED)

        self.net_model = PhyloNetworkModelOneStep(gfn_cfg, env.cfg.ENV)
        self.edges_model = MergeEdgeModel(gfn_model_cfg.EDGES_MODELING.MIXTURE)
        self.ret_model = ReticulationModel(gfn_model_cfg.RETICULATION_MODELING)
        self.lambda_conditioning = env.lambda_conditioning
        self.z_head = None
        if self.lambda_conditioning:
            z_cfg = gfn_model_cfg.LAMBDA_EMB.clone()
            z_cfg.OUTPUT_SIZE = 1
            self.z_head = MLP(z_cfg)                                    # log Z(λ) = log Z + z(λ)

        # log Z initialised from random rollouts (as PhyloGFN does)
        trajs = env.sample(200, generate_full_trajectory=False)
        self.max_reward_seen = float(np.max([x.log_reward for x in trajs]))
        init_Z = self.max_reward_seen if gfn_model_cfg.Z_PARTITION_INIT == -1 else gfn_model_cfg.Z_PARTITION_INIT
        self._Z = torch.nn.Parameter(torch.ones(256, device=device) * init_Z / 256, requires_grad=gfn_model_cfg.UPDATE_Z)
        self.to(device)

        model_params = list(self.net_model.parameters()) + list(self.edges_model.parameters()) + list(self.ret_model.parameters())
        params = [{'params': model_params, 'lr': gfn_model_cfg.LR_MODEL}]
        z_params = [self._Z] if gfn_model_cfg.UPDATE_Z else []
        if self.z_head is not None:
            z_params += list(self.z_head.parameters())
        if z_params:
            params.append({'params': z_params, 'lr': gfn_model_cfg.LR_Z})
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

    def log_Z_of(self, lambdas):
        """per-trajectory log Z(λ) [B]"""
        base = self.compute_log_Z().reshape(1).to(lambdas.device)
        if self.z_head is None:
            return base.expand(lambdas.shape[0])
        return base + self.z_head(lambdas.reshape(-1, 1).float()).squeeze(-1)

    def reinit_log_Z(self, n_random=200):
        """re-estimate log Z from random rollouts (after a change of the reward function)"""
        trajs = self.env.sample(n_random, generate_full_trajectory=False)
        self.max_reward_seen = float(np.max([x.log_reward for x in trajs]))
        with torch.no_grad():
            torch.nn.init.constant_(self._Z, self.max_reward_seen / 256)

    def grad_norm(self, *_):
        return math.sqrt(sum(p.grad.norm().item() ** 2 for p in self.gradient_clipping_params if p.grad is not None))

    def param_norm(self, *_):
        return math.sqrt(sum(p.norm().item() ** 2 for p in self.gradient_clipping_params))

    def save(self, path):
        torch.save({'generator_state_dict': self.state_dict(), 'opt_state_dict': self.opt.state_dict()}, path)

    def load(self, path, strict=False):
        """checkpoints written before the proposal heads existed load with the new heads left at their
        initialisation (missing keys are reported); the optimiser state is restored when compatible"""
        d = torch.load(path, map_location='cpu')
        res = self.load_state_dict(d['generator_state_dict'], strict=strict)
        if res.missing_keys:
            print(f'[load] {len(res.missing_keys)} parameters not in the checkpoint (kept at init): '
                  + ', '.join(sorted({k.split(".")[0] + "." + k.split(".")[1] for k in res.missing_keys})))
        try:
            self.opt.load_state_dict(d['opt_state_dict'])
        except (ValueError, KeyError) as e:
            print(f'[load] optimiser state not restored ({type(e).__name__}); fine for sampling/evaluation')

    # ------------------------------------------------------------------ one policy step
    def forward(self, input_dict):
        """One forward step for the active trajectories of a batch.

        returns dict with 'actions' (list of env action dicts) and 'log_paths_pf' [B]
        (discrete choice + continuous parameters); with the learned backward policy also
        'log_pb_step' [B] (log P_B of the transition into the current state, 0 at s_0); with
        SubTB also 'log_flow' [B]."""
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
                ids[k] = env.action_id(a)
                if a['type'] == 'merge':
                    edge_given[k] = torch.tensor(a['edge_action'], device=dev)
                elif a['type'] == 'ret':
                    ret_given[k] = torch.tensor([a['edge_action'], a['theta']], device=dev)
        input_dict = dict(input_dict)
        input_dict['input_action_ids'] = ids
        ret = self.net_model(**input_dict)
        actions = ret['actions']
        log_pf = ret['log_paths_pf']
        reps = ret['lineage_reps']
        summary = ret['summary_reps']
        stop_id = input_dict.get('stop_id', -1)
        is_merge = actions < env.n_pairs
        is_stop = (actions == stop_id) if stop_id >= 0 else torch.zeros_like(is_merge)
        is_ret = (~is_merge) & (~is_stop)
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
            pair_np = pair.cpu().numpy()                 # [fast] one transfer instead of int(pair[t]) per trajectory
            for t, k in enumerate(sel.tolist()):
                out_actions[k] = {'type': 'merge', 'pair': int(pair_np[t]), 'edge_action': (float(edges_np[t, 0]), float(edges_np[t, 1]))}
        if is_ret.any():
            sel = torch.where(is_ret)[0]
            lineage = actions[sel] - env.n_pairs
            x_rep = reps[sel, lineage]
            b_hx, theta, lp = self.ret_model(summary[sel], x_rep, ret_given[sel])
            log_pf = log_pf.index_add(0, sel, lp)
            b_np, t_np = b_hx.detach().cpu().numpy(), theta.detach().cpu().numpy()
            lineage_np = lineage.cpu().numpy()           # [fast]
            for t, k in enumerate(sel.tolist()):
                out_actions[k] = {'type': 'ret', 'lineage': int(lineage_np[t]), 'theta': float(t_np[t]), 'edge_action': float(b_np[t])}
        for k in torch.where(is_stop)[0].tolist():
            out_actions[k] = {'type': 'stop'}
        out = {'actions': out_actions, 'log_paths_pf': log_pf, 'logits': ret['logits'],
               'summary_reps': summary, 'lineage_reps': reps}
        if self.learned_pb and 'backward_taken' in input_dict:
            out['log_pb_step'] = self._backward_step(summary, reps, input_dict['states'], input_dict['backward_taken'])
        if input_dict.get('need_flow', False):
            out['log_flow'] = self.net_model.log_flow(summary)
        return out

    def _backward_step(self, summary, reps, states, taken):
        B = reps.shape[0]
        out = torch.zeros(B, device=reps.device)
        idx = [k for k in range(B) if taken[k] is not None]
        if not idx:
            return out
        options = [states[k].backward_options() for k in idx]
        lp = self.net_model.backward_log_prob(summary[idx], reps[idx], options, [taken[k] for k in idx])
        return out.index_add(0, torch.tensor(idx, device=reps.device), lp)

    def backward_only(self, input_dict):
        """log P_B of the transition into each (terminal) state of input_dict [B]"""
        summary, reps = self.net_model.encode(input_dict)
        return self._backward_step(summary[:, 0], reps, input_dict['states'], input_dict['backward_taken'])

    # ------------------------------------------------------------------ losses
    def get_loss_from_rollout_outputs(self, rollout_outputs):
        if self.loss_type == 'SUBTB':
            return self.subtb_loss(rollout_outputs)
        log_pf = rollout_outputs['log_pf']              # [n_traj] already summed over steps
        log_pb = rollout_outputs['log_pb']
        log_rewards = rollout_outputs['log_rewards']
        log_z = self.log_Z_of(rollout_outputs['lambdas']).to(log_pf)
        return self.loss_fn(log_z + log_pf, log_rewards + log_pb)

    def subtb_loss(self, ro):
        """Subtrajectory balance with per-trajectory normalisation of the λ^(j-i) weights."""
        pf, pb, flow = ro['step_log_pf'], ro['step_log_pb'], ro['step_log_flow']      # [B, T]
        lengths = ro['lengths']
        B, T = pf.shape
        dev = pf.device
        log_z = self.log_Z_of(ro['lambdas']).to(pf)
        log_r = ro['log_rewards'].to(pf)
        # log F(s_t), t = 0..T: log Z, flow head for 0 < t < T_b, log R at t = T_b
        t_idx = torch.arange(T + 1, device=dev).unsqueeze(0).expand(B, -1)
        # flow[:, u] = log F(s_u) for the visited states u = 0..T_b-1; slot 0 <- log Z, slot T_b <- log R(x)
        logF = torch.cat([log_z.unsqueeze(1), flow[:, 1:], torch.zeros(B, 1, device=dev)], dim=1)   # [B, T+1]
        terminal = (t_idx == lengths.unsqueeze(1))
        logF = torch.where(terminal, log_r.unsqueeze(1).expand_as(logF), logF)
        cF = torch.cat([torch.zeros(B, 1, device=dev), torch.cumsum(pf, dim=1)], dim=1)   # C_F(t) = sum_{u<t} log P_F(u)
        cB = torch.cat([torch.zeros(B, 1, device=dev), torch.cumsum(pb, dim=1)], dim=1)
        A = logF - cF + cB                                              # residual of (i, j) = A(i) - A(j)
        valid = t_idx <= lengths.unsqueeze(1)                           # [B, T+1]
        diff = A.unsqueeze(2) - A.unsqueeze(1)                          # [B, T+1, T+1]
        span = (torch.arange(T + 1, device=dev).unsqueeze(0) - torch.arange(T + 1, device=dev).unsqueeze(1))   # j - i
        W = torch.where(span > 0, self.subtb_lambda ** span.clamp(min=0).float(), torch.zeros_like(span, dtype=torch.float))
        W = W.unsqueeze(0) * (valid.unsqueeze(2) & valid.unsqueeze(1)).float()
        per_traj = (W * diff ** 2).sum((1, 2)) / W.sum((1, 2)).clamp(min=1e-9)
        return per_traj.mean()

    def accumulate_loss(self, rollout_outputs, factor=1.0):
        loss = self.get_loss_from_rollout_outputs(rollout_outputs) / factor
        loss.backward()
        self.loss += loss

    def update_model(self):
        loss_value = self.loss.detach()
        loss_finite = bool(torch.isfinite(loss_value).all().item())

        bad_grad_names = []
        for name, p in self.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad_grad_names.append(name)

        if (not loss_finite) or bad_grad_names:
            print("\n[NONFINITE-UPDATE]")
            print("loss finite:", loss_finite)
            print("loss:", loss_value.detach().cpu().item()
                  if loss_value.numel() == 1 else loss_value.detach().cpu())
            print("non-finite gradients:", bad_grad_names[:30])
            print("SKIPPING optimizer step to protect finite model parameters")
            self.opt.zero_grad()
            self.loss = 0
            return {
                'grad_norm': float('nan'),
                'param_norm': self.param_norm(),
                'loss': float(loss_value.detach().cpu().item())
                        if loss_value.numel() == 1 else float('nan')
            }

        total_norm = torch.nn.utils.clip_grad_norm_(
            self.gradient_clipping_params,
            self.grad_clip
        )

        if not torch.isfinite(total_norm):
            print("\n[NONFINITE-UPDATE]")
            print("gradient norm is non-finite:", total_norm)
            print("SKIPPING optimizer step")
            self.opt.zero_grad()
            self.loss = 0
            return {
                'grad_norm': float('nan'),
                'param_norm': self.param_norm(),
                'loss': float(loss_value.detach().cpu().item())
            }

        info = {
            'grad_norm': float(total_norm.detach().cpu().item()),
            'param_norm': self.param_norm(),
            'loss': float(loss_value.detach().cpu().item())
        }

        self.opt.step()
        self.opt.zero_grad()
        self.loss = 0
        return info
