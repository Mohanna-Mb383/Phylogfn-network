"""
Rollout worker for PhyloGFN-Net (network counterpart of rollout_worker_phylo.py).

PhyloGFN steps all trajectories of a batch in lock-step because every tree trajectory has
exactly N - 1 merges.  A network trajectory has N - 1 + 2R steps with R chosen by the
policy, so trajectories finish at different times: we keep a `done` mask, run the policy
only on the active trajectories, and accumulate  sum log P_F  and  sum log P_B  per
trajectory as we go (P_B is uniform: log P_B = -log |Pa(s')| of every visited state s').

record_steps = True (DB / WDB objectives, da Silva et al. 2025) additionally keeps, per transition
(s_t -> s_{t+1}): log P_F(s_{t+1}|s_t) and log F(s_t) with gradients, log P_B(s_t|s_{t+1}), the WDB
weight gamma of the transition (from the child state), and descriptors used by the residual profile
(depth, action type, closing merge, open reticulations of s_t, lineages of s_{t+1}), padded to [B, T].
"""
import numpy as np
import torch
from src.env.trajectory import Trajectory, SimpleTrajectory

ACTION_TYPES = {'merge': 0, 'merge_open': 1, 'merge_close': 2, 'ret': 3}


class RolloutWorkerNet:

    def __init__(self, env, record_steps=False, wdb_gamma='uniform'):
        self.env = env
        self.record_steps = record_steps
        self.wdb_gamma = wdb_gamma

    def rollout(self, generator, episodes, scales=None, random_spec=None, generate_full_trajectories=False,
                input_actions_set=None):
        env = self.env
        states = [env.get_initial_state() for _ in range(episodes)]
        feats = env.initial_features(episodes)                       # [B, l_max, 2, m, c]
        trajectories = [Trajectory(s) if generate_full_trajectories else SimpleTrajectory() for s in states]
        dev = feats.device
        log_pf = torch.zeros(episodes, device=dev)
        log_pb = torch.zeros(episodes, device=dev)
        log_rewards = torch.zeros(episodes, dtype=torch.float64)
        log_scores = torch.zeros(episodes, dtype=torch.float64)
        done = np.array([s.is_done for s in states])
        rec = [[] for _ in range(episodes)] if self.record_steps else None
        step = 0
        while not done.all():
            active = np.where(~done)[0]
            act_states = [states[k] for k in active]
            input_actions = None
            if input_actions_set is not None:
                input_actions = []
                for k in active:
                    seq = input_actions_set[k] if k < len(input_actions_set) else None
                    input_actions.append(seq[step] if (seq is not None and step < len(seq)) else None)
            input_dict = env.prepare_rollout_inputs(feats[active], act_states, random_spec, input_actions)
            if self.record_steps:
                input_dict['need_log_flow'] = True          # the flow head is evaluated only when the records need it
            ret = generator(input_dict)
            actions = ret['actions']
            new_states, new_feats, ls, lr = env.batch_apply_actions(actions, feats[active], act_states)
            feats[active] = new_feats
            log_pf = log_pf.index_add(0, torch.tensor(active, device=dev), ret['log_paths_pf'])
            pa = torch.tensor([float(s.num_parents()) for s in new_states], device=dev)
            log_pb = log_pb.index_add(0, torch.tensor(active, device=dev), -torch.log(pa))
            for t, k in enumerate(active):
                if self.record_steps:
                    s_t, a = act_states[t], actions[t]
                    if a['type'] == 'merge':
                        i, j = int(env.pair_rows[a['pair']]), int(env.pair_cols[a['pair']])
                        case = env.merge_case(s_t.lineages[i], s_t.lineages[j])
                        kind = ACTION_TYPES['merge'] if case == 0 else (ACTION_TYPES['merge_close'] if case == 3 else ACTION_TYPES['merge_open'])
                    else:
                        kind = ACTION_TYPES['ret']
                    n_open = sum(1 for l in s_t.lineages if l.is_bare) // 2
                    rec[k].append({
                        'log_pf': ret['log_paths_pf'][t],
                        'log_flow': ret['log_flow'][t],
                        'log_pb': -float(np.log(float(new_states[t].num_parents()))),
                        'gamma': env.transition_weight(new_states[t], self.wdb_gamma),
                        'depth': step, 'kind': kind, 'n_open': n_open,
                        'n_lin_child': len(new_states[t].lineages), 'is_last': float(new_states[t].is_done),
                    })
                states[k] = new_states[t]
                if generate_full_trajectories:
                    trajectories[k].update(new_states[t], actions[t], float(lr[t]) if new_states[t].is_done else 0.0, new_states[t].is_done)
                else:
                    trajectories[k].update(actions[t], float(lr[t]) if new_states[t].is_done else 0.0)
                if new_states[t].is_done:
                    done[k] = True
                    log_rewards[k] = lr[t]
                    log_scores[k] = ls[t]
                    if generate_full_trajectories:
                        trajectories[k].current_state.network = env.state_to_network(new_states[t])
            step += 1
        data = {
            'log_pf': log_pf,
            'log_pb': log_pb,
            'log_rewards': log_rewards.to(log_pf),
            'log_scores': log_scores,
            'random_spec': random_spec,
            'scales': scales,
            'n_ret': np.array([s.n_ret for s in states]),
        }
        if self.record_steps:
            T = max(len(r) for r in rec)
            B = episodes
            zero = torch.zeros((), device=dev)

            def pad_t(key):
                return torch.stack([torch.stack([r[key] for r in rec[k]] + [zero] * (T - len(rec[k]))) for k in range(B)])

            def pad_f(key, dtype=torch.float32):
                out = torch.zeros(B, T, dtype=dtype)
                for k in range(B):
                    for t, r in enumerate(rec[k]):
                        out[k, t] = r[key]
                return out
            data['step_log_pf'] = pad_t('log_pf')
            data['step_log_flow'] = pad_t('log_flow')
            data['step_log_pb'] = pad_f('log_pb')
            data['step_gamma'] = pad_f('gamma', torch.float64)
            data['step_is_last'] = pad_f('is_last')
            data['step_mask'] = torch.tensor([[1.0 if t < len(rec[k]) else 0.0 for t in range(T)] for k in range(B)])
            data['step_depth'] = pad_f('depth', torch.long)
            data['step_kind'] = pad_f('kind', torch.long)
            data['step_n_open'] = pad_f('n_open', torch.long)
            data['step_n_lin_child'] = pad_f('n_lin_child', torch.long)
        return data, trajectories
