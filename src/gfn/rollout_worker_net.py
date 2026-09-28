"""
Rollout worker for PhyloGFN-Net (network counterpart of rollout_worker_phylo.py).

PhyloGFN steps all trajectories of a batch in lock-step because every tree trajectory has
exactly N - 1 merges.  A network trajectory has N - 1 + 2R steps with R chosen by the
policy, so trajectories finish at different times: we keep a `done` mask, run the policy
only on the active trajectories, and accumulate  sum log P_F  and  sum log P_B  per
trajectory as we go (P_B is uniform: log P_B = -log |Pa(s')| of every visited state s').
"""
import numpy as np
import torch
from src.env.trajectory import Trajectory, SimpleTrajectory


class RolloutWorkerNet:

    def __init__(self, env):
        self.env = env

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
            ret = generator(input_dict)
            actions = ret['actions']
            new_states, new_feats, ls, lr = env.batch_apply_actions(actions, feats[active], act_states)
            feats[active] = new_feats
            log_pf = log_pf.index_add(0, torch.tensor(active, device=dev), ret['log_paths_pf'])
            pa = torch.tensor([float(s.num_parents()) for s in new_states], device=dev)
            log_pb = log_pb.index_add(0, torch.tensor(active, device=dev), -torch.log(pa))
            for t, k in enumerate(active):
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
        return data, trajectories
