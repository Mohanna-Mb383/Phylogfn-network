"""
Training data loader for PhyloGFN-Net (network counterpart of training_data_loader.py).

Same two data sources as PhyloGFN:
  (i)  on-policy trajectories with epsilon-greedy exploration (random LEGAL actions);
  (ii) replay buffer of the best networks seen so far; a training trajectory for a
       buffered network is drawn with the uniform backward policy
       (env.sample_backward_from_network) and then re-scored by the current policy.
"""
import os
import random
import pickle
from heapq import heappush, heappushpop
import torch
from src.utils.utils import schedule


class TrainingDataLoaderNet(object):

    def __init__(self, cfg, env, rollout_worker, best_path):
        self.cfg = cfg
        self.env = env
        self.rollout_worker = rollout_worker
        self.best_path = best_path
        loader_cfg = cfg.GFN.TRAINING_DATA_LOADER
        splits = loader_cfg.MINI_BATCH_SPLITS
        self.gfn_batch_size = int(loader_cfg.GFN_BATCH_SIZE / splits)
        self.best_state_batch_size = int(loader_cfg.BEST_STATE_BATCH_SIZE / splits)
        self.batch_size = self.gfn_batch_size + self.best_state_batch_size
        self.steps_per_epoch = int(loader_cfg.STEPS_PER_EPOCH * splits)
        self.buffer_size = loader_cfg.BEST_TREES_BUFFER_SIZE
        self.topology_only = loader_cfg.BEST_TREES_TOPOLOGY_ONLY
        self.best = []
        self.seen = {}
        if self.best_state_batch_size > 0:
            self.initialize_best()

    def _key(self, net):
        return net.topology_id if self.topology_only else net.signature

    def initialize_best(self):
        if os.path.isfile(self.best_path):
            self.best = pickle.load(open(self.best_path, 'rb'))
            self.seen = {self._key(n): n for n in self.best}
        else:
            trajs = self.env.sample(200, True)
            self.update_buffer([t.current_state.network for t in trajs])

    def update_buffer(self, nets):
        for net in nets:
            key = self._key(net)
            if key in self.seen:
                continue
            self.seen[key] = net
            if len(self.best) >= self.buffer_size:
                dropped = heappushpop(self.best, net)
                self.seen.pop(self._key(dropped), None)
            else:
                heappush(self.best, net)

    def generate_batch(self, generator, random_spec):
        input_actions_set = None
        if self.best_state_batch_size > 0 and len(self.best) > 0:
            input_actions_set = []
            for net in random.choices(self.best, k=self.best_state_batch_size):
                actions, _ = self.env.sample_backward_from_network(net)
                input_actions_set.append(actions)
        data, trajectories = self.rollout_worker.rollout(generator, self.batch_size, random_spec=random_spec,
                                                         generate_full_trajectories=False,
                                                         input_actions_set=input_actions_set)
        if self.best_state_batch_size > 0:
            k = self.best_state_batch_size
            threshold = min(self.best).log_score if self.best else -float('inf')
            idx = torch.where(data['log_scores'][k:] > threshold)[0].cpu().numpy()
            if len(idx) > 0:
                acts = [trajectories[k:][i].actions for i in idx]
                scores = data['log_scores'][k:][idx]
                self.update_buffer(self.env.batch_actions_to_networks(acts, scores))
        return data, trajectories

    def build_epoch_iterator(self, generator, exploration_specs):
        for step in range(self.steps_per_epoch):
            random_spec = self.generate_random_spec(exploration_specs, step)
            yield self.generate_batch(generator, random_spec), random_spec

    def generate_random_spec(self, exploration_specs, step):
        if exploration_specs is None:
            return None
        value = schedule(exploration_specs['start_value'], exploration_specs['end_value'], self.steps_per_epoch, step,
                         type=self.cfg.GFN.TRAINING_DATA_LOADER.EXPLORATION.ANNEAL_TYPE)
        if exploration_specs['exploration_method'] == 'EPS_ANNEALING':
            return {'random_action_prob': value}
        return {'T': value}
