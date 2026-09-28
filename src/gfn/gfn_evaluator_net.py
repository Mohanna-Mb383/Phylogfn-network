"""
Evaluator for PhyloGFN-Net (network counterpart of gfn_evaluator.py).

* marginal log-likelihood: the same importance-weighted lower bound as PhyloGFN (eq. 5)
      log P(Y) >= E log [ p(G) (1/K) sum_i P_B(tau_i | x_i) R(x_i) / P_F(tau_i) ]
  with the topology prior p(G) = exp(-lambda_R R(G)) / Z_lambda.  The environment already
  puts exp(-lambda_R R) into R(x) (`log_score`), so only -log Z_lambda is added here
  (PhyloGFN adds -log (2n-5)!! for its uniform prior over unrooted trees).
* Pearson correlation between the sampler's log-probability of a fixed set of networks
  (estimated by importance sampling over backward paths, eq. S1) and their log-reward.
"""
import numpy as np
import torch
import random
from tqdm import tqdm
from scipy.stats import pearsonr
from src.utils.level1_counts import log_topology_prior_normalizer


class GFNEvaluatorNet(object):

    def __init__(self, evaluation_cfg, rollout_worker, generator, states=None, verbose=True):
        self.env = rollout_worker.env
        self.rollout_worker = rollout_worker
        self.generator = generator
        self.evaluation_cfg = evaluation_cfg
        self.verbose = verbose
        self.states = states if states is not None else self.generate_initial_states()
        self.log_z_prior = log_topology_prior_normalizer(self.env.n_leaves, self.env.r_max, self.env.lambda_r, self.env.prior_type)

    def generate_initial_states(self):
        trajs = self.env.sample(self.evaluation_cfg.STATES_NUM, True)
        return [x.current_state for x in trajs]

    def evaluate_marginal_likelihood(self, traj_size=1024, chunk=256):
        with torch.no_grad():
            lw = []
            n_ret = []
            for _ in range(0, traj_size, chunk):
                data, _ = self.rollout_worker.rollout(self.generator, min(chunk, traj_size), generate_full_trajectories=False)
                lw.append(data['log_scores'].to(data['log_pf']) + data['log_pb'] - data['log_pf'])
                n_ret.append(data['n_ret'])
            lw = torch.cat(lw)
            mll = torch.logsumexp(lw, dim=0) - np.log(len(lw)) - self.log_z_prior
            n_ret = np.concatenate(n_ret)
            return mll.item(), {'mean_R': float(n_ret.mean()), 'hist_R': np.bincount(n_ret, minlength=self.env.r_max + 1).tolist()}

    def evaluate_gfn_quality_pearsonr(self, states=None):
        states = self.states if states is None else states
        logp, logr = [], []
        trajs_num = self.evaluation_cfg.TRAJECTORIES_PER_STATES
        for state in (tqdm(states) if self.verbose else states):
            net = state.network
            input_actions_set = [self.env.sample_backward_from_network(net)[0] for _ in range(trajs_num)]
            with torch.no_grad():
                data, _ = self.rollout_worker.rollout(self.generator, trajs_num, generate_full_trajectories=False,
                                                      input_actions_set=input_actions_set)
                est = torch.logsumexp(data['log_pf'] - data['log_pb'], dim=0) - np.log(trajs_num)
            logp.append(est.item())
            logr.append(float(data['log_rewards'][0]))
        return logp, logr, pearsonr(logp, logr)[0]

    def evaluate_gfn_quality(self, estimate_mll):
        logp, logr, r = self.evaluate_gfn_quality_pearsonr()
        ret = {'log_prob_reward': [logp, logr], 'log_pearsonr': r}
        if estimate_mll:
            mll, extra = self.evaluate_marginal_likelihood(1024)
            ret['mll'] = mll
            ret.update(extra)
        with torch.no_grad():
            data, trajs = self.rollout_worker.rollout(self.generator, 256, generate_full_trajectories=True)
        states = [t.current_state for t in trajs]
        scores = [s.log_score for s in states]
        ret['gfn_samples_result'] = {
            'states': states, 'log_scores': scores,
            'log_scores_mean': float(np.mean(scores)), 'log_scores_std': float(np.std(scores)),
            'log_scores_min': float(np.min(scores)), 'log_scores_max': float(np.max(scores)),
            'sampled_R': np.bincount(data['n_ret'], minlength=self.env.r_max + 1).tolist(),
        }
        return ret

    def update_states_set(self, states):
        random.shuffle(states)
        self.states = states[:self.evaluation_cfg.STATES_NUM]
