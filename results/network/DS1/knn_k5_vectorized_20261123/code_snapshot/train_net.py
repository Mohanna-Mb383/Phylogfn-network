"""
Train PhyloGFN-Net (level-1 phylogenetic networks) -- the network counterpart of train.py.

Usage:
    train_net.py <cfg_path> <sequences_path> <output_path> [--quiet] [--device=<dev>] [--seed=<s>]

Options:
    <cfg_path>          config path (see src/configs/network_cfgs/)
    <sequences_path>    .pickle / .fa sequences (same formats as train.py)
    <output_path>       output folder
    --device=<dev>      'cuda' or 'cpu' [default: cpu]
    --seed=<s>          random seed (torch, numpy, random); overrides cfg.SEED [default: -1]
    --quiet             no progress bar

Kept from train.py: epsilon-greedy exploration schedule, reward-temperature annealing
(with the log Z rescaling trick), replay buffer of best networks, checkpoints, MLL and
Pearson-r evaluation.  Removed: DDP / AMP (single device prototype).
"""
import os
import random
import pickle
import datetime
import shutil
import numpy as np
import torch
from tqdm import tqdm
from docopt import docopt

from src.configs.defaults import get_cfg_defaults
from src.utils.utils import correct_cfg_data, load_sequences, schedule, cascading_schedule
from src.env import build_env
from src.gfn.build import build_gfn
from src.utils.update_monitor import UpdateMonitor
from src.gfn.rollout_worker_net import RolloutWorkerNet
from src.gfn.training_data_loader_net import TrainingDataLoaderNet
from src.gfn.gfn_evaluator_net import GFNEvaluatorNet


def generate_exploration_spec(exploration_cfg, epoch):
    """same as train.py: per-epoch start/end values of the epsilon (or temperature) schedule"""
    assert exploration_cfg.METHOD in ['EPS_ANNEALING', 'TEMPERATURE_ANNEALING', 'NONE']
    if exploration_cfg.METHOD == 'NONE':
        return None
    start_value, end_value = exploration_cfg.START_VALUE, exploration_cfg.END_VALUE
    anneal_type, T = exploration_cfg.ANNEAL_TYPE, exploration_cfg.T
    return {
        'exploration_method': exploration_cfg.METHOD,
        'start_value': schedule(start_value, end_value, T, epoch, type=anneal_type),
        'end_value': schedule(start_value, end_value, T, epoch + 1, type=anneal_type),
    }


def current_temperature(t_cfg, epoch):
    if t_cfg.ANNEAL_TYPE == 'CASCADING':
        return cascading_schedule(t_cfg.CASCADING_SCHEDULE, epoch)
    if t_cfg.INVERSE_TEMPERATURE_ANNEALING:
        return schedule(t_cfg.START_VALUE, t_cfg.END_VALUE, t_cfg.T, epoch, type=t_cfg.ANNEAL_TYPE)
    return 1.0 / schedule(1 / t_cfg.START_VALUE, 1 / t_cfg.END_VALUE, t_cfg.T, epoch, type=t_cfg.ANNEAL_TYPE)


def set_seed(seed):
    """proposal, code-status slide: fixed seeds for exact comparisons"""
    if seed is None or int(seed) < 0:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reward_type_at(schedule_cfg, default, epoch):
    """ENV.REWARD.SCHEDULE = [[epoch, type], ...]: the reward type in force at `epoch`"""
    current = default
    for e, t in sorted((int(e), str(t)) for e, t in schedule_cfg):
        if epoch >= e:
            current = t
    return current


def train(cfg, sequences_path, output_path, device='cpu', verbose=True):
    set_seed(cfg.SEED)
    all_seqs = load_sequences(sequences_path)
    env = build_env(cfg, all_seqs)
    env.to(device)
    reward_schedule = list(cfg.ENV.REWARD.SCHEDULE)
    if reward_schedule:
        env.set_reward_type(reward_type_at(reward_schedule, cfg.ENV.REWARD.TYPE, 0))
    generator = build_gfn(cfg, env, device, ddp=False)
    rollout_worker = RolloutWorkerNet(env)
    data_loader = TrainingDataLoaderNet(cfg, env, rollout_worker, os.path.join(output_path, 'best_networks.pt'))
    evaluator = GFNEvaluatorNet(cfg.GFN.MODEL.EVALUATION, rollout_worker, generator, verbose=verbose)

    training_cfg = cfg.GFN.TRAINING_DATA_LOADER
    t_cfg = training_cfg.TEMPERATURE_ANNEALING
    history = []
    monitor = UpdateMonitor(generator, output_path)
    for epoch in range(training_cfg.EPOCHS_NUM):
        if reward_schedule:
            # proposal R4: anneal the reward function (PARSIMONY -> QUARTET -> LIKELIHOOD);
            # the replay buffer scores and log Z refer to the old reward -> re-initialise both at a switch
            new_type = reward_type_at(reward_schedule, cfg.ENV.REWARD.TYPE, epoch)
            if new_type != env.reward_type:
                if verbose:
                    print(f'epoch {epoch}: reward {env.reward_type} -> {new_type}')
                env.set_reward_type(new_type)
                generator.reinit_log_Z()
                data_loader.rescore_buffer()
        if t_cfg.TEMPERATURE_ANNEALING:
            temperature = current_temperature(t_cfg, epoch)
            if temperature != env.reward_fn.scale:
                # PhyloGFN trick: log Z scales with 1/temperature -> rescale the estimate
                current_log_z = generator.compute_log_Z().item() * (env.reward_fn.scale / temperature)
                torch.nn.init.constant_(generator._Z, current_log_z / 256)
                env.reward_fn.scale = temperature
        exploration_specs = generate_exploration_spec(training_cfg.EXPLORATION, epoch)
        iterator = data_loader.build_epoch_iterator(generator, exploration_specs)
        splits = training_cfg.MINI_BATCH_SPLITS
        losses = []
        bar = tqdm(total=data_loader.steps_per_epoch, desc=f'Epoch {epoch + 1}', leave=True) if verbose else None
        for t, ((batch, trajs), random_spec) in enumerate(iterator):
            generator.accumulate_loss(batch, splits)
            if (t + 1) % splits == 0:
                info = generator.update_model()
                monitor.after_update(epoch, t, info)
                losses.append(info['loss'])
                if generator.scheduler is not None and cfg.GFN.MODEL.LR_SCHEDULER.TYPE != 'STEP':
                    generator.scheduler.step(epoch + t / data_loader.steps_per_epoch)
                if bar is not None:
                    bar.update(1)
                    bar.set_description(f'Epoch {epoch + 1} loss {info["loss"]:.3f} logZ {generator.log_Z():.2f} '
                                        f'R-mean {batch["n_ret"].mean():.2f} best {max(x.log_score for x in data_loader.best) if data_loader.best else float("nan"):.2f}')
        if bar is not None:
            bar.close()
        if generator.loss != 0:
            generator.update_model()
        generator.save(os.path.join(output_path, 'checkpoints', 'checkpoint_%06d.pt' % epoch))
        if data_loader.best_state_batch_size > 0:
            pickle.dump(data_loader.best, open(data_loader.best_path, 'wb'))
        record = {'epoch': epoch, 'loss_mean': float(np.mean(losses)) if losses else None, 'log_Z': generator.log_Z(),
                  'temperature': env.reward_fn.scale}
        if epoch % cfg.GFN.MODEL.EVALUATION.EVALUATION_FREQ == 0:
            ev = evaluator.evaluate_gfn_quality(True)
            record.update({'mll': ev['mll'], 'pearsonr': ev['log_pearsonr'], 'mean_R': ev['mean_R'],
                           'hist_R': ev['hist_R'], 'sampled_log_score_mean': ev['gfn_samples_result']['log_scores_mean']})
            evaluator.update_states_set(evaluator.states + ev['gfn_samples_result']['states'])
            print(f'Epoch {epoch}: MLL {ev["mll"]:.3f}  PearsonR {ev["log_pearsonr"]:.3f}  '
                  f'mean R {ev["mean_R"]:.2f}  R histogram {ev["hist_R"]}')
        record.update(monitor.end_epoch(epoch))
        history.append(record)
        pickle.dump(history, open(os.path.join(output_path, 'history.pkl'), 'wb'))
    repeats = int(cfg.GFN.MODEL.EVALUATION_FINAL_MLL_REPEATS)
    final = [evaluator.evaluate_marginal_likelihood(1024)[0] for _ in range(repeats)]
    print('Final MLL: mean {:.3f} std {:.3f} ({} estimates of 1024 trajectories)'.format(np.mean(final), np.std(final), repeats))
    pickle.dump(final, open(os.path.join(output_path, 'final_mlls.p'), 'wb'))
    return history


if __name__ == '__main__':
    args = docopt(__doc__)
    all_seqs = load_sequences(args['<sequences_path>'])
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args['<cfg_path>'])

    _knn = cfg.ENV.NETWORK.get('KNN', None)

    if _knn is not None and _knn.ENABLED and cfg.GFN.TRAINING_DATA_LOADER.EXPLORATION.METHOD == 'TEMPERATURE_ANNEALING':

        raise ValueError('KNN + EXPLORATION.METHOD=TEMPERATURE_ANNEALING: actions would be sampled from softmax(logits/T) but scored with the eps-mixture log P_F. Use EPS_ANNEALING.')

    cfg = correct_cfg_data(all_seqs, 1, cfg)
    if args['--seed'] is not None and int(args['--seed']) >= 0:
        cfg.SEED = int(args['--seed'])
    out = args['<output_path>']
    parts = out.split(os.path.sep)
    parts[-1] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + parts[-1]
    out = os.path.sep.join(parts)
    cfg.OUTPUT_PATH = out
    os.makedirs(os.path.join(out, 'checkpoints'), exist_ok=True)
    cfg.dump(stream=open(os.path.join(out, 'config.yaml'), 'w'))
    train(cfg, args['<sequences_path>'], out, device=args['--device'], verbose=not args['--quiet'])
