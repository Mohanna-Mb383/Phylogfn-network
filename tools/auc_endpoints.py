"""
auc_endpoints.py -- checkpoint-wise endpoints for the time-resolved (AUC) comparison of objectives.

Per run (every checkpoint, or every k-th with --every), evaluated with a fixed seed per checkpoint:
  * Policy-FCS_beta (the Eq. 8 estimator with K = --paths backward paths on --states policy samples; plus the slope a
    and the residual sd of log p_T against log R), as in tools/fcs_eval.py section A;
  * on a problem with a saved numerical reference (--reference from tools/reference_posterior_small_n.py
    --save_reference): TV of the sampler's topology frequencies (--samples draws, aggregated over slot labellings) to
    the labelled target and, 2^-R-reweighted, to the graph posterior;
  * the wall-clock of the checkpoint from history.pkl (train_time_s = training loop only, excluding the periodic
    evaluation; elapsed_s = total), falling back to checkpoint file mtimes for runs made before the patch (these
    include evaluation time; the CSV says which source was used).
Writes one CSV per run and prints the run's AUC (trapezoid over time, with the first checkpoint's value held constant
back to t = 0) and the time-averaged value AUC / horizon.

    python tools/auc_endpoints.py <run_dir> <sequences.pickle> [--reference=<ref.pkl> --ref_cfg=<cfg.yaml>]
        [--states=128] [--paths=50] [--beta=8] [--subsets=1000] [--samples=4096] [--every=1] [--device=cpu]
        [--seed=0] [--time=train_time_s|elapsed_s|mtime] [--out=<csv>]

Comparison across runs (paired by seed):

    python tools/auc_endpoints.py --compare 'runs/*_n4_*/endpoints.csv' --metric=tv_lab --control=db
        [--horizon=<seconds>] [--variants=db,wdb_logcount,wdb_count]

Run directories are expected as <timestamp>_<problem>_<variant>_s<seed> (what run_paper_methods.sh produces).  The
AUC of every run is taken on the common horizon [0, H] with H = --horizon or the smallest final time among the runs
(so a faster-finishing variant is not penalised for stopping early, and a slower one does not get extra time); for
every variant other than --control, the paired differences (variant - control, same seed) are printed with mean,
standard error, the sign count and the Wilcoxon signed-rank p-value (two-sided; with 3-5 seeds this test has little
power, which is why the gate rule is a direction + magnitude rule, not a significance threshold).
"""
import os, sys, glob, math, random, argparse, csv, pickle, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

from src.utils.utils import load_sequences
from src.gfn.rollout_worker_net import RolloutWorkerNet

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
from fcs_eval import log_pt_of, fcs                                     # noqa: E402
import reference_posterior_small_n as refmod                            # noqa: E402


def checkpoint_times(run_dir, cks, source):
    """seconds of every checkpoint on the requested axis; returns (times, source actually used)"""
    hist_path = os.path.join(run_dir, 'history.pkl')
    by_epoch = {}
    if os.path.isfile(hist_path):
        for rec in pickle.load(open(hist_path, 'rb')):
            by_epoch[rec['epoch']] = rec
    if source in ('train_time_s', 'elapsed_s') and by_epoch and all(source in r for r in by_epoch.values()):
        times = []
        for ck in cks:
            ep = int(re.search(r'checkpoint_(\d+)\.pt$', ck).group(1))
            if ep not in by_epoch:
                raise RuntimeError('history.pkl has no record for %s' % ck)
            times.append(float(by_epoch[ep][source]))
        return np.array(times), source
    mt = np.array([os.path.getmtime(ck) for ck in cks])
    first = int(re.search(r'checkpoint_(\d+)\.pt$', cks[0]).group(1))
    # the first checkpoint is written after the first epoch: offset by one epoch duration (estimated from the spacing)
    step = np.median(np.diff(mt)) if len(mt) > 1 else 0.0
    return mt - mt[0] + step * (first + 1), 'mtime'


def auc(times, values):
    """trapezoid AUC on [0, max t] with the first value held constant on [0, t_0]"""
    t = np.concatenate([[0.0], times]); v = np.concatenate([[values[0]], values])
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    return float(_trapz(v, t)), float(t[-1])


def evaluate_run(args):
    seqs = load_sequences(args.sequences)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cfg_r, env_r, gen, _ = refmod.load_run_generator(args.run_dir, seqs, args.device)
    cks = sorted(glob.glob(os.path.join(args.run_dir, 'checkpoints', 'checkpoint_*.pt')))
    if not cks:
        raise SystemExit('no checkpoint in %s' % args.run_dir)
    cks = cks[::args.every] if cks[-1] in cks[::args.every] else cks[::args.every] + [cks[-1]]
    times, source = checkpoint_times(args.run_dir, cks, args.time)
    ref = None
    if args.reference:
        cfg_p, env_p = refmod.build_problem_env(args.ref_cfg, seqs)
        ref = refmod.load_reference(args.reference, env_p, cfg_p.ENV.EVOLUTION_MODEL.PRIOR_LAMBDA)
        print('reference %s: %d graphs, %d draws per graph' % (args.reference, len(ref['keys']), ref['identity']['draws']))
    rw = RolloutWorkerNet(env_r)
    rows = []
    print('run %s: %d checkpoints (every %d), time axis %s, FCS beta=%d states=%d K=%d%s' % (
        args.run_dir, len(cks), args.every, source, args.beta, args.states, args.paths,
        ', TV with %d samples' % args.samples if ref else ''))
    for ck, t_s in zip(cks, times):
        ep = int(re.search(r'checkpoint_(\d+)\.pt$', ck).group(1))
        gen.load(ck); gen.eval(); env_r.reward_fn.scale = 1.0
        random.seed(args.seed + ep); np.random.seed(args.seed + ep); torch.manual_seed(args.seed + ep)
        rng = np.random.default_rng(args.seed + ep)
        with torch.no_grad():
            data, trajs = rw.rollout(gen, args.states, generate_full_trajectories=True)
        nets = [t.current_state.network for t in trajs]
        log_pt, log_r = log_pt_of(env_r, rw, gen, nets, args.paths)
        a, b = np.polyfit(log_r, log_pt, 1) if len(log_pt) > 1 else (float('nan'), float('nan'))
        f_val, m_used = fcs(log_pt, log_r, args.beta, args.subsets, rng)
        row = {'epoch': ep, 'checkpoint': os.path.basename(ck), 'time_s': float(t_s), 'time_source': source,
               'fcs_beta': args.beta, 'fcs': f_val, 'fcs_subsets': m_used, 'slope_a': float(a), 'resid_sd': float((log_pt - log_r).std()),
               'mean_R': float(np.mean([n.num_reticulations() for n in nets])), 'log_Z': float(gen.log_Z())}
        if ref is not None:
            f = refmod.sampler_frequencies(env_r, gen, ref['keys'], env_p.n_leaves, args.samples)
            tv_lab, tv_graph, mass = refmod.sampler_tv(ref, env_p.lambda_r, f)
            row.update({'tv_lab': float(tv_lab), 'tv_graph': float(tv_graph)})
            row.update({'mass_R%d' % r: m for r, m in enumerate(mass)})
        rows.append(row)
        print('  epoch %4d  t = %9.1f s  FCS_%d = %.4f  a = %.3f  sd(e) = %.3f%s' % (
            ep, t_s, args.beta, f_val, a, row['resid_sd'], '  TV_lab = %.4f  TV_graph = %.4f' % (row['tv_lab'], row['tv_graph']) if ref else ''))
    out = args.out or os.path.join(args.run_dir, 'endpoints.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    tt = np.array([r['time_s'] for r in rows])
    for key in ('fcs',) + (('tv_lab', 'tv_graph') if ref else ()):
        A, H = auc(tt, np.array([r[key] for r in rows]))
        print('AUC(%s) over [0, %.0f s] = %.2f   time-averaged %.4f   final %.4f' % (key, H, A, A / H if H > 0 else float('nan'), rows[-1][key]))
    print('written %s' % out)


def parse_run_name(path, pattern):
    base = os.path.basename(os.path.dirname(os.path.abspath(path)))
    m = re.match(pattern, base)
    if not m:
        raise RuntimeError('run directory %s does not match %s' % (base, pattern))
    return m.group('variant'), int(m.group('seed'))


def interp_auc(times, values, H):
    """AUC on [0, H]: first value held constant before the first checkpoint, linear interpolation between checkpoints,
    last value held constant after the last checkpoint (only used when H exceeds the run's final time)"""
    t = np.concatenate([[0.0], times]); v = np.concatenate([[values[0]], values])
    ok = np.isfinite(v); t, v = t[ok], v[ok]
    grid = np.unique(np.concatenate([t[t <= H], [H]]))
    vals = np.interp(grid, t, v)
    return float(_trapz(vals, grid))


def compare(args):
    files = sorted(glob.glob(args.compare))
    if not files:
        raise SystemExit('no file matches %s' % args.compare)
    runs = {}
    for f in files:
        variant, seed = parse_run_name(f, args.pattern)
        rows = list(csv.DictReader(open(f)))
        if args.metric not in rows[0]:
            raise SystemExit('%s has no column %s (columns: %s)' % (f, args.metric, list(rows[0].keys())))
        t = np.array([float(r['time_s']) for r in rows]); v = np.array([float(r[args.metric]) for r in rows])
        runs[(variant, seed)] = (t, v, rows[0]['time_source'])
    variants = args.variants.split(',') if args.variants else sorted({k[0] for k in runs})
    H = args.horizon or min(t[-1] for (t, v, s) in runs.values())
    sources = sorted({s for (t, v, s) in runs.values()})
    print('metric %s, %d runs, horizon H = %.0f s (%s), time source %s' % (args.metric, len(runs), H, 'given' if args.horizon else 'smallest final time', sources))
    table = {}
    for (variant, seed), (t, v, s) in sorted(runs.items()):
        if variant not in variants:
            continue
        A = interp_auc(t, v, H)
        table[(variant, seed)] = A
        print('  %-18s seed %d: AUC = %10.2f   time-averaged %.4f   final %.4f   (last checkpoint at %.0f s)' % (variant, seed, A, A / H, v[-1], t[-1]))
    if args.control:
        from scipy.stats import wilcoxon
        for variant in variants:
            if variant == args.control:
                continue
            seeds = sorted(s for (vv, s) in table if vv == variant and (args.control, s) in table)
            if not seeds:
                print('%s: no seed in common with %s' % (variant, args.control)); continue
            d = np.array([table[(variant, s)] - table[(args.control, s)] for s in seeds]) / H
            p = wilcoxon(d).pvalue if len(d) >= 2 and np.any(d != 0) else float('nan')
            print('%s - %s (time-averaged %s, paired on %d seeds): mean %+.4f  SE %.4f  negative in %d/%d seeds  Wilcoxon p = %.3f' % (
                variant, args.control, args.metric, len(seeds), d.mean(), d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float('nan'),
                int((d < 0).sum()), len(d), p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', nargs='?'); ap.add_argument('sequences', nargs='?')
    ap.add_argument('--reference', default=None); ap.add_argument('--ref_cfg', default=None)
    ap.add_argument('--states', type=int, default=128); ap.add_argument('--paths', type=int, default=50)
    ap.add_argument('--beta', type=int, default=8); ap.add_argument('--subsets', type=int, default=1000)
    ap.add_argument('--samples', type=int, default=4096)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--time', default='train_time_s', choices=['train_time_s', 'elapsed_s', 'mtime'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--compare', default=None, help="glob of endpoints.csv files, e.g. 'runs/*_n4_*/endpoints.csv'")
    ap.add_argument('--metric', default='fcs'); ap.add_argument('--control', default=None)
    ap.add_argument('--horizon', type=float, default=None); ap.add_argument('--variants', default=None)
    ap.add_argument('--pattern', default=r'^\d{8}_\d{6}_(?P<problem>[a-z0-9]+)_(?P<variant>.+)_s(?P<seed>\d+)$')
    args = ap.parse_args()
    if args.compare:
        compare(args)
        return
    if not args.run_dir or not args.sequences:
        ap.error('run_dir and sequences are required (or --compare)')
    if bool(args.reference) != bool(args.ref_cfg):
        ap.error('--reference and --ref_cfg go together')
    evaluate_run(args)


if __name__ == '__main__':
    main()
