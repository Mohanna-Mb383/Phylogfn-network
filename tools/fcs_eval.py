"""
fcs_eval.py (version 3) -- distributional diagnostics for a trained PhyloGFN-Net run, after
da Silva, Alves, de Souza da Silva, Souza, Garg, Kaski, Mesquita, "When do GFlowNets learn the
right distribution?", ICLR 2025 (Definition 1, Theorem 5, Equation 8).

    python tools/fcs_eval.py <run_dir> <sequences.pickle> [--states=256] [--paths=100]
        [--betas=2,4,8,16] [--subsets=2000] [--stress=0|1] [--strata=1] [--strata_samples=512] [--strata_eps=0.1]
        [--kstudy=25,50,100,200,400] [--kreps=3] [--kobjects=48] [--device=cpu|cuda] [--checkpoint=<path>]
        [--seed=0] [--out=<file>]

Sections
  A. Policy set: --states networks sampled from the trained policy (the paper's P_S: subsets of the terminal
     states of a batch from a fixed policy).  Per network: log p_T(x) by K = --paths backward paths (Eq. 8; the
     estimator of GFNEvaluatorNet.evaluate_gfn_quality_pearsonr) and log R(x) = log_score (untempered).
     Reported: regression log p_T = a log R + b; Pearson and Spearman r; residual e(x) = log p_T - log R (its sd
     is the per-object balance error in nats, 0 for a balanced sampler); Policy-FCS per beta (Definition 1,
     extended to densities, see the caveat below); within-R FCS (ratio consistency inside one reticulation
     count; a consistency diagnostic, not a stratum-mass estimate).
  B. Residual association by trajectory features (screening, not localisation): e(x) grouped by R, by the step of
     the first RETICULATE, by the step of the first closing merge and by the presence of a 3-node cycle, with the
     share of residual variance explained (eta^2).  e(x) is a property of the terminal object; the features are
     properties of ONE trajectory that produced it and are correlated with each other, so a large eta^2 identifies
     features associated with terminal imbalance and says where transition-level residuals (which need a flow
     head) should be inspected; it does not establish which transition violated balance.
  C. Posterior mass over R, stratified: for every r = 0..R_MAX an explicit proposal q_r = the trained policy run
     with R_MAX := r, mixed at each step with the uniform distribution over legal actions (probability
     --strata_eps), whose density is computed exactly (continuous heads unchanged).  Terminals with R(x) = r keep
     their weight w = R(x) p_B(tau|x) / q_r(tau); the others get 0.  Z_r-hat = mean of w is unbiased for
     Z_r = sum over objects with R = r of R(x) because q_r has full support on every trajectory of every object with
     R <= r (the masked softmax is positive on all legal actions) and the uniform backward policy is unchanged on
     those states (parent sets coincide for n_ret <= r).  Reported per stratum: acceptance rate, log Z_r-hat with a
     bootstrap standard error, ESS of the accepted weights, "unavailable" when no accepted sample exists;
     P-hat(R = r) = Z_r-hat / sum_j Z_j-hat under the sampled prior and with 2^-r under the stated prior over
     graphs (CODE_VERIFICATION_REPORT.md, F1).  The same-policy-sample estimate is also printed, with its
     per-stratum ESS, as a cheap cross-check that is valid only for strata the policy actually visits.
  D. Stress set (--stress=1), fixed protocol: --states policy + --states random-rollout (env.sample: uniform over
     LEGAL ACTIONS at each step, log-uniform branch lengths, theta ~ U(0.02, 0.98); not uniform over networks) +
     --states top buffer networks if best_networks.pt exists; same seed, same K, same betas, same subset count.
     Stress-FCS is conditional on this protocol and is reported separately from Policy-FCS.
  E. K-convergence study (--kstudy): on --kobjects fixed policy networks, log p_T(x) is re-estimated with each K
     in the list and --kreps independent replicates; a, sd(e) and FCS_8 are printed per K with their spread over
     replicates, so that K is chosen from evidence.

Caveats (state them when reporting)
  * Theorem 5 is stated for a finite terminal set.  Here p_T and R are densities over (network, branch lengths,
    theta) and FCS is computed by normalising the densities on sampled objects; this keeps the ratio-matching
    interpretation (for positive densities, equality of the normalised ratios on almost every pair implies
    p_T = c R almost everywhere), but the finite-state equivalence theorem and Corollary 1 are not invoked
    without the extension argument written in the proposal.
  * log p_T(x) from K backward paths is biased low (Jensen); whether the bias is uniform across x is what
    section E measures.
  * In section C the discrete mixture probability is evaluated exactly; the continuous component inherits the
    PhyloGFN clipping approximation (the density of a clipped sample is evaluated as if unclipped).  A stratum
    with ESS/M < 0.1 is marked UNSTABLE regardless of its bootstrap standard error.
  * FCS is a diagnostic, not a bound on the global TV (Corollary 2 has a term that is not computable for DS-scale
    spaces); its relation to TV is validated on an enumerable small problem.
"""
import os, sys, glob, random, math, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from src.configs.defaults import get_cfg_defaults
from src.utils.utils import correct_cfg_data, load_sequences
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_net import RolloutWorkerNet


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('sequences')
    ap.add_argument('--states', type=int, default=256)
    ap.add_argument('--paths', type=int, default=100)
    ap.add_argument('--betas', default='2,4,8,16')
    ap.add_argument('--subsets', type=int, default=2000)
    ap.add_argument('--stress', type=int, default=0)
    ap.add_argument('--strata', type=int, default=1)
    ap.add_argument('--strata_samples', type=int, default=512)
    ap.add_argument('--strata_eps', type=float, default=0.1)
    ap.add_argument('--kstudy', default='')
    ap.add_argument('--kreps', type=int, default=3)
    ap.add_argument('--kobjects', type=int, default=48)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--selfcheck', type=int, default=1)
    ap.add_argument('--out', default=None)
    return ap.parse_args()


def load_run(run_dir, sequences, device, checkpoint=None):
    seqs = load_sequences(sequences)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(run_dir, 'config.yaml'))
    cfg = correct_cfg_data(seqs, 1, cfg)
    env = build_env(cfg, seqs)
    for name in ('D', 'U', 'U_inv'):                      # numpy >= 2 may return a complex JC eigendecomposition
        t = getattr(env.evolution_model, name)
        if torch.is_complex(t):
            t.data = t.data.real.clone()
    env.to(device)
    gen = build_gfn(cfg, env, device, ddp=False)
    if checkpoint is None:
        cks = sorted(glob.glob(os.path.join(run_dir, 'checkpoints', 'checkpoint_*.pt')))
        if not cks:
            raise SystemExit('no checkpoint in %s' % run_dir)
        checkpoint = cks[-1]
    gen.load(checkpoint)
    gen.eval()
    env.reward_fn.scale = 1.0                             # untempered target
    return cfg, env, gen, checkpoint


# ---------------------------------------------------------------------------- estimators
def log_pt_of(env, rw, gen, nets, K):
    log_pt, log_r = [], []
    for net in nets:
        paths = [env.sample_backward_from_network(net)[0] for _ in range(K)]
        with torch.no_grad():
            data, _ = rw.rollout(gen, K, generate_full_trajectories=False, input_actions_set=paths)
        lw = (data['log_pf'] - data['log_pb']).double()
        log_pt.append((torch.logsumexp(lw, 0) - math.log(K)).item())
        log_r.append(float(data['log_scores'][0]))
    return np.array(log_pt), np.array(log_r)


def mixture_rollout(env, gen, B, eps, r_max):
    """B trajectories from q = policy (with env.r_max := r_max) mixed with the uniform-over-legal-actions
    distribution with probability eps at every discrete step; returns log q, log p_B, log R, R per trajectory.
    Replicates RolloutWorkerNet.rollout and adds the exact mixture density."""
    old_r_max = env.r_max
    env.r_max = r_max
    try:
        states = [env.get_initial_state() for _ in range(B)]
        feats = env.initial_features(B)
        dev = feats.device
        log_q = torch.zeros(B, dtype=torch.float64)
        log_pb = torch.zeros(B, dtype=torch.float64)
        log_r = torch.zeros(B, dtype=torch.float64)
        action_lists = [[] for _ in range(B)]
        done = np.array([s.is_done for s in states])
        while not done.all():
            active = np.where(~done)[0]
            act_states = [states[k] for k in active]
            inp = env.prepare_rollout_inputs(feats[active], act_states, {'random_action_prob': eps}, None)
            with torch.no_grad():
                ret = gen(inp)
            actions = ret['actions']
            ids = torch.tensor([env.action_id(a, env.n_pairs) for a in actions], device=dev)
            logits = ret['logits']
            logp_disc = torch.log_softmax(logits, dim=1)[torch.arange(len(active)), ids].double()
            n_legal = inp['action_mask'].sum(1).double()
            log_mix = torch.logaddexp(math.log(1.0 - eps) + logp_disc, math.log(eps) - torch.log(n_legal)) if eps > 0 else logp_disc
            log_cont = ret['log_paths_pf'].double() - logp_disc           # continuous heads: same density under q
            new_states, new_feats, ls, lr = env.batch_apply_actions(actions, feats[active], act_states)
            feats[active] = new_feats
            for t, k in enumerate(active):
                action_lists[k].append(actions[t])
                log_q[k] += (log_mix[t] + log_cont[t]).item()
                log_pb[k] += -math.log(new_states[t].num_parents())
                states[k] = new_states[t]
                if new_states[t].is_done:
                    done[k] = True
                    log_r[k] = float(ls[t])
        R = np.array([s.n_ret for s in states])
        return log_q.cpu().numpy(), log_pb.cpu().numpy(), log_r.cpu().numpy(), R, action_lists
    finally:
        env.r_max = old_r_max


def selfcheck(env, gen, rw, say, rng):
    """A. with eps = 0 the proposal density equals the policy density: compare log q of mixture_rollout with the
    teacher-forced log P_F of RolloutWorkerNet on the same action sequences (checks the discrete + continuous
    accounting, assumption B).  C. the empirical action frequencies of the model's own epsilon-greedy sampler at a
    fixed state match (1-eps) softmax + eps uniform-over-legal (assumption A)."""
    lq, lpb, lr, R, acts = mixture_rollout(env, gen, 32, 0.0, env.r_max)
    with torch.no_grad():
        data, _ = rw.rollout(gen, 32, generate_full_trajectories=False, input_actions_set=acts)
    d = np.abs(lq - data['log_pf'].double().cpu().numpy()).max()
    d2 = np.abs(lpb - data['log_pb'].double().cpu().numpy()).max()
    say('selfcheck B: max |log q(eps=0) - teacher-forced log P_F| = %.2e ; max |log p_B| difference = %.2e  (both must be ~1e-5 or below)' % (d, d2))
    # empirical mixture test at a fixed non-terminal state
    s = env.get_initial_state()
    for _ in range(2):
        s, _, _ = env.apply_action(s, env.random_action(s), None)
    feats = env.initial_features(1)
    eps = 0.3
    counts = {}
    n_draw = 0
    with torch.no_grad():
        for _ in range(40):
            inp = env.prepare_rollout_inputs(feats.repeat(500, 1, 1, 1, 1), [s] * 500, {'random_action_prob': eps}, None)
            ret = gen(inp)
            for a in ret['actions']:
                k = env.action_id(a, env.n_pairs); counts[k] = counts.get(k, 0) + 1; n_draw += 1
        inp = env.prepare_rollout_inputs(feats, [s], None, None)
        logits = gen(inp)['logits'][0]
    legal = env.legal_actions(s)
    p_pol = torch.softmax(logits, 0).double().cpu().numpy()
    p_mix = (1 - eps) * p_pol + eps * legal / legal.sum()
    emp = np.array([counts.get(k, 0) / n_draw for k in range(env.n_actions)])
    say('selfcheck A: %d draws at a fixed state, %d legal actions: max |empirical - mixture| = %.4f (binomial sd ~ %.4f); illegal actions drawn: %d' % (
        n_draw, int(legal.sum()), np.abs(emp - p_mix).max(), np.sqrt(0.25 / n_draw), sum(counts.get(k, 0) for k in range(env.n_actions) if not legal[k])))


def stratified_masses(env, gen, M, eps, rng, say):
    logZ = {}
    rows = []
    for r in range(env.r_max + 1):
        lq, lpb, lr, R, _ = mixture_rollout(env, gen, M, eps, r)
        acc = R == r
        if acc.sum() == 0:
            rows.append((r, 0.0, None, None, None))
            say('   R=%d: proposal q_%d (policy with R_MAX=%d, eps=%.2f): acceptance 0/%d -> unavailable' % (r, r, r, eps, M))
            continue
        lw = np.where(acc, lr + lpb - lq, -np.inf)
        m = lw[acc].max()
        w = np.exp(lw - m) * 0.0; w[acc] = np.exp(lw[acc] - m)
        logZ_r = m + math.log(w.sum() / M)
        ess = w[acc].sum() ** 2 / (w[acc] ** 2).sum()
        boots = []
        for _ in range(200):
            idx = rng.integers(0, M, M)
            wb = w[idx]
            boots.append(m + math.log(max(wb.sum(), 1e-300) / M))
        se = float(np.std(boots))
        logZ[r] = logZ_r
        rows.append((r, acc.mean(), logZ_r, se, ess))
        flag = '   UNSTABLE (ESS/M < 0.1: bootstrap SE not trustworthy)' if ess / M < 0.1 else ''
        say('   R=%d: proposal q_%d (policy with R_MAX=%d, eps=%.2f): acceptance %d/%d = %.3f   log Z_r = %.3f +- %.3f (bootstrap)   ESS = %.1f%s' % (
            r, r, r, eps, acc.sum(), M, acc.mean(), logZ_r, se, ess, flag))
    if logZ:
        mx = max(logZ.values())
        tot = sum(math.exp(v - mx) for v in logZ.values())
        tot2 = sum(math.exp(v - mx - r * math.log(2)) for r, v in logZ.items())
        for r in range(env.r_max + 1):
            if r in logZ:
                say('   P_target(R=%d) = %.4f (sampled prior)   %.4f (stated prior over graphs, x 2^-R)' % (
                    r, math.exp(logZ[r] - mx) / tot, math.exp(logZ[r] - mx - r * math.log(2)) / tot2))
            else:
                say('   P_target(R=%d) = unavailable (no accepted proposal sample)' % r)
        say('   stratified evidence: log sum_r Z_r = %.3f' % (mx + math.log(tot)))
    return logZ


# ---------------------------------------------------------------------------- FCS and helpers
def tv_restricted(lp, lr):
    p = np.exp(lp - lp.max()); p /= p.sum()
    q = np.exp(lr - lr.max()); q /= q.sum()
    return 0.5 * np.abs(p - q).sum()


def fcs(log_pt, log_r, beta, m, rng, groups=None, within=False):
    n = len(log_pt)
    idx = np.arange(n)
    vals, tries = [], 0
    while len(vals) < m and tries < 50 * m:
        tries += 1
        if within:
            g = rng.choice(np.unique(groups))
            pool = idx[groups == g]
            if len(pool) < beta:
                continue
            S = rng.choice(pool, beta, replace=False)
        else:
            if n < beta:
                break
            S = rng.choice(idx, beta, replace=False)
        vals.append(tv_restricted(log_pt[S], log_r[S]))
    return (float(np.mean(vals)) if vals else float('nan')), len(vals)


def eta2(values, groups):
    tot = np.var(values)
    if tot == 0:
        return 0.0
    gm = np.mean(values)
    between = sum((groups == g).sum() * (values[groups == g].mean() - gm) ** 2 for g in np.unique(groups)) / len(values)
    return float(between / tot)


def _nodes(root):
    out, seen, stack = [], set(), [root]
    while stack:
        v = stack.pop()
        if v.idx in seen:
            continue
        seen.add(v.idx)
        out.append(v)
        stack.extend(v.children)
    return out


def trajectory_features(env, traj):
    first_ret, first_close = -1, -1
    for t, (s, s2, a, lr, done) in enumerate(traj.transitions):
        if a['type'] == 'ret' and first_ret < 0:
            first_ret = t
        if a['type'] == 'merge' and first_close < 0:
            i, j = int(env.pair_rows[a['pair']]), int(env.pair_cols[a['pair']])
            if env.merge_case(s.lineages[i], s.lineages[j]) == 3:
                first_close = t
    net = traj.current_state.network
    tri = 0
    for v in _nodes(net.root):
        if v.kind == 'ret':
            p0, p1 = v.parents
            if p0 in p1.children or p1 in p0.children:
                tri = 1
    return net.num_reticulations(), first_ret, first_close, tri


def report_set(say, name, log_pt, log_r, R, betas, m, rng, logz):
    a, b = np.polyfit(log_r, log_pt, 1)
    e = log_pt - log_r
    say('[%s] n=%d  regression log p_T = a log R + b: a = %.4f  b = %.3f  (-b = %.3f, learned log Z = %.3f)' % (name, len(log_pt), a, b, -b, logz))
    say('[%s] pearson r = %.4f  spearman = %.4f  residual e = log p_T - log R: mean %.3f  sd %.3f nats' % (
        name, pearsonr(log_pt, log_r)[0], spearmanr(log_pt, log_r)[0], e.mean(), e.std()))
    for beta in betas:
        f_all, m_all = fcs(log_pt, log_r, beta, m, rng)
        f_in, m_in = fcs(log_pt, log_r, beta, m, rng, R, within=True)
        say('[%s] FCS beta=%2d: %.4f (m=%d)   within-R: %.4f (m=%d)' % (name, beta, f_all, m_all, f_in, m_in))
    return e


def main():
    args = parse()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg, env, gen, ck = load_run(args.run_dir, args.sequences, args.device, args.checkpoint)
    rw = RolloutWorkerNet(env)
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    say('run %s  checkpoint %s  N=%d  R_MAX=%d  LAMBDA_R=%g  PRIOR_TYPE=%s  K=%d  seed=%d' % (
        args.run_dir, os.path.basename(ck), env.n_leaves, env.r_max, env.lambda_r, env.prior_type, args.paths, args.seed))
    betas = [int(x) for x in args.betas.split(',')]
    logz = gen.log_Z()
    if args.selfcheck:
        selfcheck(env, gen, rw, say, rng)

    # ---------------------------------------------------------------- A
    with torch.no_grad():
        data, trajs = rw.rollout(gen, args.states, generate_full_trajectories=True)
    nets = [t.current_state.network for t in trajs]
    feats = np.array([trajectory_features(env, t) for t in trajs])
    R, first_ret, first_close, tri = feats[:, 0], feats[:, 1], feats[:, 2], feats[:, 3]
    say('A. policy set: %d networks; R histogram %s' % (len(nets), np.bincount(R, minlength=env.r_max + 1).tolist()))
    log_pt, log_r = log_pt_of(env, rw, gen, nets, args.paths)
    e = report_set(say, 'policy', log_pt, log_r, R, betas, args.subsets, rng, logz)

    # ---------------------------------------------------------------- B
    say('B. residual association by trajectory features (screening; mean+-sd (n) per value; eta^2 = share of residual variance explained):')
    for label, g in (('R', R), ('first RETICULATE step', first_ret), ('first closing-merge step', first_close), ('3-node cycle', tri)):
        parts = ['%d: %.2f+-%.2f (%d)' % (v, e[g == v].mean(), e[g == v].std(), (g == v).sum()) for v in np.unique(g)]
        say('   %-26s eta^2 = %.3f   %s' % (label, eta2(e, g), '  '.join(parts[:12]) + ('  ...' if len(parts) > 12 else '')))

    # ---------------------------------------------------------------- C
    say('C. posterior mass over R:')
    lw = (data['log_scores'].double().cpu() + data['log_pb'].double().cpu() - data['log_pf'].double().cpu()).numpy()
    w = np.exp(lw - lw.max()); w /= w.sum()
    for r in range(env.r_max + 1):
        sel = R == r
        if sel.sum() == 0:
            say('   same-policy-sample estimate R=%d: unavailable (0 policy samples in the stratum); sampler frequency 0' % r)
        else:
            ess_r = w[sel].sum() ** 2 / (w[sel] ** 2).sum()
            say('   same-policy-sample estimate R=%d: sampler frequency %.4f   IS mass %.4f   stratum ESS %.1f (n=%d)' % (r, sel.mean(), w[sel].sum(), ess_r, sel.sum()))
    if args.strata:
        say('   stratified estimate (explicit proposals q_r, M=%d each):' % args.strata_samples)
        stratified_masses(env, gen, args.strata_samples, args.strata_eps, rng, say)

    # ---------------------------------------------------------------- D
    if args.stress:
        rr = [t.current_state.network for t in env.sample(args.states, True)]
        origin = ['policy'] * len(nets) + ['random-rollout'] * len(rr)
        allnets = nets + rr
        path = os.path.join(args.run_dir, 'best_networks.pt')
        if os.path.isfile(path):
            best = sorted(pickle.load(open(path, 'rb')), key=lambda b: -b.log_score)[:args.states]
            allnets += best; origin += ['buffer'] * len(best)
        origin = np.array(origin)
        Rs = np.array([n.num_reticulations() for n in allnets])
        lp2, lr2 = log_pt_of(env, rw, gen, allnets[len(nets):], args.paths)
        lp_all = np.concatenate([log_pt, lp2]); lr_all = np.concatenate([log_r, lr2])
        say('D. stress set under the fixed protocol (%d policy + %d random-rollout + buffer top %d; seed %d; K=%d): %d networks; R histogram %s' % (
            args.states, args.states, args.states, args.seed, args.paths, len(allnets), np.bincount(Rs, minlength=env.r_max + 1).tolist()))
        e2 = report_set(say, 'stress', lp_all, lr_all, Rs, betas, args.subsets, rng, logz)
        for o in sorted(set(origin)):
            sel = origin == o
            say('[stress] origin=%-15s n=%3d  residual mean %.3f  sd %.3f' % (o, sel.sum(), e2[sel].mean(), e2[sel].std()))

    # ---------------------------------------------------------------- E
    if args.kstudy:
        Ks = [int(x) for x in args.kstudy.split(',')]
        objs = nets[:args.kobjects]
        Ro = R[:args.kobjects]
        say('E. K-convergence study on %d fixed policy networks, %d replicates per K:' % (len(objs), args.kreps))
        for K in Ks:
            A, S, F = [], [], []
            for rep in range(args.kreps):
                lp, lr = log_pt_of(env, rw, gen, objs, K)
                a, b = np.polyfit(lr, lp, 1)
                A.append(a); S.append((lp - lr).std()); F.append(fcs(lp, lr, 8, 500, rng)[0])
            say('   K=%4d: a = %.4f +- %.4f   sd(e) = %.3f +- %.3f   FCS_8 = %.4f +- %.4f' % (
                K, np.mean(A), np.std(A), np.mean(S), np.std(S), np.mean(F), np.std(F)))
    if args.out:
        open(args.out, 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
