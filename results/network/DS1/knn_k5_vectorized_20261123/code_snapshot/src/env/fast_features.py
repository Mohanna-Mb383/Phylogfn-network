"""Batched likelihood-feature updates for PhyloNetworkEnv.batch_apply_actions, WITHOUT replacing any action logic.

What it does
------------
`install_fast_features(env)` wraps `env.batch_apply_actions`.  For every call with features:
  1. PLAN (read-only, before any state is mutated): for each trajectory whose action type is exactly 'merge' or 'ret'
     (configurable), derive the arguments the published `apply_action` passes to `_merge_features(feats, i, j, case, bl,
     br, last_step)` / `_reticulate_features(feats, i, n, theta, b_hx)`.  Any other action type (explicit STOP, ...) or any
     derivation error -> no plan for that trajectory.
  2. BATCH: compute all planned results with a few batched tensor ops (same formulas and operand order as the
     per-trajectory functions).
  3. RUN THE FORK'S OWN LOOP: the original `batch_apply_actions` runs unchanged, so STOP handling, termination, terminal
     log-likelihood, state-specific lambda, topology prior, PARSIMONY/QUARTET rewards and legality asserts are all the
     fork's code.  During the loop `env._merge_features` / `env._reticulate_features` are instance-level stubs: a stub
     returns the precomputed tensor ONLY IF it is called with the feature row of a planned trajectory (identified by
     data pointer, shape and strides) AND with argument values exactly equal to the plan.  Every other call is forwarded
     to the original method (counted as a miss with its reason).  Stubs are removed in `finally`.
Safety
------
  * `install_fast_features` first runs a self-check of the batched algebra against the env's OWN `_merge_features` /
    `_reticulate_features` (all 4 merge cases, last/non-last step, reticulations).  Mismatch or incompatible signatures
    -> not installed (returns False, prints why); the env is left untouched.
  * PHYLOGFN_FAST_FEATURES=off|on|verify   (default on).  verify: each stub hit is also recomputed with the original
    method and must agree within PHYLOGFN_FAST_FEATURES_RTOL (default 0, i.e. bitwise); otherwise RuntimeError.
  * env._fast_features_stats: counters (planned, hits, misses by reason, verify max diffs).
"""
import inspect
import numbers
import os

import numpy as np

import torch

MERGE_TYPES = ('merge',)
RET_TYPES = ('ret',)
_MERGE_SIG = ['feats', 'i', 'j', 'case', 'bl', 'br', 'last_step']
_RET_SIG = ['feats', 'i', 'n', 'theta', 'b_hx']


def _mode():
    return os.environ.get('PHYLOGFN_FAST_FEATURES', 'on').strip().lower()


# --------------------------------------------------------------------------------------------- batched algebra
def batched_merge(env, feats, K, I, J, C, BL, BR, LAST):
    """feats [B, l_max, 2, m, c]; K,I,J,C long [Mm]; BL,BR python floats; LAST bool list -> [Mm, l_max, 2, m, c]."""
    dt, dev = feats.dtype, feats.device
    em = env.evolution_model
    Mm = len(BL)
    b = torch.tensor([BL, BR], dtype=dt, device=dev).t().contiguous()              # [Mm, 2]
    last = torch.tensor(LAST, dtype=torch.bool, device=dev)
    M = em.get_transition_matrices(b.reshape(-1)).reshape(Mm, 2, env.c, env.c)
    pl = torch.exp(em.compute_log_prior_p(b) / env.m)
    pl = torch.where(last[:, None], torch.ones_like(pl), pl)                       # root step: no per-edge prior
    prior = torch.where(last, torch.exp(em.compute_log_prior_p(b[:, 0] + b[:, 1]) / env.m),
                        torch.ones_like(b[:, 0]))[:, None, None]
    uA, uB, vA, vB = feats[K, I, 0], feats[K, I, 1], feats[K, J, 0], feats[K, J, 1]
    tu = lambda f: torch.einsum('bmv,bvc->bmc', f, M[:, 0]) * pl[:, 0, None, None]  # noqa: E731
    tv = lambda f: torch.einsum('bmv,bvc->bmc', f, M[:, 1]) * pl[:, 1, None, None]  # noqa: E731
    tuA, tuB, tvA, tvB = tu(uA), tu(uB), tv(vA), tv(vB)
    Cb = C[:, None, None]
    plain = tuA * tvA * prior
    wA = torch.where(Cb == 3, (tuA * tvB + tuB * tvA) * prior, plain)
    wB = torch.where(Cb == 1, tuB * tvA * prior, torch.where(Cb == 2, tuA * tvB * prior, wA))
    src = torch.arange(env.l_max, device=dev)[None, :].expand(Mm, -1)
    src = (src + (src >= J[:, None]).long()).clamp(max=env.l_max - 1)              # drop row j, shift up
    out = feats[K[:, None], src]                                                    # one gather (copy)
    out[:, env.l_max - 1] = 1.0                                                     # appended all-ones row
    ar = torch.arange(Mm, device=dev)
    out[ar, I, 0] = wA
    out[ar, I, 1] = wB
    return out


def batched_reticulate(env, feats, K, I, N, TH, BH):
    """Reticulations are computed ONE AT A TIME with exactly the call shapes of the original
    (_transport(xA.unsqueeze(0), b) with b of shape [1]): torch.matmul dispatches a [1,c,c] einsum output (strides
    (1,c,1)) to a different BLAS kernel than a folded [k>=2,c,c] batch, which changes the last bit (observed on
    torch 2.5.1/MKL: 1.1e-16 in M, 1.1e-12 relative in features).  Reticulations are a small fraction of transitions;
    only the cheap elementwise updates are batched."""
    dt, dev = feats.dtype, feats.device
    em = env.evolution_model
    bh = torch.tensor(BH, dtype=dt, device=dev)                                       # one transfer; slices are [1]
    xA = feats[K, I, 0]                                                               # [Mr, m, c] (gather)
    Lh = torch.empty_like(xA)
    for t in range(len(BH)):
        b1 = bh[t:t + 1]
        M = em.get_transition_matrices(b1)                                            # [1, c, c], original layout
        o = torch.einsum('bmv,bvc->bmc', xA[t:t + 1], M)
        Lh[t] = (o * torch.exp(em.compute_log_prior_p(b1) / env.m).reshape(-1, 1, 1))[0]
    th = torch.tensor(TH, dtype=dt, device=dev)[:, None, None]
    th1 = torch.tensor([1.0 - t for t in TH], dtype=dt, device=dev)[:, None, None]   # python 1.0 - theta, as original
    out = feats[K]                                                                    # copy
    ar = torch.arange(len(TH), device=dev)
    out[ar, I, 0] = th * Lh
    out[ar, I, 1] = 1.0
    out[ar, N, 0] = th1 * Lh
    out[ar, N, 1] = 1.0
    return out


# --------------------------------------------------------------------------------------------- planning
def _plan(env, actions, states):
    """Read-only derivation of the feature-function arguments (mirrors the published apply_action)."""
    merges, rets = {}, {}
    for k, a in enumerate(actions):
        try:
            s = states[k]
            t = a.get('type') if isinstance(a, dict) else None
            if t in MERGE_TYPES:
                L = list(s.lineages)
                i, j = int(env.pair_rows[a['pair']]), int(env.pair_cols[a['pair']])
                if not (0 <= i < j < len(L)):
                    continue
                merges[k] = (i, j, int(env.merge_case(L[i], L[j])), float(a['edge_action'][0]),
                             float(a['edge_action'][1]), len(L) == 2)
            elif t in RET_TYPES:
                L = list(s.lineages)
                rets[k] = (int(a['lineage']), len(L), float(a['theta']), float(a['edge_action']))
        except Exception:                                                             # noqa: BLE001
            continue                                                                  # -> original path
    return merges, rets


def _row_key(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, str(t.device))


def _bind(sig_names, args, kwargs):
    if len(args) + len(kwargs) != len(sig_names):
        return None
    d = dict(zip(sig_names, args))
    for k, v in kwargs.items():
        if k not in sig_names or k in d:
            return None
        d[k] = v
    return d if len(d) == len(sig_names) else None


def _same(x, y):
    """x: value the fork passed; y: planned python value. Exact equality only; tensors never match (-> original)."""
    if torch.is_tensor(x):
        return False
    if isinstance(y, bool):
        return isinstance(x, (bool, np.bool_)) and bool(x) == y
    if isinstance(x, (bool, np.bool_)):
        return False
    if isinstance(x, numbers.Number) and not isinstance(x, complex):
        if isinstance(y, int) and not float(x).is_integer():
            return False
        return float(x) == float(y)
    return False


def _compare(a, b, rtol, what, stats):
    d = (a.double() - b.double()).abs()
    fin = torch.isfinite(a) & torch.isfinite(b)
    same_nf = bool(torch.equal(torch.isfinite(a), torch.isfinite(b)))
    md = float(d[fin].max()) if fin.any() else 0.0
    mr = float((d[fin] / b.double().abs()[fin].clamp_min(1e-300)).max()) if fin.any() else 0.0
    stats['verify_max_abs'] = max(stats.get('verify_max_abs', 0.0), md)
    stats['verify_max_rel'] = max(stats.get('verify_max_rel', 0.0), mr)
    if not same_nf or mr > rtol:
        raise RuntimeError('fast_features verify failed for %s: max abs %.3e max rel %.3e (rtol %.1e), '
                           'nonfinite pattern equal %s' % (what, md, mr, rtol, same_nf))


# --------------------------------------------------------------------------------------------- install
def self_check(env, device=None, dtype=torch.float64, seed=0):
    """Batched algebra vs the env's own per-trajectory methods on random inputs; returns (ok, message, max_rel)."""
    cls = type(env)
    for name, sig in (('_merge_features', _MERGE_SIG), ('_reticulate_features', _RET_SIG)):
        f = getattr(cls, name, None)
        if f is None:
            return False, 'env has no %s' % name, None
        params = [p for p in inspect.signature(f).parameters][1:]
        if params != sig:
            return False, '%s signature %s != expected %s' % (name, params, sig), None
    rtol = float(os.environ.get('PHYLOGFN_FAST_FEATURES_RTOL', '0'))               # default: bitwise
    g = torch.Generator().manual_seed(seed)
    dev = device or getattr(env, 'device', None) or 'cpu'
    B, L = 6, env.l_max
    feats = torch.rand(B, L, 2, env.m, env.c, generator=g, dtype=torch.float64).to(dtype=dtype, device=dev)
    worst = 0.0
    cases = [(0, 1, c, last) for c in (0, 1, 2, 3) for last in (False, True)] + [(2, 5, 3, False), (0, L - 1, 1, False)]
    K = torch.tensor([k % B for k in range(len(cases))], device=dev)
    I = torch.tensor([c[0] for c in cases], device=dev)
    J = torch.tensor([c[1] for c in cases], device=dev)
    C = torch.tensor([c[2] for c in cases], device=dev)
    BL = [float(torch.empty(()).uniform_(1e-4, 0.9, generator=g)) for _ in cases]
    BR = [float(torch.empty(()).uniform_(1e-4, 0.9, generator=g)) for _ in cases]
    LAST = [c[3] for c in cases]
    got = batched_merge(env, feats, K, I, J, C, BL, BR, LAST)
    for t, (i, j, c, last) in enumerate(cases):
        ref = cls._merge_features(env, feats[int(K[t])], i, j, c, BL[t], BR[t], last)
        d = (got[t].double() - ref.double()).abs().max().item()
        worst = max(worst, d / max(ref.double().abs().max().item(), 1e-300))
        if not torch.equal(torch.isfinite(got[t]), torch.isfinite(ref)) or d > rtol * max(ref.abs().max().item(), 1e-300):
            return False, 'merge mismatch case=%d last=%s: max abs %.3e' % (c, last, d), worst
    rcases = [(0, 3, 0.3, 0.05), (4, 7, 0.9, 1e-3), (1, L - 1, 0.5, 0.7), (2, 5, 0.1, 0.31), (3, 6, 0.77, 0.002)]
    Kr = torch.tensor([k % B for k in range(len(rcases))], device=dev)
    got = batched_reticulate(env, feats, Kr, torch.tensor([r[0] for r in rcases], device=dev),
                             torch.tensor([r[1] for r in rcases], device=dev), [r[2] for r in rcases],
                             [r[3] for r in rcases])
    for t, (i, n, th, bh) in enumerate(rcases):
        ref = cls._reticulate_features(env, feats[int(Kr[t])], i, n, th, bh)
        d = (got[t].double() - ref.double()).abs().max().item()
        worst = max(worst, d / max(ref.double().abs().max().item(), 1e-300))
        if d > rtol * max(ref.abs().max().item(), 1e-300):
            return False, 'reticulation mismatch: max abs %.3e' % d, worst
    return True, 'ok', worst


def install_fast_features(env, device=None, verbose=True):
    if _mode() == 'off':
        if verbose:
            print('[fast_features] PHYLOGFN_FAST_FEATURES=off: not installed')
        return False
    if getattr(env, '_fast_features_installed', False):
        return True
    ok, msg, worst = self_check(env, device=device)
    if not ok:
        print('[fast_features] NOT installed: self-check failed: %s' % msg)
        return False
    orig_bap = env.batch_apply_actions                                                # bound original
    cls = type(env)
    stats = {'calls': 0, 'planned_merge': 0, 'planned_ret': 0, 'hit_merge': 0, 'hit_ret': 0, 'miss': {},
             'selfcheck_max_rel': worst}
    env._fast_features_stats = stats
    checked = {}

    def miss(reason):
        stats['miss'][reason] = stats['miss'].get(reason, 0) + 1

    def fast_bap(actions, feats, states, *args, **kwargs):
        mode = _mode()
        if mode == 'off' or feats is None or states is None or not torch.is_tensor(feats) or feats.dim() != 5:
            return orig_bap(actions, feats, states, *args, **kwargs)
        dkey = str(feats.device)
        if dkey not in checked:                                   # bitwise self-check on the device actually used
            ok_d, msg_d, worst_d = self_check(env, device=feats.device, dtype=feats.dtype)
            checked[dkey] = ok_d
            stats['selfcheck_%s' % dkey] = (ok_d, msg_d, worst_d)
            if not ok_d:
                print('[fast_features] DISABLED on %s: self-check failed: %s (original per-trajectory path used)'
                      % (dkey, msg_d))
        if not checked[dkey]:
            return orig_bap(actions, feats, states, *args, **kwargs)
        stats['calls'] += 1
        merges, rets = _plan(env, actions, states)
        res = {}
        if merges:
            ks = sorted(merges)
            v = [merges[k] for k in ks]
            dev = feats.device
            out = batched_merge(env, feats, torch.tensor(ks, device=dev), torch.tensor([x[0] for x in v], device=dev),
                                torch.tensor([x[1] for x in v], device=dev), torch.tensor([x[2] for x in v], device=dev),
                                [x[3] for x in v], [x[4] for x in v], [x[5] for x in v])
            for t, k in enumerate(ks):
                res[k] = ('m', merges[k], out[t])
            stats['planned_merge'] += len(ks)
        if rets:
            ks = sorted(rets)
            v = [rets[k] for k in ks]
            dev = feats.device
            out = batched_reticulate(env, feats, torch.tensor(ks, device=dev), torch.tensor([x[0] for x in v], device=dev),
                                     torch.tensor([x[1] for x in v], device=dev), [x[2] for x in v], [x[3] for x in v])
            for t, k in enumerate(ks):
                res[k] = ('r', rets[k], out[t])
            stats['planned_ret'] += len(ks)
        rowkey = {_row_key(feats[k]): k for k in res}
        rtol = float(os.environ.get('PHYLOGFN_FAST_FEATURES_RTOL', '0'))
        used = set()

        def stub_merge(*a, **kw):
            d = _bind(_MERGE_SIG, a, kw)
            if d is None or not torch.is_tensor(d['feats']):
                miss('merge:signature'); return cls._merge_features(env, *a, **kw)
            k = rowkey.get(_row_key(d['feats']))
            if k is None or res[k][0] != 'm':
                miss('merge:unplanned_row'); return cls._merge_features(env, *a, **kw)
            if k in used:
                miss('merge:repeat_call'); return cls._merge_features(env, *a, **kw)
            p = res[k][1]
            if not all(_same(d[n], p[t]) for t, n in enumerate(_MERGE_SIG[1:])):
                miss('merge:args_differ'); return cls._merge_features(env, *a, **kw)
            used.add(k)
            stats['hit_merge'] += 1
            if mode == 'verify':
                _compare(res[k][2], cls._merge_features(env, *a, **kw), rtol, 'merge', stats)
            return res[k][2]

        def stub_ret(*a, **kw):
            d = _bind(_RET_SIG, a, kw)
            if d is None or not torch.is_tensor(d['feats']):
                miss('ret:signature'); return cls._reticulate_features(env, *a, **kw)
            k = rowkey.get(_row_key(d['feats']))
            if k is None or res[k][0] != 'r':
                miss('ret:unplanned_row'); return cls._reticulate_features(env, *a, **kw)
            if k in used:
                miss('ret:repeat_call'); return cls._reticulate_features(env, *a, **kw)
            p = res[k][1]
            if not all(_same(d[n], p[t]) for t, n in enumerate(_RET_SIG[1:])):
                miss('ret:args_differ'); return cls._reticulate_features(env, *a, **kw)
            used.add(k)
            stats['hit_ret'] += 1
            if mode == 'verify':
                _compare(res[k][2], cls._reticulate_features(env, *a, **kw), rtol, 'reticulation', stats)
            return res[k][2]

        env._merge_features = stub_merge
        env._reticulate_features = stub_ret
        try:
            return orig_bap(actions, feats, states, *args, **kwargs)
        finally:
            del env._merge_features
            del env._reticulate_features
            unused = len(res) - len(used)
            if unused:
                miss('planned_but_not_called')
                stats['planned_unused'] = stats.get('planned_unused', 0) + unused

    env.batch_apply_actions = fast_bap
    env._fast_features_installed = True
    if verbose:
        print('[fast_features] installed (mode=%s, self-check max rel diff %.3e)' % (_mode(), worst))
    return True
