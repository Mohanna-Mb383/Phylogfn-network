"""
Simulate an alignment on a known random level-1 network under the displayed-tree model
(every site follows one displayed tree, chosen with the inheritance weights), Jukes-Cantor.

Usage:
    python tools/simulate_network_data.py OUT.pickle [--leaves=6] [--sites=300] [--rets=1] [--seed=0] [--theta=0.5]

The true network is written next to the alignment as OUT.truth.txt (edge list + theta),
so the sampler's posterior over R and over reticulation placements can be compared with
the truth (validation slides: O1 / O3 small-N checks).
"""
import sys, random, pickle, math
import numpy as np


def is_level1(children, rets):
    """every biconnected component of the underlying undirected graph holds <= 1 reticulation node"""
    import networkx as nx
    G = nx.Graph()
    for p, lst in children.items():
        for c, _ in lst:
            G.add_edge(p, c)
    for comp in nx.biconnected_components(G):
        if len(comp & set(rets)) > 1:
            return False
    return True


def reaches(children, src, dst):
    stack, seen = [src], set()
    while stack:
        w = stack.pop()
        if w == dst:
            return True
        if w in seen:
            continue
        seen.add(w)
        stack.extend(c for c, _ in children.get(w, []))
    return False


def simulate(n_leaves=6, sites=300, rets=1, seed=0, theta=0.5):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    # ---- random binary tree by random merges
    lineages = list(range(n_leaves))
    children = {}
    nxt = n_leaves
    while len(lineages) > 1:
        a, b = rng.sample(lineages, 2)
        children[nxt] = [(a, rng.uniform(0.05, 0.3)), (b, rng.uniform(0.05, 0.3))]
        lineages.remove(a); lineages.remove(b); lineages.append(nxt); nxt += 1
    root = nxt - 1
    # ---- add reticulations: h above an existing node x (edge p -> x becomes p -> h -> x),
    #      second parent u subdividing another edge a -> b, plus the edge u -> h
    rets_info = {}
    tries = 0
    while len(rets_info) < rets and tries < 2000:
        tries += 1
        parents = {c: p for p, lst in children.items() for c, _ in lst}
        x = rng.choice([v for v in parents if v not in rets_info])
        p = parents[x]
        a, b = rng.choice([(a, b) for a, lst in children.items() for b, _ in lst if (a, b) != (p, x)])
        if reaches(children, x, a):          # u would be below x: directed cycle
            continue
        h, u = nxt, nxt + 1
        trial = {k: list(v) for k, v in children.items()}
        trial[p] = [(h if c == x else c, d) for c, d in trial[p]]
        trial[h] = [(x, rng.uniform(0.05, 0.3))]
        d_ab = [d for c, d in trial[a] if c == b][0]
        trial[a] = [(u if c == b else c, d) for c, d in trial[a]]
        trial[u] = [(b, d_ab / 2), (h, rng.uniform(0.05, 0.3))]
        if not is_level1(trial, list(rets_info) + [h]):
            continue
        children = trial
        nxt += 2
        rets_info[h] = (p, u, theta)
    # ---- simulate sites under Jukes-Cantor
    def jc(d):
        q = 0.25 * (1 - math.exp(-4 * d / 3))
        M = np.full((4, 4), q); np.fill_diagonal(M, 1 - 3 * q); return M
    seqs = np.zeros((n_leaves, sites), dtype=int)
    for i in range(sites):
        keep = {h: (p0 if nprng.random() < t else p1) for h, (p0, p1, t) in rets_info.items()}
        state = {root: int(nprng.integers(4))}
        stack = [root]
        while stack:
            v = stack.pop()
            for c, d in children.get(v, []):
                if c in rets_info and keep[c] != v:
                    continue                      # this site does not inherit through this parent edge
                state[c] = int(nprng.choice(4, p=jc(d)[state[v]]))
                stack.append(c)
        for leaf in range(n_leaves):
            seqs[leaf, i] = state[leaf]
    alphabet = np.array(list('ACGT'))
    strings = [''.join(alphabet[seqs[k]]) for k in range(n_leaves)]
    truth = {'children': children, 'reticulations': rets_info, 'root': root}
    return strings, truth


if __name__ == '__main__':
    out = sys.argv[1]
    kw = {}
    names = {'leaves': 'n_leaves', 'sites': 'sites', 'rets': 'rets', 'seed': 'seed', 'theta': 'theta'}
    for a in sys.argv[2:]:
        k, v = a.lstrip('-').split('=')
        kw[names[k]] = float(v) if k == 'theta' else int(v)
    strings, truth = simulate(**kw)
    pickle.dump({'taxon_%d' % k: s for k, s in enumerate(strings)}, open(out, 'wb'))   # same format as DS1-DS8
    with open(out.replace('.pickle', '') + '.truth.txt', 'w') as f:
        f.write('root %d\n' % truth['root'])
        for p, lst in truth['children'].items():
            for c, d in lst:
                f.write('edge %d %d %.4f\n' % (p, c, d))
        for h, (p0, p1, t) in truth['reticulations'].items():
            f.write('reticulation %d parents %d %d theta %.3f\n' % (h, p0, p1, t))
    print('wrote', out, 'with', len(strings), 'sequences of', len(strings[0]), 'sites;',
          len(truth['reticulations']), 'reticulation(s)')
