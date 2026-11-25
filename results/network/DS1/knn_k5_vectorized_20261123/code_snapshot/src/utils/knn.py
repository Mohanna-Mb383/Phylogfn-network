"""
k-NN candidate filtering (proposal objective O2, pipeline stage (2) "Filter").

Every open lineage carries an embedding vector (stage (1), see src/embeddings/).  At each
construction step the policy scores, for each open lineage i, only the MERGE partners among
the k nearest lineages of i in embedding space; the union of these pairs (symmetric) is the
candidate set.  Two ways to obtain the neighbour lists:

  EXACT        recompute all pairwise distances among the l open lineages every step:
               O(l^2 d) arithmetic, but no MLP scoring of the non-candidates -- the term that
               costs 78 % of a PhyloGFN step (proposal, Evidence 2) becomes O(l k).
  INCREMENTAL  keep a neighbour list per lineage and update it locally after every action
               (merge: the new ancestor inherits the union of its children's lists and every
               list that mentioned a child is patched; reticulate: the two new slots copy the
               child's list).  O(l k d) per step.  Lists are approximate after the first step;
               Recall@k against exact scoring is the proposal's own measure of that (T9).

Ancestor embeddings (stage (1), Path B "Dirichlet-energy structural embeddings computed on the
partial graph"): the embedding of a new ancestor w with children u, v and branch lengths
b_u, b_v is the minimiser of the Dirichlet energy  (1/b_u)|e_w - e_u|^2 + (1/b_v)|e_w - e_v|^2,
i.e. the 1/b-weighted mean of the children (harmonic extension, computed bottom-up).
'MEAN' ignores the branch lengths.
"""
import numpy as np

_EPS_B = 1e-3     # branch-length floor in the 1/b weights


# ------------------------------------------------------------------------------- distances
def normalise(e, metric):
    if metric == 'COSINE':
        n = np.linalg.norm(e)
        return e / n if n > 0 else e
    return e


def distance(e1, e2, metric):
    if metric == 'COSINE':
        return float(1.0 - np.dot(e1, e2))           # embeddings are L2-normalised
    return float(np.linalg.norm(e1 - e2))


def distance_matrix(E, metric):
    """E [l, d] -> [l, l]"""
    if metric == 'COSINE':
        return 1.0 - E @ E.T
    sq = np.sum(E * E, axis=1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (E @ E.T), 0.0))


# ------------------------------------------------------------------------------- embeddings of ancestors
def combine_embeddings(eu, ev, bu, bv, propagation, metric):
    if eu is None or ev is None:
        return None
    if propagation == 'DIRICHLET':
        wu, wv = 1.0 / (bu + _EPS_B), 1.0 / (bv + _EPS_B)
        e = (wu * eu + wv * ev) / (wu + wv)
    else:
        e = 0.5 * (eu + ev)
    return normalise(e.astype(np.float32), metric)


# ------------------------------------------------------------------------------- EXACT mode
def exact_neighbour_pairs(lineages, k, metric):
    """set of (i, j), i < j, such that j is among the k nearest of i or i among the k nearest of j"""
    E = np.stack([l.emb for l in lineages]).astype(np.float64)
    D = distance_matrix(E, metric)
    np.fill_diagonal(D, np.inf)
    l = len(lineages)
    k = min(k, l - 1)
    nn = np.argpartition(D, k - 1, axis=1)[:, :k]           # k smallest per row (unordered)
    pairs = set()
    for i in range(l):
        for j in nn[i]:
            j = int(j)
            pairs.add((i, j) if i < j else (j, i))
    return pairs


# ------------------------------------------------------------------------------- INCREMENTAL mode
def init_neighbour_lists(lineages, k, metric):
    E = np.stack([l.emb for l in lineages]).astype(np.float64)
    D = distance_matrix(E, metric)
    np.fill_diagonal(D, np.inf)
    l = len(lineages)
    kk = min(k, l - 1)
    for i, lin in enumerate(lineages):
        order = np.argsort(D[i])[:kk]
        lin.nbrs = [(lineages[j].key, float(D[i, j])) for j in order]


def _trim(nbrs, k):
    nbrs.sort(key=lambda t: t[1])
    return nbrs[:k]


def update_neighbours_after_merge(new_lineages, w, u, v, k, metric):
    """w replaced u and v.  new_lineages already contains w (and not u, v)."""
    by_key = {l.key: l for l in new_lineages}
    # w's list: best k of the union of u's and v's lists (re-measured from w)
    cand = {}
    for src in (u, v):
        for key, _ in (src.nbrs or []):
            if key in by_key and key != w.key:
                cand[key] = distance(w.emb, by_key[key].emb, metric)
    w.nbrs = _trim([(key, d) for key, d in cand.items()], k)
    # every other list that mentioned u or v now mentions w instead
    dropped = {u.key, v.key}
    for l in new_lineages:
        if l is w or l.nbrs is None:
            continue
        if any(key in dropped for key, _ in l.nbrs):
            kept = [(key, d) for key, d in l.nbrs if key not in dropped and key in by_key]
            kept.append((w.key, distance(l.emb, w.emb, metric)))
            l.nbrs = _trim(kept, k)
        else:
            l.nbrs = [(key, d) for key, d in l.nbrs if key in by_key]


def update_neighbours_after_reticulate(new_lineages, p1, p2, x, k, metric):
    """x was replaced by the slot lineages p1 (same key as x) and p2 (fresh key), same embedding."""
    base = [(key, d) for key, d in (x.nbrs or []) if key != p2.key]
    p1.nbrs = list(base)
    p2.nbrs = list(base)


def incremental_neighbour_pairs(lineages):
    pos = {l.key: i for i, l in enumerate(lineages)}
    pairs = set()
    for i, l in enumerate(lineages):
        for key, _ in l.nbrs:
            j = pos.get(key)
            if j is None or j == i:
                continue
            pairs.add((i, j) if i < j else (j, i))
    return pairs
