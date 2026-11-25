"""
Proposal pipeline stage (1) "Embed": one vector per taxon, used by the k-NN candidate filter.

    KMER         k-mer frequency profile (4^k features, gaps and ambiguity codes skipped);
                 the supervisors' recommended alignment-free baseline (k = 4, 5, 6).
    ONEHOT       flattened one-hot of the ALIGNED sequence; with the cosine metric its
                 distance is (up to gaps) the p-distance ("Hamming" baseline of the deck).
    PRECOMPUTED  vectors computed elsewhere (e.g. DNABERT-2 run on a GPU machine):
                 .npy of shape [N, d] in the order of the sequence file, or a pickle holding
                 a list / array in that order.
    DNABERT2     Path A of the proposal: mean-pooled last hidden state of the genomic language
                 model zhihan1996/DNABERT-2-117M on the gap-free sequence.  Needs
                 `pip install transformers einops` and a model download; NOT exercised in the
                 delivery environment (no model access there) -- use PRECOMPUTED for a run
                 whose embeddings were made with tools/embed_dnabert2.py.

All vectors are L2-normalised when the metric is COSINE.
"""
import itertools
import pickle
import numpy as np

_BASES = 'ACGT'


def kmer_profile(seq, k):
    seq = seq.upper().replace('U', 'T')
    index = {''.join(p): i for i, p in enumerate(itertools.product(_BASES, repeat=k))}
    v = np.zeros(len(index), dtype=np.float64)
    for i in range(len(seq) - k + 1):
        w = seq[i:i + k]
        j = index.get(w)
        if j is not None:
            v[j] += 1.0
    if v.sum() > 0:
        v /= v.sum()
    return v


def onehot_profile(seq):
    seq = seq.upper().replace('U', 'T')
    v = np.zeros((len(seq), 4), dtype=np.float64)
    for i, ch in enumerate(seq):
        j = _BASES.find(ch)
        if j >= 0:
            v[i, j] = 1.0
    return v.reshape(-1)


def p_distance_matrix(seqs):
    """fraction of mismatching sites among sites where both sequences carry a base"""
    n = len(seqs)
    D = np.zeros((n, n))
    arr = [np.array(list(s.upper().replace('U', 'T'))) for s in seqs]
    ok = [np.isin(a, list(_BASES)) for a in arr]
    for i in range(n):
        for j in range(i + 1, n):
            both = ok[i] & ok[j]
            D[i, j] = D[j, i] = float(np.mean(arr[i][both] != arr[j][both])) if both.any() else 1.0
    return D


def load_precomputed(path, n):
    if path.endswith('.npy'):
        E = np.load(path)
    else:
        obj = pickle.load(open(path, 'rb'))
        E = np.asarray(list(obj.values()) if isinstance(obj, dict) else obj)
    E = np.asarray(E, dtype=np.float64)
    assert E.shape[0] == n, f'{path}: {E.shape[0]} vectors for {n} sequences'
    return E


def dnabert2_embeddings(seqs, model_name='zhihan1996/DNABERT-2-117M', device='cpu', max_len=512):
    """Path A.  Mean-pooled last hidden state over the tokens of the gap-free sequence.
    Sequences longer than max_len tokens are embedded in chunks and the chunk vectors averaged."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:                                   # pragma: no cover
        raise ImportError('DNABERT-2 embeddings need `pip install transformers einops` and model access') from e
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    out = []
    with torch.no_grad():
        for s in seqs:
            s = ''.join(ch for ch in s.upper().replace('U', 'T') if ch in _BASES)
            ids = tok(s, return_tensors='pt')['input_ids'][0]
            vecs = []
            for start in range(0, len(ids), max_len):
                chunk = ids[start:start + max_len].unsqueeze(0).to(device)
                hidden = model(chunk)[0]                       # [1, T, d]
                vecs.append(hidden[0].mean(0).cpu().numpy())
            out.append(np.mean(vecs, axis=0))
    return np.stack(out)


def build_leaf_embeddings(knn_cfg, seqs, metric):
    kind = knn_cfg.EMBEDDING
    if kind == 'KMER':
        E = np.stack([kmer_profile(s, int(knn_cfg.KMER_K)) for s in seqs])
    elif kind in ('ONEHOT', 'PDIST'):
        E = np.stack([onehot_profile(s) for s in seqs])
    elif kind == 'PRECOMPUTED':
        E = load_precomputed(knn_cfg.EMBEDDING_PATH, len(seqs))
    elif kind == 'DNABERT2':
        E = dnabert2_embeddings(seqs, knn_cfg.DNABERT2_MODEL)
    else:
        raise ValueError(f'unknown embedding kind {kind}')
    E = E.astype(np.float32)
    if metric == 'COSINE':
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        E = E / norms
    return E
