"""Plot MLL and the sampled-R distribution per epoch from one or more training runs.

    python tools/plot_history.py OUT.pdf 'label 1::runs/<run1>' 'label 2::runs/<run2>' ...
"""
import sys, pickle, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAVY, TEAL, ORANGE, GREY = '#12213A', '#0E7C86', '#B3441E', '#8FA5B2'


def load(run):
    cands = sorted(glob.glob(run + '*'))
    path = [c for c in cands if os.path.isdir(c)][-1]
    return pickle.load(open(os.path.join(path, 'history.pkl'), 'rb'))


def main():
    out = sys.argv[1]
    runs = [a.split('::', 1) for a in sys.argv[2:]]
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 5.2), squeeze=False)
    for k, (label, run) in enumerate(runs):
        h = [r for r in load(run) if 'mll' in r]
        ep = [r['epoch'] for r in h]
        ax = axes[0, k]
        ax.plot(ep, [r['mll'] for r in h], color=TEAL, marker='o', ms=3)
        ax.set_title(label, fontsize=10, color=NAVY)
        ax.set_ylabel('MLL estimate' if k == 0 else '')
        ax.grid(alpha=0.3)
        ax2 = axes[1, k]
        H = np.array([r['hist_R'] for r in h], dtype=float)
        H = H / H.sum(1, keepdims=True)
        cols = [GREY, TEAL, ORANGE, NAVY][:H.shape[1]]
        ax2.stackplot(ep, H.T, labels=[f'R = {i}' for i in range(H.shape[1])], colors=cols, alpha=0.9)
        ax2.set_ylim(0, 1); ax2.set_xlabel('epoch'); ax2.set_ylabel('sampled R (share)' if k == 0 else '')
        if k == n - 1:
            ax2.legend(loc='lower right', fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
