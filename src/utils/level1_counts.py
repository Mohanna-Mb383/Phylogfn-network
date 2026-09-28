"""
Exact number of rooted binary leaf-labelled level-1 networks with n leaves and k
reticulations, from Proposition 5.4 of Bouvel, Gambette & Mansouri (2020),
"Counting phylogenetic networks of level 1 and 2", J. Math. Biol. 81:1357-1395:

    r(n, k, m) = sum_{p=0}^{k} (2n+3k-m-2)! (m-2k-1)! 2^{p+m+1-n-3k}
                 / [ (n+2k-m-1)! p! (k-p)! (m-4k+p)! (2k-p-1)! ]          (k >= 1)

summed over the number m of inner arcs; r(n, 0) = (2n-3)!! (rooted binary trees).
Checked against the paper's Table 2 and against the direct enumeration of the MDP
(tests/test_network_env.py): 3/21/12, 15/228/360/120, 105/2805/8550/7140/1680, ...

Used for the topology prior  p(G) = exp(-lambda_R R(G)) / Z_lambda,
    Z_lambda = sum_{k=0}^{R_MAX} r(n, k) exp(-lambda_R k),
whose log enters the marginal-likelihood estimate (the analogue of PhyloGFN's
`tree_factor = -log (2n-5)!!`, the uniform prior over unrooted tree topologies).
"""
from fractions import Fraction
from functools import lru_cache
from math import factorial, log, exp


@lru_cache(maxsize=None)
def r_nkm(n, k, m):
    tot = Fraction(0)
    for p in range(0, k + 1):
        args = [2 * n + 3 * k - m - 2, m - 2 * k - 1, n + 2 * k - m - 1, p, k - p, m - 4 * k + p, 2 * k - p - 1]
        if any(a < 0 for a in args):
            continue
        num = factorial(args[0]) * factorial(args[1]) * Fraction(2) ** (p + m + 1 - n - 3 * k)
        den = factorial(args[2]) * factorial(args[3]) * factorial(args[4]) * factorial(args[5]) * factorial(args[6])
        tot += num / den
    return tot


@lru_cache(maxsize=None)
def level1_count(n, k):
    """number of level-1 networks with n labelled leaves and exactly k reticulations"""
    if k == 0:
        v = 1
        for i in range(1, 2 * n - 2, 2):
            v *= i
        return v
    tot = sum(r_nkm(n, k, m) for m in range(1, 4 * n + 4 * k))
    assert tot.denominator == 1
    return int(tot)


def log_topology_prior_normalizer(n, r_max, lambda_r, prior_type='EXP_R'):
    """EXP_R:     log Z = log sum_{k<=r_max} r(n,k) exp(-lambda_r k)   (exact integers, log-sum-exp)
       UNIFORM_R: log Z = log sum_{k<=r_max} exp(-lambda_r k)          (the within-R uniform part is already in the reward)"""
    if prior_type == 'UNIFORM_R':
        terms = [-lambda_r * k for k in range(0, r_max + 1)]
    else:
        terms = [log(level1_count(n, k)) - lambda_r * k for k in range(0, r_max + 1)]
    mx = max(terms)
    return mx + log(sum(exp(t - mx) for t in terms))


def log_topology_prior(n, r_max, lambda_r, k):
    """log p(G) for a network with k reticulations"""
    return -lambda_r * k - log_topology_prior_normalizer(n, r_max, lambda_r)


if __name__ == '__main__':
    for n in range(3, 9):
        print(n, [level1_count(n, k) for k in range(0, 4)])
