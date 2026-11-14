# sim6 paper-style comparison figures

These figures summarize the revised sim6 experiment after the continuous
Gaussian numerical stabilization.

## Policy-FCS versus epoch

`sim6_policy_fcs_vs_epoch.png`

The curve shows mean Policy-FCS_8 across the three seeds. The shaded region
is one standard deviation. Lower FCS indicates better agreement between the
learned terminal distribution and the target distribution.

## Balance residual versus depth

`sim6_balance_residual_by_depth.png`

This is a depth-wise transition-balance diagnostic at the final checkpoint,
averaged over the available seeds for each DB/WDB method.

It is inspired by the depth-wise balance-violation diagnostics used by
da Silva et al. (ICLR 2025), but it is not a reproduction of their figure:
our state space, residual definition, network MDP, data, and evaluation
protocol are those of PhyloGFN-Net.
