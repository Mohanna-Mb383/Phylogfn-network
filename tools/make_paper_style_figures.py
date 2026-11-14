#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import csv
import re

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

DS1_OUT = ROOT / "results" / "DS1" / "paper_methods"
DS1_FIG = DS1_OUT / "figures"
SIM6_FIG = ROOT / "results" / "sim6" / "figures"

DS1_OUT.mkdir(parents=True, exist_ok=True)
DS1_FIG.mkdir(parents=True, exist_ok=True)
SIM6_FIG.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. DS1 complete TB run: merge original + resumed logs
# ============================================================

epoch_re = re.compile(
    r"Epoch\s+(\d+):\s+MLL\s+([-+0-9.eE]+)\s+"
    r"PearsonR\s+([-+0-9.eE]+)\s+"
    r"mean R\s+([-+0-9.eE]+)\s+"
    r"R histogram\s+\[([^\]]+)\]"
)

logs = [
    ROOT / "logs_ds1_complete" / "ds1_baseline_s1.txt",
    ROOT / "logs_ds1_complete" / "ds1_baseline_resume_s1.txt",
]

records = {}

for path in logs:
    if not path.exists():
        print("WARNING: missing", path)
        continue

    for line in path.read_text(errors="replace").splitlines():
        m = epoch_re.search(line)
        if not m:
            continue

        epoch = int(m.group(1))
        hist = [int(x.strip()) for x in m.group(5).split(",")]

        records[epoch] = {
            "epoch": epoch,
            "mll": float(m.group(2)),
            "pearson_r": float(m.group(3)),
            "mean_R": float(m.group(4)),
            "R0": hist[0] if len(hist) > 0 else 0,
            "R1": hist[1] if len(hist) > 1 else 0,
            "R2": hist[2] if len(hist) > 2 else 0,
        }

if not records:
    raise RuntimeError("No DS1 epoch records found")

rows = [records[k] for k in sorted(records)]

csv_path = DS1_OUT / "ds1_tb_metrics.csv"
with csv_path.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("DS1 epochs recovered:", len(rows))
print("DS1 first epoch:", rows[0]["epoch"])
print("DS1 last epoch:", rows[-1]["epoch"])


# MLL figure
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(
    [r["epoch"] for r in rows],
    [r["mll"] for r in rows],
    linewidth=1.5
)
ax.axvline(235, linestyle="--", linewidth=1.0)
ax.set_xlabel("Epoch")
ax.set_ylabel("MLL")
ax.set_title("DS1 TB baseline: marginal log-likelihood")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(DS1_FIG / "ds1_tb_mll_vs_epoch.png", dpi=250)
fig.savefig(DS1_FIG / "ds1_tb_mll_vs_epoch.pdf")
plt.close(fig)


# Pearson figure
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(
    [r["epoch"] for r in rows],
    [r["pearson_r"] for r in rows],
    linewidth=1.5
)
ax.axvline(235, linestyle="--", linewidth=1.0)
ax.set_xlabel("Epoch")
ax.set_ylabel("Pearson correlation")
ax.set_title("DS1 TB baseline: reward-probability correlation")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(DS1_FIG / "ds1_tb_pearson_vs_epoch.png", dpi=250)
fig.savefig(DS1_FIG / "ds1_tb_pearson_vs_epoch.pdf")
plt.close(fig)


# R histogram composition
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ep = np.array([r["epoch"] for r in rows])
r0 = np.array([r["R0"] for r in rows])
r1 = np.array([r["R1"] for r in rows])
r2 = np.array([r["R2"] for r in rows])

ax.plot(ep, r0, label="R=0")
ax.plot(ep, r1, label="R=1")
ax.plot(ep, r2, label="R=2")
ax.axvline(235, linestyle="--", linewidth=1.0)
ax.set_xlabel("Epoch")
ax.set_ylabel("Samples in evaluation batch")
ax.set_title("DS1 TB baseline: reticulation-count distribution")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(DS1_FIG / "ds1_tb_R_histogram_vs_epoch.png", dpi=250)
fig.savefig(DS1_FIG / "ds1_tb_R_histogram_vs_epoch.pdf")
plt.close(fig)


# ============================================================
# 2. sim6 revised: Policy-FCS versus epoch
# ============================================================

variant_order = [
    "tb",
    "db",
    "wdb_logcount",
    "wdb_count",
    "wdb_depth",
    "wdb_logcount_la",
]

labels = {
    "tb": "TB",
    "db": "DB",
    "wdb_logcount": "WDB-logcount",
    "wdb_count": "WDB-count",
    "wdb_depth": "WDB-depth",
    "wdb_logcount_la": "WDB-logcount + Look-Ahead",
}

runs = defaultdict(dict)

for endpoint in sorted(
    (ROOT / "runs_sim6_revised").glob("*/endpoints.csv")
):
    run_name = endpoint.parent.name
    m = re.search(r"_sim6_(.+)_s(\d+)$", run_name)
    if not m:
        continue

    variant = m.group(1)
    seed = int(m.group(2))

    with endpoint.open() as fh:
        data = list(csv.DictReader(fh))

    runs[variant][seed] = data


if runs:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    for variant in variant_order:
        if variant not in runs:
            continue

        by_epoch = defaultdict(list)

        for seed, data in runs[variant].items():
            for row in data:
                by_epoch[int(float(row["epoch"]))].append(
                    float(row["fcs"])
                )

        epochs = np.array(sorted(by_epoch))
        means = np.array([
            np.mean(by_epoch[e]) for e in epochs
        ])
        sds = np.array([
            np.std(by_epoch[e], ddof=1)
            if len(by_epoch[e]) > 1 else 0.0
            for e in epochs
        ])

        line, = ax.plot(
            epochs,
            means,
            linewidth=1.7,
            label=labels.get(variant, variant)
        )

        ax.fill_between(
            epochs,
            means - sds,
            means + sds,
            alpha=0.12,
            color=line.get_color()
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Policy-FCS$_8$ (lower is better)")
    ax.set_title("sim6 revised experiment: distributional error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        SIM6_FIG / "sim6_policy_fcs_vs_epoch.png",
        dpi=250
    )
    fig.savefig(
        SIM6_FIG / "sim6_policy_fcs_vs_epoch.pdf"
    )
    plt.close(fig)

else:
    print("WARNING: no revised sim6 endpoints.csv files found")


# ============================================================
# 3. sim6 revised: final balance residual versus depth
#    Paper-style analogue of depth diagnostics
# ============================================================

profiles = defaultdict(list)

for path in sorted(
    (ROOT / "runs_sim6_revised").glob("*/residual_profile.txt")
):
    run_name = path.parent.name
    m = re.search(r"_sim6_(.+)_s(\d+)$", run_name)
    if not m:
        continue

    variant = m.group(1)

    depth = {}
    active = False

    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("by depth t:"):
            active = True
            continue

        if active and line.startswith("by "):
            break

        if active:
            mm = re.match(
                r"\s*(\d+)\s+([-+0-9.eE]+)\s+\(",
                line
            )
            if mm:
                depth[int(mm.group(1))] = float(mm.group(2))

    if depth:
        profiles[variant].append(depth)


if profiles:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    for variant in variant_order:
        if variant not in profiles:
            continue

        depths = sorted(
            set().union(*[set(x.keys()) for x in profiles[variant]])
        )

        means = []
        for d in depths:
            vals = [
                p[d] for p in profiles[variant] if d in p
            ]
            means.append(np.mean(vals))

        ax.plot(
            depths,
            means,
            marker="o",
            linewidth=1.5,
            label=labels.get(variant, variant)
        )

    ax.set_xlabel("Transition depth")
    ax.set_ylabel("Mean squared balance residual")
    ax.set_yscale("log")
    ax.set_title(
        "sim6 revised experiment: final-checkpoint residual profile"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        SIM6_FIG / "sim6_balance_residual_by_depth.png",
        dpi=250
    )
    fig.savefig(
        SIM6_FIG / "sim6_balance_residual_by_depth.pdf"
    )
    plt.close(fig)

else:
    print("WARNING: no revised sim6 residual_profile.txt files found")


# ============================================================
# 4. DS1 README
# ============================================================

final = records[max(records)]

readme = f"""# DS1 complete TB baseline

## Experiment

- Dataset: `dataset/benchmark_datasets/DS1.pickle`
- Configuration: `src/configs/network_cfgs/cfg_net_DS1_300ep.yaml`
- Objective: Trajectory Balance (TB)
- Device: NVIDIA L40S / CUDA
- Seed: 1
- Requested training budget: 300 epochs

## Interruption and continuation

The original run completed epochs 0--235 and was then interrupted by a
`KeyboardInterrupt`.

Training was continued from
`checkpoint_000235.pt`. The checkpoint restores the model parameters and
optimizer state. The continuation starts at epoch 236.

Because the Python random stream was reinitialized when the resume process
started, the resumed segment is a continuation from the same trained state
but is not bitwise identical to an uninterrupted run.

The dashed vertical line at epoch 235 in the convergence figures marks this
boundary.

## Final result

- Final epoch: {final["epoch"]}
- Final reported MLL: `-7043.143 ± 0.104`
- Final PearsonR: `{final["pearson_r"]:.3f}`
- Final mean reticulation count: `{final["mean_R"]:.2f}`
- Final R histogram: `[{final["R0"]}, {final["R1"]}, {final["R2"]}]`

## Figures

### MLL convergence

![DS1 TB MLL](figures/ds1_tb_mll_vs_epoch.png)

### Pearson correlation

![DS1 TB Pearson](figures/ds1_tb_pearson_vs_epoch.png)

### Reticulation-count distribution

![DS1 TB R histogram](figures/ds1_tb_R_histogram_vs_epoch.png)

## Data

The compact epoch-wise data used for the figures are stored in:

`ds1_tb_metrics.csv`

Raw checkpoints and large run directories are intentionally not committed.
"""

(DS1_OUT / "README.md").write_text(readme)


# ============================================================
# 5. sim6 figure README
# ============================================================

sim_readme = """# sim6 paper-style comparison figures

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
"""

(SIM6_FIG / "README.md").write_text(sim_readme)


# ============================================================
# 6. results index link
# ============================================================

index = ROOT / "results" / "README.md"

if index.exists():
    text = index.read_text()

    marker = "## Paper-method diagnostics"

    if marker not in text:
        text += """

## Paper-method diagnostics

- [DS1 complete TB baseline](DS1/paper_methods/README.md)
- [sim6 revised comparison figures](sim6/figures/README.md)
"""
        index.write_text(text)


print("DONE")
print("DS1 output:", DS1_OUT)
print("sim6 figures:", SIM6_FIG)
