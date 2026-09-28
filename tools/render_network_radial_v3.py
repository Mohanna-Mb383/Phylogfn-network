import os
import math
import pickle
import runpy

import numpy as np
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA = "dataset/benchmark_datasets/DS1.pickle"

OUT = "results/DS1_300ep/network_radial_v3_annotated.png"
PARAMS = "results/DS1_300ep/network_radial_v3_edges.txt"


# ============================================================
# Get exactly the same trained-network sample
# ============================================================

d = runpy.run_path("tools/render_network_now.py")

state = d["state"]
net = state.network
root = net.root

with open(DATA, "rb") as f:
    dataset = pickle.load(f)

taxa = list(dataset.keys())


# ============================================================
# Traverse ACTUAL NetNode objects
# ============================================================

queue = [root]
seen = set()
node_by_idx = {}

while queue:
    n = queue.pop(0)

    if id(n) in seen:
        continue

    seen.add(id(n))
    node_by_idx[n.idx] = n

    for child in n.children:
        queue.append(child)


# ============================================================
# Build actual network graph with branch lengths
# ============================================================

G = nx.DiGraph()

for idx, n in node_by_idx.items():
    G.add_node(
        idx,
        kind=n.kind,
        ret_id=n.ret_id,
    )

for idx, n in node_by_idx.items():

    for child, dist in zip(
        n.children,
        n.child_dists
    ):
        G.add_edge(
            idx,
            child.idx,
            length=float(dist),
        )


root_idx = root.idx

retic_nodes = [
    idx
    for idx, n in node_by_idx.items()
    if n.kind == "ret"
]

leaf_nodes = [
    idx
    for idx, n in node_by_idx.items()
    if n.kind == "leaf"
]

internal_nodes = [
    idx
    for idx, n in node_by_idx.items()
    if n.kind == "tree"
    and idx != root_idx
]


print("actual nodes =", G.number_of_nodes())
print("actual edges =", G.number_of_edges())
print("reticulations =", retic_nodes)


# ============================================================
# Map reticulation idx -> H1/H2
# ============================================================

h_name = {}

for idx in retic_nodes:
    ret_id = node_by_idx[idx].ret_id
    h_name[idx] = f"H{ret_id + 1}"


# ============================================================
# Build spanning TREE only for positioning
# ============================================================

T = nx.DiGraph()
T.add_nodes_from(G.nodes())

for node in G.nodes():

    if node == root_idx:
        continue

    parents = list(G.predecessors(node))

    if not parents:
        continue

    # Reticulation: slot 0 parent is used only for layout.
    # Both real parents will still be drawn.
    if node in retic_nodes:

        r = node_by_idx[node]

        preferred = None

        if r.parents[0] is not None:
            preferred = r.parents[0].idx

        if preferred in parents:
            parent = preferred
        else:
            parent = parents[0]

    else:
        parent = parents[0]

    T.add_edge(parent, node)


# ============================================================
# Leaf order using topology, not alphabetically
# ============================================================

ordered_leaves = []


def collect_leaves(node):

    children = list(T.successors(node))

    if not children:

        if node in leaf_nodes:
            ordered_leaves.append(node)

        return

    for child in children:
        collect_leaves(child)


collect_leaves(root_idx)

for leaf in leaf_nodes:
    if leaf not in ordered_leaves:
        ordered_leaves.append(leaf)


N = len(ordered_leaves)

print("leaves =", N)


# ============================================================
# Circular leaf positions
# ============================================================

leaf_angle = {}

for i, leaf in enumerate(ordered_leaves):

    # start at top and go clockwise
    angle = (
        math.pi / 2
        - 2.0 * math.pi * i / N
    )

    leaf_angle[leaf] = angle


# ============================================================
# Descendant leaves
# ============================================================

memo = {}


def descendant_leaves(node):

    if node in memo:
        return memo[node]

    if node in leaf_angle:
        memo[node] = [node]
        return memo[node]

    result = []

    for child in T.successors(node):
        result.extend(
            descendant_leaves(child)
        )

    memo[node] = result

    return result


def circular_mean(angles):

    if not angles:
        return 0.0

    x = sum(math.cos(a) for a in angles)
    y = sum(math.sin(a) for a in angles)

    return math.atan2(y, x)


# ============================================================
# Depth
# ============================================================

depth = {root_idx: 0}

queue = [root_idx]

while queue:

    u = queue.pop(0)

    for v in T.successors(u):

        if v not in depth:
            depth[v] = depth[u] + 1
            queue.append(v)


max_depth = max(depth.values())


# ============================================================
# Coordinates
# ============================================================

pos = {}

for node in T.nodes():

    if node == root_idx:
        pos[node] = (0.0, 0.0)
        continue

    if node in leaf_angle:
        angle = leaf_angle[node]

    else:

        leaves = descendant_leaves(node)

        angles = [
            leaf_angle[x]
            for x in leaves
            if x in leaf_angle
        ]

        angle = circular_mean(angles)

    r = depth.get(node, 0) / max_depth

    if node in leaf_nodes:
        r = 1.0

    pos[node] = (
        r * math.cos(angle),
        r * math.sin(angle),
    )


# ============================================================
# Edge categories
# ============================================================

retic_edges = [
    (u, v)
    for u, v in G.edges()
    if v in retic_nodes
]

normal_edges = [
    (u, v)
    for u, v in G.edges()
    if v not in retic_nodes
]


# ============================================================
# θ for an incoming reticulation edge
# ============================================================

def inheritance_weight(parent_idx, ret_idx):

    h = node_by_idx[ret_idx]

    if h.parents[0] is not None:
        if h.parents[0].idx == parent_idx:
            return float(h.theta)

    if h.parents[1] is not None:
        if h.parents[1].idx == parent_idx:
            return 1.0 - float(h.theta)

    return None


# ============================================================
# Edge label position
# ============================================================

def midpoint_offset(u, v, amount=0.012):

    x1, y1 = pos[u]
    x2, y2 = pos[v]

    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0

    dx = x2 - x1
    dy = y2 - y1

    norm = math.sqrt(
        dx * dx + dy * dy
    )

    if norm > 0:

        x += (-dy / norm) * amount
        y += (dx / norm) * amount

    return x, y


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(24, 24)
)


# Ordinary branches
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=normal_edges,
    ax=ax,
    arrows=False,
    width=1.5,
    edge_color="#98a3aa",
)


# Reticulation incoming branches
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=retic_edges,
    ax=ax,
    arrows=False,
    width=3.5,
    edge_color="#168f95",
)


# Internal tree nodes
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=internal_nodes,
    ax=ax,
    node_size=100,
    node_color="#98a3aa",
    edgecolors="none",
)


# Leaves
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=leaf_nodes,
    ax=ax,
    node_size=115,
    node_color="#98a3aa",
    edgecolors="none",
)


# Reticulation nodes
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=retic_nodes,
    ax=ax,
    node_size=750,
    node_color="#70aeb1",
    edgecolors="#4c7779",
    linewidths=1.2,
)


# Root
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=[root_idx],
    ax=ax,
    node_size=780,
    node_color="#687b84",
    edgecolors="none",
)


# ============================================================
# ALL branch-length labels
# ============================================================

for u, v, data in G.edges(data=True):

    b = float(data["length"])

    # Reticulation incoming edge gets special label later
    if v in retic_nodes:
        continue

    x, y = midpoint_offset(
        u, v, 0.008
    )

    ax.text(
        x,
        y,
        f"{b:.4f}",
        fontsize=5.5,
        color="#68757b",
        ha="center",
        va="center",
        bbox=dict(
            facecolor="white",
            alpha=0.60,
            edgecolor="none",
            pad=0.25,
        ),
    )


# ============================================================
# Reticulation labels: branch length + theta
# ============================================================

for u, v in retic_edges:

    b = float(
        G.edges[u, v]["length"]
    )

    theta = inheritance_weight(
        u, v
    )

    x, y = midpoint_offset(
        u, v, 0.018
    )

    if theta is None:
        text = f"b={b:.5f}"
    else:
        text = (
            f"b={b:.5f}\n"
            f"θ={theta:.3f}"
        )

    ax.text(
        x,
        y,
        text,
        fontsize=9,
        fontweight="bold",
        color="#087b80",
        ha="center",
        va="center",
        bbox=dict(
            facecolor="white",
            alpha=0.90,
            edgecolor="#7db7b9",
            linewidth=0.7,
            pad=1.2,
        ),
    )


# ============================================================
# H1/H2 labels
# ============================================================

for h in retic_nodes:

    x, y = pos[h]

    ax.text(
        x,
        y,
        h_name[h],
        fontsize=14,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )


# Root label
x, y = pos[root_idx]

ax.text(
    x,
    y,
    "root",
    fontsize=13,
    fontweight="bold",
    color="white",
    ha="center",
    va="center",
)


# ============================================================
# Taxon labels
# ============================================================

LABEL_R = 1.075

for leaf in ordered_leaves:

    angle = leaf_angle[leaf]

    tx = LABEL_R * math.cos(angle)
    ty = LABEL_R * math.sin(angle)

    degrees = math.degrees(angle)

    if -90 <= degrees <= 90:
        rotation = degrees
        ha = "left"
    else:
        rotation = degrees + 180
        ha = "right"

    if 0 <= leaf < len(taxa):
        label = taxa[leaf].replace("_", " ")
    else:
        label = f"L{leaf}"

    ax.text(
        tx,
        ty,
        label,
        fontsize=8.5,
        color="#66757b",
        rotation=rotation,
        rotation_mode="anchor",
        ha=ha,
        va="center",
    )


# ============================================================
# Title + legend
# ============================================================

ax.text(
    -1.30,
    1.27,
    (
        "DS1 sampled level-1 phylogenetic network\n"
        f"checkpoint epoch 299 | "
        f"log score = {state.log_score:.2f} | "
        f"R = {state.n_ret}"
    ),
    fontsize=15,
    color="#596b72",
    ha="left",
)


ax.text(
    -1.30,
    -1.27,
    (
        "Gray edge label = branch length (b)\n"
        "Teal reticulation edge label = branch length (b) "
        "and inheritance probability (θ)"
    ),
    fontsize=10,
    color="#65767c",
    ha="left",
)


ax.set_xlim(-1.34, 1.34)
ax.set_ylim(-1.34, 1.34)

ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()


os.makedirs(
    os.path.dirname(OUT),
    exist_ok=True
)

plt.savefig(
    OUT,
    dpi=350,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Save numerical edge table
# ============================================================

with open(PARAMS, "w") as f:

    f.write(
        f"log_score\t{state.log_score}\n"
    )

    f.write(
        f"R\t{state.n_ret}\n\n"
    )

    f.write(
        "parent\tchild\tbranch_length\tinheritance_theta\n"
    )

    for u, v, data in G.edges(data=True):

        b = float(data["length"])

        if v in retic_nodes:

            theta = inheritance_weight(
                u, v
            )

            f.write(
                f"{u}\t{h_name[v]}"
                f"\t{b:.10f}"
                f"\t{theta:.10f}\n"
            )

        else:

            child_name = (
                h_name[v]
                if v in retic_nodes
                else str(v)
            )

            f.write(
                f"{u}\t{child_name}"
                f"\t{b:.10f}\t-\n"
            )


print()
print("ANNOTATED NETWORK CREATED:")
print(OUT)

print()
print("EDGE PARAMETER TABLE:")
print(PARAMS)

print()

for h in retic_nodes:

    obj = node_by_idx[h]

    print(h_name[h])
    print(
        " parent 1:",
        obj.parents[0].idx,
        "b =", obj.parent_dists[0],
        "theta =", obj.theta,
    )
    print(
        " parent 2:",
        obj.parents[1].idx,
        "b =", obj.parent_dists[1],
        "theta =", 1.0 - obj.theta,
    )
    print(
        " child:",
        obj.children[0].idx,
        "b =", obj.child_dist,
    )
