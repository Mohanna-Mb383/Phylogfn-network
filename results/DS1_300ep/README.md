# DS1 300-epoch network result

## Annotated network figure

Below is one sampled level-1 phylogenetic network from the trained DS1 model (epoch 299 checkpoint).

![Annotated DS1 network](network_radial_v3_annotated.png)

## What this figure shows

- This is a sampled phylogenetic **network**, not a tree.
- The figure comes from the trained **PhyloGFN-Net** model on **DS1**.
- The network contains **2 reticulation nodes**: **H1** and **H2**.
- Gray edge labels show **branch lengths**.
- Reticulation edges also show **inheritance weight (theta)**.
- The root is shown at the center/top of the network layout.

## Files

- `network_radial_v3_annotated.png`  
  Annotated network figure with branch lengths and inheritance weights.

- `network_radial_v3_edges.txt`  
  Text table containing the numerical branch lengths and inheritance weights.

## Interpretation

This result shows that the trained model can generate a level-1 phylogenetic network and assign:
- branch lengths to edges,
- and inheritance probabilities to reticulation parents.

This figure is one sampled example from the checkpoint at epoch 299.
