# Geometric Deep Learning — Synthetic Retinotopy

Seminar project for "Advanced Deep Learning" (Data Science, Uni Leipzig).

---

## The question

Does a graph neural network learn brain-to-visual-field mapping better when it uses **hyperbolic distances** instead of Euclidean (straight-line) distances between nodes?

---

## Background

The visual cortex (the part of the brain that processes what you see) is organised as a map: each spot on the brain surface corresponds to a specific spot in your visual field. This mapping is called **retinotopy**.

The mapping is distorted: the center of your vision (fovea) gets far more brain tissue than the periphery — even though both span the same amount of visual angle. This is called **cortical magnification**.

The consequence for graph neural networks: two neighboring brain nodes that are 1mm apart can represent very different "distances" in the visual field, depending on whether they sit in the fovea or the periphery. A network using Euclidean distances treats both cases identically. Hyperbolic distances compress large gaps and could better reflect the true visual-field structure.

---

## Approach

1. **Synthetic data** — we generate a brain graph with a known ground-truth retinotopic map (so we can measure how wrong the model is)
2. **Two conditions** — train the same GNN twice: once with Euclidean edge weights, once with hyperbolic edge weights (controlled by a curvature parameter κ)
3. **Compare** — which condition has lower prediction error?

Based on deepRetinotopy (Ribeiro et al. 2021). Stack: PyTorch Geometric, geoopt, nilearn.

---

## Setup

```bash
pip install -r requirements.txt
```

Generate the data:
```bash
python data/generate_data.py data/config.yaml
```

This creates two files in `data/`:
- `retinotopy_groundtruth.pt` — the brain graph + ground-truth map + simulated fMRI signals
- `curved_metrics_kappa.pt` — hyperbolic edge weights for each κ value

See `data/data.md` for a full description of every field.

---

## Data at a glance

| | |
|--|--|
| Nodes | 17,263 (visual cortex vertices) |
| Edges | 102,432 directed (51,216 undirected mesh edges × 2) |
| Node input (`data.x`) | pRF response to 100 stimuli — shape `(17263, 100)` |
| Prediction target (`data.y`) | `[eccentricity°, polar_angle_rad]` — shape `(17263, 2)` |
| Train / Val / Test | 70% / 15% / 15% (node masks, configurable in `config.yaml`) |
| Kappa values | 0.0 (Euclidean), 0.01, 0.05, 0.1, 0.5, 1.0 |

---

## Repo structure

```
data/
  generate_data.py   — data generation script
  config.yaml        — all parameters (brain, retinotopy, pRF, splits, kappa values)
  data.md            — full documentation of the output files
model/
  model.md           — model architecture notes
```
