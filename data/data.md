# Task: Generate synthetic data for the model to train on.

## 1. How the Retinotopy was made
Started from the fsaverage average brain and keep only the visual cortex (the back of the brain, MNI y < −70 mm → 17,263 surface vertices). The most posterior vertex is taken as the occipital pole, which represents the fovea (center of vision).
For every vertex then derive its true retinotopic coordinates:

- Polar angle (direction in the visual field) = the angle of the vertex around the pole, in the x–z plane.
- Eccentricity (distance from the center of vision, in degrees) = the geodesic distance from the pole (shortest path along the surface, computed with Dijkstra) passed through a log-polar cortical-magnification model. This bakes the magnification warp into the "true" map, so the underlying metric is non-Euclidean by construction.

Finally 100 visual stimuli were simulated and computed each vertex's pRF response (Gaussian receptive field).

## 2. Data Description

**N = 17,263 nodes** (vertices), **E = 102,432 directed edges**.

| Field | Meaning | Shape | Type | Range |
|------|---------|-------|------|-------|
| `data.x` | Node features: 6 numbers describing each node (see table below) | `(17263, 6)` | `float32` | — |
| `data.edge_index` | Graph wiring: which nodes are connected (row 0 = source, row 1 = target) | `(2, 102432)` | `int64` | `0 ... 17262` |
| `data.pos` | 3D position: location of each node on the brain (mm) | `(17263, 3)` | `float32` | Brain coordinates |
| `data.y` | pRF responses: how each node reacts to 100 stimuli (training target) | `(17263, 100)` | `float32` | `~0 ... 1` |
| `data.eccentricity_gt` | Ground-truth eccentricity: correct distance from the center of vision (degrees) | `(17263,)` | `float32` | `0.5 ... 12` |
| `data.polar_angle_gt` | Ground-truth polar angle: correct direction in the visual field (radians) | `(17263,)` | `float32` | `0 ... 2π` |

The 6 Columns Inside `data.x`

| Column | Name | Meaning | Range |
|--------|------|---------|-------|
| `x[:, 0]` | `pos_x` | Left–right position (mm) | `-50 ... -2` |
| `x[:, 1]` | `pos_y` | Front–back position (mm) | `-104 ... -70` |
| `x[:, 2]` | `pos_z` | Up–down position (mm) | `-19 ... 56` |
| `x[:, 3]` | `eccentricity` | Distance from the center of vision (degrees) | `0.5 ... 12` |
| `x[:, 4]` | `polar_angle` | Direction in the visual field (radians) | `-π ... π` |
| `x[:, 5]` | `magnification` | Cortical magnification factor | `0.78 ... 1.88` |

## 3. Access

```bash
pip install torch torch-geometric
```

```python
import torch

ckpt = torch.load('retinotopy_groundtruth.pt', map_location='cpu')
data = ckpt['data']     # the graph + map + training signal

print(data)
# Data(x=[17263, 6], edge_index=[2, 102432], pos=[17263, 3],
#      y=[17263, 100], eccentricity_gt=[17263], polar_angle_gt=[17263])
```