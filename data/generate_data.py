import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch
from torch_geometric.data import Data
import yaml


def load_config(path):
    path = Path(path)
    with open(path) as f:
        return yaml.safe_load(f) if path.suffix in (".yml", ".yaml") else json.load(f)


def build_graph(cfg):
    from nilearn import datasets, surface

    brain = cfg["brain"]

    # Load the fsaverage "average brain" (FreeSurfer, Fischl et al. 1999).
    # The brain surface exists in two versions: the outer surface (pial) and
    # the inner surface (white matter). We use their midpoint (midthickness)
    # because that's what Ribeiro et al. 2021 use in deepRetinotopy.
    fsaverage          = datasets.fetch_surf_fsaverage(brain.get("surface", "fsaverage"))
    coords_pial, faces = surface.load_surf_mesh(fsaverage.pial_left)
    coords_white, _    = surface.load_surf_mesh(fsaverage.white_left)
    coords = (np.array(coords_pial) + np.array(coords_white)) / 2.0
    faces  = np.array(faces)
    print(f"Vertices: {coords.shape[0]:,}  Faces: {faces.shape[0]:,}")

    # Keep only the visual cortex: the back of the brain (occipital lobe).
    # In MNI coordinates, "back" = negative y-axis. Everything below -70 mm
    # on the y-axis is roughly where the visual cortex sits. This threshold
    # comes from anatomy — fMRI studies show visual responses only in this region.
    visual_mask = coords[:, 1] < brain.get("y_threshold", -70.0)
    visual_idx  = np.where(visual_mask)[0]
    coords_vis  = coords[visual_idx]
    N           = len(coords_vis)
    print(f"Visual-cortex vertices: {N:,}  ({100 * N / len(coords):.1f}% of brain)")

    # The occipital pole (the most posterior vertex, i.e. the smallest y-value)
    # represents the fovea — the center of vision. This is established by fMRI
    # retinotopy: when subjects look at the center of the screen, this exact
    # spot on the brain activates. It's our anchor point for everything below.
    occipital_pole_idx = int(np.argmin(coords_vis[:, 1]))
    occipital_pole     = coords_vis[occipital_pole_idx]

    # Polar angle = the compass direction of a node around the occipital pole.
    # We measure it in the x-z plane (left-right vs up-down), because those two
    # axes correspond to the horizontal and vertical axes of the visual field.
    # arctan2 converts (dx, dz) into an angle in [-π, π] radians.
    dx          = coords_vis[:, 0] - occipital_pole[0]
    dz          = coords_vis[:, 2] - occipital_pole[2]
    polar_angle = np.arctan2(dz, dx)  # [-π, π]

    # Build the graph: each triangle in the 3D mesh contributes up to 3 edges.
    # We only keep edges where both endpoints are inside the visual cortex mask.
    # The original face indices run from 0 to 163,841 (the full brain), so we
    # need a lookup table (old_to_new) to remap them to our compact 0..N-1 range.
    old_to_new = {old: new for new, old in enumerate(visual_idx)}
    edge_set   = set()
    for tri in faces:
        if tri[0] in old_to_new and tri[1] in old_to_new and tri[2] in old_to_new:
            a, b, c = old_to_new[tri[0]], old_to_new[tri[1]], old_to_new[tri[2]]
            for u, v in [(a, b), (b, c), (a, c)]:
                edge_set.add((min(u, v), max(u, v)))

    edges             = np.array(list(edge_set)).T
    u_idx, v_idx      = edges[0], edges[1]
    # Edge weight = straight-line distance between two neighboring nodes in mm.
    # This is the Euclidean (flat) baseline — what we later compare against
    # the hyperbolic version.
    euclidean_weights = np.linalg.norm(coords_vis[u_idx] - coords_vis[v_idx], axis=1)
    print(f"Unique edges: {edges.shape[1]:,}  "
          f"Edge length min/mean/max (mm): {euclidean_weights.min():.2f} {euclidean_weights.mean():.2f} {euclidean_weights.max():.2f}")

    # Geodesic distance = shortest path from each node to the fovea pole,
    # measured along the brain surface (not through tissue).
    # We need this because the cortex is a curved sheet: a straight line between
    # two nodes would cut through the brain, which is not how signals travel.
    # Dijkstra finds the shortest path hop-by-hop through the mesh edges.
    row = np.concatenate([u_idx, v_idx])
    col = np.concatenate([v_idx, u_idx])
    dat = np.concatenate([euclidean_weights, euclidean_weights])
    adj = sp.csr_matrix((dat, (row, col)), shape=(N, N))
    geo_from_pole = dijkstra(adj, directed=False, indices=occipital_pole_idx)
    n_unreachable = np.isinf(geo_from_pole).sum()
    if n_unreachable > 0:
        raise ValueError(f"{n_unreachable} nodes unreachable from occipital pole — graph is disconnected. Try adjusting y_threshold.")
    print(f"geo_from_pole shape: {geo_from_pole.shape}  distance min/max (mm): {geo_from_pole.min():.1f} {geo_from_pole.max():.1f}")

    # Convert geodesic distance → eccentricity (degrees of visual angle).
    # fMRI measurements show this relationship is logarithmic (Schwartz 1980):
    # equal visual-angle steps correspond to exponentially growing distances on
    # the cortex. The inverse of log is exp, so:
    #   eccentricity = exp(surface_distance / scale) - a_param
    # This bakes in cortical magnification: the fovea gets exponentially more
    # cortex per degree of visual field than the periphery.
    ret     = cfg["retinotopy"]
    a_param = ret.get("a_param", 1.6)   # deg   — foveal offset; Schwartz (1980) p.659, citing Drasdo (1977)
    scale   = ret.get("scale",   15.1)  # mm/deg — foveal magnification; Cowey & Rolls (1974) Table 1 p.452: 15.1 mm/deg
    k_param = ret.get("k_param", 0.065) # deg⁻¹ — magnification fall-off (Horton & Hoyt 1991)

    eccentricity  = np.exp(geo_from_pole / scale) - a_param
    eccentricity  = np.clip(eccentricity, ret.get("ecc_min", 0.1), ret.get("ecc_max", 12.0))
    # Magnification factor: how many mm² of cortex are devoted to 1 deg² of visual field.
    # High near fovea (many neurons per degree), low at the periphery.
    magnification = 1.0 / (k_param * eccentricity + a_param)  # mm²/deg²
    print(f"eccentricity min/max: {eccentricity.min():.2f} {eccentricity.max():.2f} deg")

    # Convert polar coordinates (eccentricity, polar_angle) to Cartesian (vf_x, vf_y)
    # in visual field space (degrees). This is just: x = r·cos(θ), y = r·sin(θ).
    # We need Cartesian coordinates to compute distances between nodes and stimuli
    # during the pRF simulation below.
    vf_x = eccentricity * np.cos(polar_angle)
    vf_y = eccentricity * np.sin(polar_angle)

    return dict(
        N=N, coords_vis=coords_vis, occipital_pole_idx=occipital_pole_idx,
        polar_angle=polar_angle, eccentricity=eccentricity, magnification=magnification,
        vf_x=vf_x, vf_y=vf_y, geo_from_pole=geo_from_pole,
        u_idx=u_idx, v_idx=v_idx, euclidean_weights=euclidean_weights,
    )


def simulate_prf(graph, cfg):
    # pRF = population Receptive Field.
    # Each brain node is tuned to a specific region of the visual field.
    # When a stimulus appears there, the node fires strongly. When the stimulus
    # is far away, it barely fires. This tuning follows a Gaussian (bell curve):
    # response = exp(-0.5 * (distance_in_visual_field / sigma)²)
    #
    # sigma (the width of the bell curve) grows with eccentricity because
    # peripheral neurons "watch" a larger patch of the visual field — they're
    # less precise than foveal neurons (Harvey & Dumoulin 2011).
    #
    # We simulate S=100 stimuli at random positions. The output is a (N × S)
    # matrix: how strongly each of the N nodes responds to each of the S stimuli.
    # This matrix is what the GNN sees as input — it's our synthetic fMRI signal.
    prf = cfg["prf"]
    S   = prf.get("n_stimuli", 100)
    rng = np.random.default_rng(prf.get("seed", 42))

    stim_ecc = rng.uniform(0.0, prf.get("stim_ecc_max", 10.0), S)
    stim_ang = rng.uniform(-np.pi, np.pi, S)
    stim_x   = stim_ecc * np.cos(stim_ang)
    stim_y   = stim_ecc * np.sin(stim_ang)

    sigma     = prf.get("sigma_base", 0.2) + prf.get("sigma_slope", 0.4) * graph["eccentricity"]
    responses = np.zeros((graph["N"], S), dtype=np.float32)
    for s in range(S):
        dist_vf         = np.sqrt((graph["vf_x"] - stim_x[s])**2 + (graph["vf_y"] - stim_y[s])**2)
        responses[:, s] = np.exp(-0.5 * (dist_vf / sigma)**2)

    responses += prf.get("noise_level", 0.01) * rng.standard_normal(responses.shape).astype(np.float32)
    print(f"responses shape: {responses.shape}  value range: {responses.min():.2f} {responses.max():.2f}")
    return responses


def make_splits(N, cfg):
    # Split the N nodes into train / val / test.
    # We shuffle the node indices randomly, then cut at the configured ratios.
    # All three splits share the same graph — only the masks differ.
    split  = cfg.get("split", {})
    r_train = split.get("train", 0.7)
    r_val   = split.get("val",   0.15)
    rng     = np.random.default_rng(split.get("seed", 0))

    perm    = rng.permutation(N)
    n_train = int(N * r_train)
    n_val   = int(N * r_val)

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    print(f"Split — train: {train_mask.sum()} | val: {val_mask.sum()} | test: {test_mask.sum()} nodes")
    return train_mask, val_mask, test_mask


def hyperbolic_weights(distances, kappa):
    # κ-parametrised distance transformation: d_hyp = (2/√κ) · arcsinh(√κ/2 · d_euc)
    # At κ=0 this reduces to d_euc (Euclidean baseline).
    # At κ>0, large distances are compressed more than small ones.
    #
    # Note: this is NOT a proper point-to-point distance in hyperbolic space.
    # The arcsinh form appears in Ganea et al. (2018) but in the context of
    # point-to-hyperplane distance in the Poincaré ball — a different geometric
    # quantity. This transformation is motivated by hyperbolic geometry but does
    # not constitute a mathematically rigorous hyperbolic embedding.
    if kappa == 0.0:
        return distances.copy()
    sq = np.sqrt(kappa)
    return (2.0 / sq) * np.arcsinh((sq / 2.0) * distances)


def _vf_pseudo(graph):
    # Relative visual-field position vectors per directed edge: (Δvf_x, Δvf_y).
    # Order matches edge_index: first block u→v, second block v→u.
    u, v  = graph["u_idx"], graph["v_idx"]
    dvf_x = np.concatenate([graph["vf_x"][v] - graph["vf_x"][u],
                             graph["vf_x"][u] - graph["vf_x"][v]])
    dvf_y = np.concatenate([graph["vf_y"][v] - graph["vf_y"][u],
                             graph["vf_y"][u] - graph["vf_y"][v]])
    return np.column_stack([dvf_x, dvf_y]).astype(np.float32)  # (2E, 2)


def save_groundtruth(graph, responses, train_mask, val_mask, test_mask, path):
    u, v  = graph["u_idx"], graph["v_idx"]
    euc_w = graph["euclidean_weights"]

    # PyTorch Geometric expects edges in both directions (u→v and v→u).
    # We duplicate each undirected edge, and duplicate the weights accordingly.
    ei = np.vstack([np.concatenate([u, v]), np.concatenate([v, u])])
    ea = np.concatenate([euc_w, euc_w])
    ep = _vf_pseudo(graph)

    # data.x  = pRF responses, shape (N, 100) — the GNN input.
    #           Each row is one node. Each column is one stimulus.
    #           This is the synthetic "fMRI signal" the model has to learn from.
    # data.y  = ground-truth targets, shape (N, 2) — [eccentricity, polar_angle].
    #           Eccentricity in degrees (0.1–12), polar angle in radians (−π … π).
    #           These are NOT in data.x — putting them there would be data leakage
    #           (the model would see the answers it's supposed to predict).
    data = Data(
        x               = torch.tensor(responses, dtype=torch.float32),
        edge_index      = torch.tensor(ei, dtype=torch.long),
        edge_attr       = torch.tensor(ea, dtype=torch.float32),
        edge_pseudo     = torch.tensor(ep),
        pos             = torch.tensor(graph["coords_vis"], dtype=torch.float32),
        vf_pos          = torch.tensor(np.column_stack([graph["vf_x"],
                                                        graph["vf_y"]]).astype(np.float32)),
        y               = torch.tensor(np.column_stack([graph["eccentricity"],
                                                        graph["polar_angle"]]).astype(np.float32)),
        eccentricity_gt = torch.tensor(graph["eccentricity"].astype(np.float32)),
        polar_angle_gt  = torch.tensor(graph["polar_angle"].astype(np.float32)),
        magnification   = torch.tensor(graph["magnification"].astype(np.float32)),
        train_mask      = train_mask,
        val_mask        = val_mask,
        test_mask       = test_mask,
    )

    torch.save({'data': data, 'meta': {
        'layer': 1,
        'content': 'ground-truth retinotopic map + graph + pRF training signal',
        'n_nodes': graph["N"],
        'n_directed_edges': int(ei.shape[1]),
        'occipital_pole_idx': graph["occipital_pole_idx"],
        'note': 'Same graph for every kappa. Pair with edges_kappa_*.pt files.',
    }}, path)
    print(f"saved: {path}")
    print(f"  x: {tuple(data.x.shape)}  edge_index: {tuple(data.edge_index.shape)}"
          f"  edge_pseudo: {tuple(data.edge_pseudo.shape)}  y: {tuple(data.y.shape)}")


def save_hyperbolic(graph, kappa_values, out_dir):
    euc_w  = graph["euclidean_weights"]
    geo    = graph["geo_from_pole"]
    ep_euc = _vf_pseudo(graph)  # (2E, 2) Euclidean VF pseudo-coordinates

    for kappa in kappa_values:
        hyp_edges = hyperbolic_weights(euc_w, kappa)
        hyp_pole  = hyperbolic_weights(geo,   kappa)

        # Transform pseudo-coordinate magnitude hyperbolically, preserve direction.
        mag    = np.linalg.norm(ep_euc, axis=1)
        hyp_mag = hyperbolic_weights(mag, kappa)
        sf     = np.where(mag > 1e-8, hyp_mag / mag, 1.0)
        hyp_ep = (ep_euc * sf[:, np.newaxis]).astype(np.float32)

        out = {
            "edge_attr":    torch.tensor(np.concatenate([hyp_edges, hyp_edges]), dtype=torch.float32),
            "edge_pseudo":  torch.tensor(hyp_ep),
            "dist_to_pole": torch.tensor(hyp_pole.astype(np.float32)),
            "kappa": kappa,
        }
        path = out_dir / f"edges_kappa_{kappa}.pt"
        torch.save(out, path)
        print(f"saved: {path}  edge_attr {tuple(out['edge_attr'].shape)}"
              f"  edge_pseudo {tuple(out['edge_pseudo'].shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    out     = cfg.get("output", {})
    out_dir = Path(out.get("dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    graph                       = build_graph(cfg)
    responses                   = simulate_prf(graph, cfg)
    train_mask, val_mask, test_mask = make_splits(graph["N"], cfg)

    save_groundtruth(graph, responses, train_mask, val_mask, test_mask,
                     out_dir / out.get("groundtruth_filename", "retinotopy_groundtruth.pt"))
    save_hyperbolic(graph,
                    cfg.get("hyperbolic", {}).get("kappa_values", [0.0, 0.1]),
                    out_dir)


if __name__ == "__main__":
    main()
