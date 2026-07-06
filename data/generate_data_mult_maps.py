"""
Per subject:
  1. load HCP midthickness GIFTI surface (.surf.gii)
  2. extract the visual-cortex patch (occipital geodesic disc)
  3. assign a synthetic retinotopic map (eccentricity + polar angle),
     with cortical magnification baked in
  4. simulate pRF responses to a fixed stimulus set  -> GNN input features
  5. package into a torch_geometric Data object

Distance functions (Euclidean / hyperbolic / spherical) are added in a
SEPARATE step; here edge_attr just holds Euclidean edge lengths.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nib       ### fmri library 
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch
from torch_geometric.data import Data
import yaml 
import os 


### load mesh and construct graph 
def load_surface(path):
    """Load an HCP GIFTI surface -> (coords [N,3] float, faces [M,3] int)."""
    gii = nib.load(str(path))
    coords, faces = gii.agg_data(("NIFTI_INTENT_POINTSET", "NIFTI_INTENT_TRIANGLE"))
    return np.asarray(coords, dtype=np.float64), np.asarray(faces, dtype=np.int64)

### get vertices, edges between them and their Euclidean distance 
def build_adjacency(coords, faces):
    """Undirected edges + sparse adjacency (weight = edge length in mm)."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    u, v = e[:, 0], e[:, 1]
    w = np.linalg.norm(coords[u] - coords[v], axis=1)   # Euclidean distance 
    N = coords.shape[0]
    adj = sp.csr_matrix((np.concatenate([w, w]),
                         (np.concatenate([u, v]), np.concatenate([v, u]))),
                        shape=(N, N))
    return u, v, w, adj

### get the occipital pole (represent middle of foveal)
def occipital_pole(coords, ap_axis):
    """Most posterior vertex — the foveal representation anchor."""
    return int(np.argmin(coords[:, ap_axis]))

### filters mesh for visual cortex (estimated by the distance to the foveal node

def edges_to_adj(u, v, w, N):
    return sp.csr_matrix((np.concatenate([w, w]),
                          (np.concatenate([u, v]), np.concatenate([v, u]))),
                         shape=(N, N))


def restrict_to_patch(coords, u, v, w, keep_idx, pole):
    """Keep patch vertices and EVERY edge between two of them (preserves connectivity)."""
    N = coords.shape[0]
    keep_mask  = np.zeros(N, bool);        keep_mask[keep_idx] = True
    old_to_new = -np.ones(N, np.int64);    old_to_new[keep_idx] = np.arange(keep_idx.size)

    edge_in = keep_mask[u] & keep_mask[v]              # both endpoints inside
    return (coords[keep_idx],
            old_to_new[u[edge_in]], old_to_new[v[edge_in]], w[edge_in],
            int(old_to_new[pole]))


### build the cortex subgraph --> relabels the nodes and therefore the faces/edges 
### need node idx from 0 to N-1, but filtering returns list with jumps  
def extract_subgraph(coords, faces, keep_idx):
    """Restrict the mesh to keep_idx and remap face indices to 0..K-1."""
    N = coords.shape[0]
    keep_mask = np.zeros(N, dtype=bool); keep_mask[keep_idx] = True
    old_to_new = -np.ones(N, dtype=np.int64); old_to_new[keep_idx] = np.arange(keep_idx.size)

    face_inside = keep_mask[faces].all(axis=1)
    faces_sub = old_to_new[faces[face_inside]]
    return coords[keep_idx], faces_sub

### simulates retinomap ("groundtruth" for training)
def retinotopy_map(coords, geo_from_pole, pole, cfg):
    """Eccentricity (cortical magnification), polar angle, magnification factor."""
    # eccentricity = exp(cortical distance) — the inverse of the log magnification law
    # cfg.scale: how fast eccentricity grows with cortical distance
    # cfg.a_param: 	foveal offset
    # cfg.ecc_min: lower clip (keeps values above 0)
    # cfg.ecc_max: the maximum eccentricity your stimuli (known do be  12°)
    
    ecc = np.clip(np.exp(geo_from_pole / cfg["retinotopy"]["scale"]) - cfg["retinotopy"]["a_param"], cfg["retinotopy"]["ecc_min"], cfg["retinotopy"]["ecc_max"])
    magnification = 1.0 / (cfg["retinotopy"]["k_param"] * ecc + cfg["retinotopy"]["a_param"])

    # polar angle = azimuth around the pole in the plane orthogonal to the AP axis
    ax = [a for a in (0, 1, 2) if a != cfg["brain"]["ap_axis"]]
    p = coords[pole]
    polar = np.arctan2(coords[:, ax[1]] - p[ax[1]], coords[:, ax[0]] - p[ax[0]])
    return ecc, polar, magnification

### syntheses response values (prfs) by sampling from a Gaussian 
# incorporating cortical magnification 

def simulate_prf(ecc, polar, cfg):
    """Gaussian pRF responses to `n_stimuli` random stimuli -> [N, n_stimuli]."""
    rng = np.random.default_rng(cfg["prf"]["seed"])
    vf_x, vf_y = ecc * np.cos(polar), ecc * np.sin(polar)

    
    ### sample n random stimuli positions (eccentricity and angle)
    stim_ecc = rng.uniform(0.0, cfg["prf"]["stim_ecc_max"], cfg["prf"]["n_stimuli"])
    stim_ang = rng.uniform(-np.pi, np.pi, cfg["prf"]["n_stimuli"])
    sx, sy   = stim_ecc * np.cos(stim_ang), stim_ecc * np.sin(stim_ang)
    
    ## calculates tolerance of vertices to position of stimuli being ideal for them 
    ## stimuli in pheriphery doesn't need as close to the ideal position to create response 
    ## of vertices processing them 
    
    sigma = cfg["prf"]["sigma_base"] + cfg["prf"]["sigma_slope"] * ecc          # pRF grows with eccentricity
    resp = np.zeros((ecc.size, cfg["prf"]["n_stimuli"]), dtype=np.float32)
    for s in range(cfg["prf"]["n_stimuli"]):
        d = np.sqrt((vf_x - sx[s]) ** 2 + (vf_y - sy[s]) ** 2)
        resp[:, s] = np.exp(-0.5 * (d / sigma) ** 2)
    resp += cfg["prf"]["noise_level"] * rng.standard_normal(resp.shape).astype(np.float32)
    return resp

def hyperbolic_weights(distances, kappa):
    # Maps Euclidean distances to hyperbolic distances with curvature κ.
    # Formula: d_hyp = (2/√κ) · arcsinh(√κ/2 · d_euc)
    # At κ=0 this reduces to d_euc (Euclidean baseline).
    # At κ>0, large distances are compressed logarithmically — nodes that are
    # far apart in brain space get a smaller effective distance in the GNN,
    # which helps for the densely-packed fovea region where Euclidean distances
    # overestimate how "different" nearby nodes actually are in the visual field.
    if kappa == 0.0:
        return distances.copy()
    sq = np.sqrt(kappa)
    return (2.0 / sq) * np.arcsinh((sq / 2.0) * distances)

def spherical_weights(distances, kappa):
    """Spherical analog: d_sph = (2/√κ) · arcsin(√κ/2 · d_euc).
       kappa = curvature magnitude (kappa=0 -> Euclidean).
       DOMAIN: arcsin needs (√κ/2)·d_euc ≤ 1, i.e. kappa ≤ (2/max_dist)²."""
    if kappa == 0.0:
        return distances.copy()
    sq = np.sqrt(kappa)
    x  = (sq / 2.0) * distances
    if np.any(x > 1.0):
        max_k = (2.0 / distances.max())**2
        raise ValueError(f"arcsin domain exceeded (max arg {x.max():.3f} > 1). "
                         f"Use kappa < {max_k:.4g} for these distances.")
    return (2.0 / sq) * np.arcsin(x)

### necessary for naming the curvature distances 
def _tag(kappa):
    # attribute names can't contain '.' or '-'  ->  0.1 -> "0p1", -0.1 -> "m0p1"
    return str(kappa).replace(".", "p").replace("-", "m")


def build_data(coords, u, v, w, ecc, polar, mag, responses, geo, pole, cfg):
    ei = np.vstack([np.concatenate([u, v]), np.concatenate([v, u])])   # both directions
    
    def edge_tensor(vals):                       # [E] undirected -> [2E, 1] both directions
        return torch.tensor(np.concatenate([vals, vals])[:, None], dtype=torch.float32)
    
    data = Data(
        x          = torch.tensor(responses, dtype=torch.float32),  # data for each vertex
        edge_index = torch.tensor(ei, dtype=torch.long),            # all edges 
        edge_attr_euc  = edge_tensor(w),         # pseudo coordinates for Spline
        pos        = torch.tensor(coords, dtype=torch.float32),
        groundtruth         = torch.tensor(np.column_stack([ecc, polar]).astype(np.float32)),
        geo_from_pole = torch.tensor(geo.astype(np.float32)),
        magnification = torch.tensor(mag.astype(np.float32)),
    )
    data.pole_idx = pole
    
    for k in cfg["hyperbolic"]["curv_values"]:
        data[f"edge_attr_hyp_{_tag(k)}"] = edge_tensor(hyperbolic_weights(w, k))
    for k in cfg["spherical"]["curv_values"]:
        print(f"kappa: {k}")
        data[f"edge_attr_sph_{_tag(k)}"] = edge_tensor(spherical_weights(w, k))    
        
    return data

### builds mesh for one subject and simulates map and stimuli on it 
def process_subject(surf_path, cfg):
    coords, faces = load_surface(surf_path)

    u, v, w, adj  = build_adjacency(coords, faces)       # subgraph geometry, once
    pole          = occipital_pole(coords, cfg["brain"]["ap_axis"])
    geo_full      = dijkstra(adj, directed=False, indices=pole)
    keep          = np.where(geo_full <= cfg["brain"]["patch_radius_mm"])[0]
    
    coords, u, v, w, pole = restrict_to_patch(coords, u, v, w, keep, pole)

    adj_sub = edges_to_adj(u, v, w, coords.shape[0])
    geo = dijkstra(adj_sub, directed=False, indices=pole)

    if np.isinf(geo).any():
        raise ValueError("patch is disconnected — raise patch_radius_mm or check ap_axis")

    ecc, polar, mag = retinotopy_map(coords, geo, pole, cfg)
    responses       = simulate_prf(ecc, polar, cfg)
    return build_data(coords, u, v, w, ecc, polar, mag, responses, geo, pole, cfg)

def main():
    
    here = Path(__file__).parent
    
    surf_dir = Path(os.path.join(here, "surfaces"))
    out_dir  = os.path.join(here, "datasets"); Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    with open(here / "generate_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    hemi = "L"   # one hemisphere for now
    surf_paths = sorted(surf_dir.rglob(f"*.{hemi}.midthickness*.surf.gii"))
    print(f"found {len(surf_paths)} meshes in {surf_dir}")

    examples = []
    for p in surf_paths:
        data = process_subject(p, cfg)
        data.subject = p.name.split(".")[0]
        examples.append(data)
        print(f"  {data.subject}: nodes={data.num_nodes} edges={data.num_edges}")

    torch.save(examples,Path(os.path.join(out_dir,"retinotopy_dataset.pt")))
    print(f"saved {len(examples)} subjects")
    #print(f"saved {len(examples)} subjects -> {out_dir/'retinotopy_dataset.pt'}")
        
    
        
if __name__ == "__main__":
    main()
        