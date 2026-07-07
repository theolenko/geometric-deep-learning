import torch
from torch_geometric.loader import DataLoader
import json 
from pathlib import Path

def make_split(dataset_path="data/datasets/retinotopy_dataset.pt",
               out_path="data/datasets/split.json",
               ratios=(0.7, 0.15, 0.15), seed=0, force=False):
    """Teilt einmalig nach Subject und speichert die IDs.
    Existiert split.json bereits, wird sie geladen statt neu erzeugt (außer force=True)."""
    out_path = Path(out_path)

    # --- schon vorhanden? -> laden und zurückgeben ---
    if out_path.exists() and not force:
        with open(out_path) as f:
            split = json.load(f)
        print(f"Split existiert bereits, geladen: {out_path} "
              f"{ {k: len(v) for k, v in split.items()} }")
        return split

    # --- sonst neu erzeugen ---
    examples = torch.load(dataset_path, weights_only=False)
    ids = [d.subject for d in examples]

    g    = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ids), generator=g).tolist()
    ids  = [ids[i] for i in perm]
    n_tr, n_va = int(ratios[0] * len(ids)), int(ratios[1] * len(ids))

    split = {"train": ids[:n_tr],
             "val":   ids[n_tr:n_tr + n_va],
             "test":  ids[n_tr + n_va:]}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Split neu erstellt: {out_path} { {k: len(v) for k, v in split.items()} }")
    return split

def _norm_stats(dataset, key):
    vals = torch.cat([d[key] for d in dataset])
    return vals.min().item(), vals.max().item()


def load_data(distance_key="edge_attr_euc",
                  dataset_path="data/datasets/retinotopy_dataset.pt",
                  split_path="data/datasets/split.json",
                  batch_size=1):
    examples = torch.load(dataset_path, weights_only=False)
    by_id    = {d.subject: d for d in examples}
    with open(split_path) as f:
        split = json.load(f)

    train = [by_id[s] for s in split["train"]]
    val   = [by_id[s] for s in split["val"]]
    test  = [by_id[s] for s in split["test"]]

    lo, hi = _norm_stats(train, distance_key)      # Normalisierung aus TRAIN
    for grp in (train, val, test):
        for d in grp:
            d.edge_attr = ((d[distance_key] - lo) / (hi - lo + 1e-12)).clamp(0.0, 1.0)

    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(val,   batch_size=batch_size, shuffle=False),
            DataLoader(test,  batch_size=batch_size, shuffle=False))
