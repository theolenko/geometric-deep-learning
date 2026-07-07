import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from model.model import ConvNet
from training.training import fit
from training.exp_manager import ExpManager
import yaml
from data.load_data import make_split, load_data
from data.generate_data_mult_maps import _tag


def build_distance_keys(gen_cfg):
    # keys = ["edge_attr_euc"] already done 
    keys = [f"edge_attr_hyp_{_tag(k)}" for k in gen_cfg["hyperbolic"]["curv_values"]]
    keys += [f"edge_attr_sph_{_tag(k)}" for k in gen_cfg["spherical"]["curv_values"]]
    return keys


def main():

    ### load running params
    with open("model/run_config.yaml", "r") as f:
        run_cfg = yaml.safe_load(f)

    with open("data/generate_config.yaml", "r") as f:
        gen_cfg = yaml.safe_load(f)

    ### load device
    requested_device = run_cfg["running_params"]["device"]
    if isinstance(requested_device, str) and requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested in config but no CUDA device is available. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    ### split into train/val/test once — distance-independent
    make_split(ratios=tuple(run_cfg["data"]["split_ratios"]), seed=run_cfg["data"]["seed"])

    base_exp_dir = run_cfg["running_params"]["exp_dir"]

    for distance_key in build_distance_keys(gen_cfg):
        print(f"\n=== Training with distance_key={distance_key} ===")

        model = ConvNet(dim=run_cfg["dim"],
                        n_responses=run_cfg["n_resp"],
                        kernel_size=run_cfg["kernel_s"])

        exp_m = ExpManager(exp_dir=os.path.join(base_exp_dir, distance_key),
                           run_cfg=run_cfg,
                           patience=run_cfg["running_params"]["patience"])

        ldr_train, ldr_val, ldr_test = load_data(distance_key=distance_key)

        model, train_losses_l1, train_losses_mae, val_losses = fit(ldr_train=ldr_train,
                                                                   ldr_test=ldr_val,
                                                                   model=model,
                                                                   epochs=run_cfg["running_params"]["epochs"],
                                                                   device=device,
                                                                   exp_manager=exp_m)
        exp_m.save_losses()


if __name__ == "__main__":
    main()