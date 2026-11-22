from pathlib import Path

import numpy as np
import torch


def load_splits(device):
    directory = Path(__file__).resolve().parents[1] / "dataset" / "processed"
    with np.load(directory / "sd_flow_2020_handled.npz") as data:
        train_flow = data["flow"]
        train_observed = data["observed"]
    with np.load(directory / "sd_flow_2021_handled.npz") as data:
        flow_2021 = data["flow"]
        observed_2021 = data["observed"]
        dates_2021 = data["timestamps"]

    cutoff = np.searchsorted(dates_2021, np.datetime64("2021-04-01"))
    observed_train_values = train_flow[train_observed]
    mean = float(observed_train_values.mean())
    std = float(observed_train_values.std())

    arrays = {
        "train": (train_flow, train_observed),
        "val": (flow_2021[:cutoff], observed_2021[:cutoff]),
        "test": (flow_2021[cutoff:], observed_2021[cutoff:]),
    }
    splits = {}
    for name, (flow, observed) in arrays.items():
        splits[name] = {
            "flow": torch.as_tensor((flow - mean) / std, device=device),
            "observed": torch.as_tensor(observed, device=device),
            "starts": torch.arange(len(flow) - 11, device=device),
        }
    return splits, mean, std


def get_batch(split, starts):
    flow = split["flow"]
    observed = split["observed"]
    x = flow[starts[:, None] + torch.arange(8, device=flow.device)]
    y_indices = starts[:, None] + torch.arange(8, 12, device=flow.device)
    return x, flow[y_indices], observed[y_indices]
