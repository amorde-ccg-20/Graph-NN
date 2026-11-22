from pathlib import Path

import numpy as np
import pandas as pd


data_dir = Path(__file__).resolve().parents[1] / "dataset" / "processed"

with np.load(data_dir / "sd_flow_2020_15min.npz") as data:
    training_flow = data["flow"]

global_median = float(np.nanmedian(training_flow))
sensor_medians = pd.DataFrame(training_flow).median().fillna(global_median)

for year in (2020, 2021):
    with np.load(data_dir / f"sd_flow_{year}_15min.npz") as data:
        flow = data["flow"]
        observed = ~np.isnan(flow)
        timestamps = data["timestamps"]
        sensor_ids = data["sensor_ids"]

    filled = pd.DataFrame(flow).ffill(limit=4).fillna(sensor_medians)
    output = data_dir / f"sd_flow_{year}_handled.npz"
    np.savez_compressed(
        output,
        timestamps=timestamps,
        sensor_ids=sensor_ids,
        flow=filled.to_numpy(dtype=np.float32),
        observed=observed,
    )
    print(f"{year}: saved {output.name}; {int((~observed).sum()):,} missing values marked")
