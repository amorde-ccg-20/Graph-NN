import h5py
import numpy as np
import pandas as pd


def load_flow(path, metadata_path, year):
    metadata = pd.read_csv(metadata_path)
    sensor_ids = metadata.loc[metadata["District"] == 11, "ID"].astype(str).tolist()
    if len(sensor_ids) != 716:
        raise ValueError(f"Expected 716 San Diego sensors; found {len(sensor_ids)}")

    with h5py.File(path) as source:
        table = source["t"]
        all_ids = [value.decode() for value in table["axis0"][:]]
        positions = [all_ids.index(sensor_id) for sensor_id in sensor_ids]
        timestamps = pd.to_datetime(table["axis1"][:])
        values = table["block0_values"]
        selected = np.empty((len(timestamps), len(sensor_ids)), dtype=np.float32)
        for start in range(0, len(timestamps), 1000):
            stop = min(start + 1000, len(timestamps))
            selected[start:stop] = values[start:stop, :][:, positions]

    data = pd.DataFrame(selected, index=timestamps, columns=sensor_ids)
    return data.loc[str(year)]
