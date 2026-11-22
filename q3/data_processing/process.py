import numpy as np


def resample_flow(data, aggregation):
    if aggregation == "mean":
        result = data.resample("15min").mean()
    elif aggregation == "sum":
        result = data.resample("15min").sum(min_count=1)
    else:
        raise ValueError("Aggregation must be 'mean' or 'sum'")
    return result.where(data.resample("15min").count() == 3)


def save_flow(path, data):
    np.savez_compressed(
        path,
        timestamps=data.index.to_numpy(dtype="datetime64[ns]"),
        sensor_ids=data.columns.to_numpy(dtype=str),
        flow=data.to_numpy(dtype=np.float32),
    )
