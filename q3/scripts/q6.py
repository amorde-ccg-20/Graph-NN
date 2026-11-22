import json
from pathlib import Path

import numpy as np
import torch

from q3.data_processing.windows import get_batch, load_splits
from q3.graph.compare import compare_graphs
from q3.models.adaptive import AdaptiveForecastModel
from q3.models.forecast import ForecastModel
from q3.visualization.predictions import plot_predictions


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs"
    report_figures = root.parent / "report" / "figures"
    selected_ids = [1119780, 1121692, 1119960]
    splits, mean, std = load_splits(torch.device("cpu"))
    test = splits["test"]
    fixed = torch.as_tensor(np.load(root / "dataset" / "processed" / "sd_adj.npy"), dtype=torch.float32)
    fixed_normalized = fixed / fixed.sum(dim=1, keepdim=True)

    with np.load(root / "dataset" / "processed" / "sd_flow_2021_handled.npz") as data:
        all_ids = data["sensor_ids"].astype(int)
        dates = data["timestamps"]
    cutoff = np.searchsorted(dates, np.datetime64("2021-04-01"))
    test_dates = dates[cutoff:]
    day_start = np.searchsorted(test_dates, np.datetime64("2021-04-02"))
    starts = torch.arange(day_start - 11, day_start + 96 - 11)
    x, y, observed = get_batch(test, starts)
    columns = [int(np.flatnonzero(all_ids == sensor_id)[0]) for sensor_id in selected_ids]
    actual = y[:, 3, columns].numpy() * std + mean
    valid = observed[:, 3, columns].numpy()
    actual[~valid] = np.nan

    models = {
        "Temporal": (ForecastModel(use_graph=False), "q4_temporal.pt"),
        "Fixed": (ForecastModel(), "q4_fixed_graph.pt"),
        "Adaptive": (AdaptiveForecastModel(), "q5_adaptive.pt"),
        "Hybrid": (AdaptiveForecastModel(hybrid=True), "q5_hybrid.pt"),
    }
    predictions = {}
    with torch.no_grad():
        for name, (model, checkpoint) in models.items():
            model.load_state_dict(torch.load(output / checkpoint, map_location="cpu", weights_only=True))
            model.eval()
            values = model(x, fixed_normalized)[:, 3, columns].numpy() * std + mean
            values[~valid] = np.nan
            predictions[name] = values
    plot_predictions(test_dates[day_start:day_start + 96], actual, predictions, selected_ids,
                     report_figures / "q6_predictions.pdf")

    adaptive = models["Adaptive"][0].adaptive_adjacency().detach().numpy()
    analysis = compare_graphs(adaptive, fixed.numpy())
    (output / "q6_graph_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
