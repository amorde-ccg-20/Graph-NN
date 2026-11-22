import json
from pathlib import Path

import numpy as np
import torch

from q3.data_processing.windows import load_splits
from q3.models.adaptive import AdaptiveForecastModel
from q3.scripts.q4 import train
from q3.visualization.graph import plot_adjacency


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a RunPod GPU PyTorch image")
    device = torch.device("cuda")
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs"
    splits, mean, std = load_splits(device)
    fixed = torch.as_tensor(np.load(root / "dataset" / "processed" / "sd_adj.npy"),
                            dtype=torch.float32, device=device)
    fixed = fixed / fixed.sum(dim=1, keepdim=True)

    q4 = json.loads((output / "q4_results.json").read_text(encoding="utf-8"))
    results = {"fixed_graph": q4["fixed_graph"]}
    with (output / "q5_training.log").open("w", encoding="utf-8") as log:
        for name, hybrid in (("adaptive", False), ("hybrid", True)):
            results[name] = train(name, True, splits, fixed, mean, std, output, log,
                                  model_factory=lambda hybrid=hybrid: AdaptiveForecastModel(hybrid=hybrid),
                                  checkpoint_prefix="q5")
            print(name, results[name], flush=True)

    model = AdaptiveForecastModel().to(device)
    model.load_state_dict(torch.load(output / "q5_adaptive.pt", map_location=device, weights_only=True))
    learned = model.adaptive_adjacency().detach().cpu().numpy()
    plot_adjacency(learned, output / "q5_adaptive_adjacency.pdf")

    model = AdaptiveForecastModel(hybrid=True).to(device)
    model.load_state_dict(torch.load(output / "q5_hybrid.pt", map_location=device, weights_only=True))
    results["hybrid"]["fixed_weight"] = torch.sigmoid(model.mix_logit).item()
    results["settings"] = {
        "embedding_dim": 16,
        "sparsification": None,
        "hybrid_mix": "learned",
        "shared_training_settings": q4["settings"],
    }
    (output / "q5_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
