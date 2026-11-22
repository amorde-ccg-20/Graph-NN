import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from q3.data_processing.windows import get_batch, load_splits
from q3.models.forecast import ForecastModel


def evaluate(model, split, adjacency, mean, std, batch_size):
    model.eval()
    absolute = squared = count = 0.0
    with torch.no_grad():
        for starts in split["starts"].split(batch_size):
            x, y, observed = get_batch(split, starts)
            prediction = model(x, adjacency)
            error = (prediction - y) * std
            absolute += (error.abs() * observed).sum().item()
            squared += (error.square() * observed).sum().item()
            count += observed.sum().item()
    return {"mae": absolute / count, "rmse": (squared / count) ** 0.5}


def train(name, use_graph, splits, adjacency, mean, std, output, log,
          model_factory=None, checkpoint_prefix="q4"):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = (model_factory() if model_factory else ForecastModel(hidden=16, use_graph=use_graph)).to(adjacency.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    checkpoint = output / f"{checkpoint_prefix}_{name}.pt"
    best_mae = float("inf")
    stale_epochs = 0
    start_time = time.perf_counter()

    for epoch in range(1, 11):
        model.train()
        loss_sum = count = 0.0
        shuffled = splits["train"]["starts"][torch.randperm(len(splits["train"]["starts"]), device=adjacency.device)]
        for starts in tqdm(shuffled.split(64), desc=f"{name} epoch {epoch}"):
            x, y, observed = get_batch(splits["train"], starts)
            optimizer.zero_grad()
            prediction = model(x, adjacency)
            loss = ((prediction - y).abs() * observed).sum() / observed.sum().clamp_min(1)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * observed.sum().item()
            count += observed.sum().item()

        validation = evaluate(model, splits["val"], adjacency, mean, std, 64)
        elapsed = time.perf_counter() - start_time
        line = (f"{name} epoch {epoch}: train MAE {loss_sum / count * std:.3f}, "
                f"val MAE {validation['mae']:.3f}, val RMSE {validation['rmse']:.3f}, "
                f"elapsed {elapsed:.1f}s")
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

        if validation["mae"] < best_mae:
            best_mae = validation["mae"]
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= 3:
                break

    model.load_state_dict(torch.load(checkpoint, map_location=adjacency.device, weights_only=True))
    return {
        "validation": evaluate(model, splits["val"], adjacency, mean, std, 64),
        "test": evaluate(model, splits["test"], adjacency, mean, std, 64),
        "training_seconds": time.perf_counter() - start_time,
        "epochs": epoch,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a RunPod GPU PyTorch image")
    device = torch.device("cuda")
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    splits, mean, std = load_splits(device)
    adjacency = torch.as_tensor(np.load(root / "dataset" / "processed" / "sd_adj.npy"),
                                dtype=torch.float32, device=device)
    adjacency = adjacency / adjacency.sum(dim=1, keepdim=True)

    results = {}
    with (output / "q4_training.log").open("w", encoding="utf-8") as log:
        for name, use_graph in (("temporal", False), ("fixed_graph", True)):
            results[name] = train(name, use_graph, splits, adjacency, mean, std, output, log)
            print(name, results[name], flush=True)

    results["settings"] = {
        "hidden": 16,
        "batch_size": 64,
        "max_epochs": 10,
        "patience": 3,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "seed": 42,
        "mean": mean,
        "std": std,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    (output / "q4_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
