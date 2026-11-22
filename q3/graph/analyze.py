import argparse
from pathlib import Path

import numpy as np

from q3.visualization.graph import plot_adjacency, plot_neighbours


parser = argparse.ArgumentParser()
parser.add_argument("--sensor", type=int, required=True)
parser.add_argument("--center", choices=("include", "exclude"), required=True)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
processed = root / "dataset" / "processed"
with np.load(processed / "sd_flow_2020_handled.npz") as data:
    ids = data["sensor_ids"].astype(np.int64)
adjacency = np.load(processed / "sd_adj.npy")
if adjacency.shape != (len(ids), len(ids)):
    raise ValueError("Adjacency shape does not match the sensor order")

figures = root.parent / "figures"
figures.mkdir(parents=True, exist_ok=True)
plot_adjacency(adjacency, figures / "q3_fixed_adjacency.pdf")

links = adjacency != 0
if args.center == "exclude":
    np.fill_diagonal(links, False)
plot_neighbours(links.sum(axis=1), figures / "q3_neighbour_distribution.pdf")

node = int(np.flatnonzero(ids == args.sensor)[0])
one_hop = np.flatnonzero(links[node])
two_hop = np.flatnonzero(links[node] | links[one_hop].any(axis=0))
if args.center == "exclude":
    two_hop = two_hop[two_hop != node]

print("nodes:", len(ids))
print("nonzero entries:", np.count_nonzero(adjacency))
print("self-loops:", np.count_nonzero(np.diag(adjacency)))
print("symmetric:", np.allclose(adjacency, adjacency.T))
print("selected sensor:", args.sensor)
print("one-hop IDs:", ids[one_hop].tolist())
print("at-most-two-hop IDs:", ids[two_hop].tolist())
