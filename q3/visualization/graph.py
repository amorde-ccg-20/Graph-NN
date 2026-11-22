import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_adjacency(adjacency, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(adjacency, interpolation="nearest", aspect="equal")
    ax.set_xlabel("Sensor index")
    ax.set_ylabel("Sensor index")
    fig.colorbar(image, ax=ax, label="Connection strength")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_neighbours(counts, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(counts, bins=np.arange(counts.max() + 2) - 0.5)
    ax.set_xlabel("Number of neighbours")
    ax.set_ylabel("Number of sensors")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
