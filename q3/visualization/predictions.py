import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_predictions(times, actual, predictions, sensor_ids, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for column, (ax, sensor_id) in enumerate(zip(axes, sensor_ids)):
        ax.plot(times, actual[:, column], color="black", linewidth=1.6, label="Actual")
        for name, values in predictions.items():
            ax.plot(times, values[:, column], linewidth=1, label=name)
        ax.set_ylabel(f"{sensor_id}\nvehicles / 15 min")
    axes[0].legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("2 April 2021")
    fig.savefig(path)
    plt.close(fig)
