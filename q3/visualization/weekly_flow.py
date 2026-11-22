import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_week(data, path):
    week = data.loc["2020-01-06":"2020-01-12"]
    sensors = list(week.columns[:3])
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, sensor in zip(axes, sensors):
        ax.plot(week.index, week[sensor], linewidth=0.8)
        ax.set_ylabel(f"Sensor {sensor}")
    axes[-1].set_xlabel("Timestamp")
    fig.suptitle("San Diego traffic flow: 6-12 January 2020")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
