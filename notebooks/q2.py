from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


sensors = np.arange(40)
initial = np.exp(-((sensors - 20) ** 2) / 8)

predicted = [initial]
for _ in range(20):
    x = predicted[-1]
    predicted.append(0.5 * (np.roll(x, 1) + np.roll(x, -1)))

actual = [np.roll(initial, -t) for t in range(21)]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True, layout="constrained")
for ax, values, title in zip(axes, (predicted, actual), ("Predicted Motion", "True motion")):
    image = ax.imshow(values, origin="lower", aspect="auto", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Sensor index")
axes[0].set_ylabel("Time step")
fig.colorbar(image, ax=axes, label="Slowdown")

output = Path(__file__).resolve().parents[1] / "figures" / "q2_rollout_heatmaps.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output)
plt.close(fig)
print(f"Saved {output}")
