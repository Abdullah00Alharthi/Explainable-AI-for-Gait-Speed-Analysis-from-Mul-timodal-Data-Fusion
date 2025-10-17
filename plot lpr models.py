import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# Load Data
# =============================
datasets = {
    "Markers": [np.load("scores1.npy"), np.load("scores11.npy"), np.load("scores111.npy")],
    "EMG": [np.load("scores2.npy"), np.load("scores22.npy"), np.load("scores222.npy")],
    "GRF": [np.load("scores3.npy"), np.load("scores33.npy"), np.load("scores333.npy")],
    "F&M": [np.load("scores4.npy"), np.load("scores44.npy"), np.load("scores444.npy")]
}

# Extract accuracy arrays
accuracy_data = {k: [v[0][:, 1], v[1][:, 1], v[2][:, 1]] for k, v in datasets.items()}

# =============================
# Line Plot (Distinct Visual)
# =============================
plt.figure(figsize=(13, 7))
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16  # larger text
plt.rcParams["font.weight"] = "bold"

colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628",
          "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
markers = ["o", "s", "^", "d", "p", "*", "h", "x", "D", "<", ">", "v"]

labels = [
    "Single CNN Markers", "Dual CNN Markers", "Quads CNN Markers",
    "Single CNN EMG", "Dual CNN EMG", "Quads CNN EMG",
    "Single CNN GRF", "Dual CNN GRF", "Quads CNN GRF",
    "Single CNN F&M", "Dual CNN F&M", "Quads CNN F&M"
]

all_accuracies = np.concatenate(list(accuracy_data.values()))
x = np.arange(all_accuracies.shape[1])

for i, (label, acc) in enumerate(zip(labels, all_accuracies)):
    plt.plot(
        x, acc,
        label=label,
        linewidth=2.5,
        color=colors[i % len(colors)],
        marker=markers[i % len(markers)],
        markersize=5,
        alpha=0.85
    )

plt.title("LRP Accuracy Perturbation Analysis — CNN Models", fontsize=20, pad=15)
plt.xlabel("Perturbation Steps", fontsize=18, labelpad=10)
plt.ylabel("Model Accuracy", fontsize=18, labelpad=10)
plt.grid(True, linestyle="--", color="gray", alpha=0.4)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# ✅ Compatible legend setup
ax = plt.gca()
legend = ax.legend(title="CNN Models", fontsize=12, loc="upper right", ncol=2)
if legend.get_title() is not None:
    legend.get_title().set_fontsize(13)  # manually set title font size

plt.tight_layout()
plt.show()

# =============================
# Heatmap (Contrasting Visual)
# =============================
sns.set(style="whitegrid")

heatmap_data = np.vstack(all_accuracies)
plt.figure(figsize=(14, 7), dpi=120)

ax = sns.heatmap(
    heatmap_data,
    cmap="viridis",
    cbar_kws={'label': 'Model Accuracy'},
    linewidths=0.3,
    linecolor="black",
)

plt.title("CNN Model Accuracy Heatmap (Perturbation Steps)", fontsize=18, pad=15)
plt.xlabel("Perturbation Steps", fontsize=16, labelpad=10)
plt.ylabel("CNN Models", fontsize=16, labelpad=10)

x_steps = np.arange(0, heatmap_data.shape[1], 10)
plt.xticks(x_steps + 0.5, [str(i) for i in x_steps], rotation=0, fontsize=12)
plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0, fontsize=12)

plt.tight_layout()
plt.show()
