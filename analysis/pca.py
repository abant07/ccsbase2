import sys
import os

# --- Add this block before your other imports ---
# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (ccsbase2)
parent_dir = os.path.dirname(current_dir)
# Add parent to sys.path so Python can find 'utils'
sys.path.append(parent_dir)
# -----------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import Utils  # adjust if needed


# toggle
train = True
csvfile = "train_data.csv" if train else "test_data.csv"
title = "Training PCA" if train else "Test PCA"

# load data
df = pd.read_csv(f"../pretrained/{csvfile}")

# adduct vocabulary (from this split only)
adducts = sorted(df["adduct"].unique().tolist())

# compute descriptors
u = Utils()
X = np.vstack([
    u.calculate_descriptors(
        row["smi"], row["mass"], row["z"], adducts, row["adduct"]
    )
    for _, row in df.iterrows()
])

# scale + PCA
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

pca = PCA(n_components=2)
Z = pca.fit_transform(X_s)

var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100

# color by subclass (no legend)
colors = df["subclass"].astype("category").cat.codes

# plot (4-quadrant style)
plt.figure(figsize=(7, 7))
plt.scatter(
    Z[:, 0],
    Z[:, 1],
    c=colors,
    cmap=plt.cm.tab20,
    s=14,
    alpha=0.5,
    linewidths=0,
)

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.axvline(0, color="black", linestyle="--", linewidth=1)

plt.xlabel(f"PC1 ({var1:.1f} %)")
plt.ylabel(f"PC2 ({var2:.1f} %)")
plt.title(title)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()


