import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = True
csvfile = "train_data.csv" if train else "test_data.csv"
title = "Training CCS Distribution" if train else "Test CCS Distribution"

# load data
df = pd.read_csv(f"../pretrained/{csvfile}")

# CCS histogram bins: 100 to 700 in steps of 10
bins = np.arange(100, 701, 10)

# weights so each entry contributes equally to 100%
weights = np.ones(len(df)) / len(df) * 100

plt.figure(figsize=(10, 6))
plt.hist(df["ccs"], bins=bins, weights=weights)

plt.ylabel("Percent of total datapoints (%)")
plt.xlabel("CCS (bins of 10, range 100–700)")
plt.title(title)

plt.tight_layout()
plt.show()
