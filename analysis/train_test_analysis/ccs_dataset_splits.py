import pandas as pd
import matplotlib.pyplot as plt

train = False
csvfile = "train_data.csv" if train else "test_data.csv"
title = "Training Subclasses" if train else "Test Subclasses"

df = pd.read_csv(f"../pretrained/{csvfile}")

counts_all = df["subclass"].value_counts()
top = counts_all.head(50)
other = counts_all.iloc[50:].sum()

counts = pd.concat([top, pd.Series({"Other": other})]).sort_values(ascending=False)

perc = (counts / counts.sum()) * 100

plt.figure(figsize=(max(8, 0.4 * len(perc)), 6))
bars = plt.bar(perc.index, perc.values)

for i, bar in enumerate(bars):
    bar.set_color(plt.cm.tab20(i % 20))

plt.ylabel("Percent of total datapoints (%)")
plt.xlabel(title)

plt.ylim(0, perc.max() * 1.1)
plt.xticks(rotation=45, ha="right", fontsize=8)

plt.tight_layout()
plt.show()
