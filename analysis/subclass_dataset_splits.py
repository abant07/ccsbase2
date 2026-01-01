import pandas as pd
import matplotlib.pyplot as plt

train = True
csvfile = "train_data.csv" if train else "test_data.csv"
title = "Training Subclasses" if train else "Test Subclasses"

df = pd.read_csv(f"../pretrained/{csvfile}")

counts = df["subclass"].value_counts()

counts = counts.head(50)

total = counts.sum()
perc = (counts / total) * 100
perc = perc.sort_values(ascending=False)

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
