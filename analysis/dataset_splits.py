import pandas as pd
import matplotlib.pyplot as plt

train = True
csvfile = "train_data.csv" if train else "test_data.csv"
title = "Training Subclass Distribution" if train else "Test Subclass Distribution"

# load data
df = pd.read_csv(f"../pretrained/{csvfile}")

# count subclasses
counts = df["subclass"].value_counts()

# keep top 100
counts = counts.head(100)

# scale to percent of total datapoints
total = counts.sum()
perc = (counts / total) * 100
perc = perc.sort_values(ascending=False)

# vertical histogram (bar chart) with different colors
plt.figure(figsize=(max(8, 0.4 * len(perc)), 6))
bars = plt.bar(perc.index, perc.values)

# apply different colors automatically
for i, bar in enumerate(bars):
    bar.set_color(plt.cm.tab20(i % 20))

plt.ylabel("Percent of total datapoints (%)")
plt.xlabel(title)
plt.title("Chemical subclass distribution (Top 100, % of total)")

plt.ylim(0, perc.max() * 1.1)

# smaller, slanted labels
plt.xticks(rotation=45, ha="right", fontsize=8)

plt.tight_layout()
plt.show()
