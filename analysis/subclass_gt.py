import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "../CCSMLDatabase.db"
TABLE = "master_clean"

# pull non-predicted (GT) subclasses
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    f"""
    SELECT subclass
    FROM {TABLE}
    WHERE subclass NOT LIKE '% (predicted)'
    """,
    conn,
)
conn.close()

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

for i, bar in enumerate(bars):
    bar.set_color(plt.cm.tab20(i % 20))

plt.ylabel("Percent of total datapoints (%)")
plt.xlabel("Ground Truth Subclasses")
plt.title("Subclass distribution (Top 100, % of total)")

plt.ylim(0, perc.max() * 1.1)
plt.xticks(rotation=45, ha="right", fontsize=8)

plt.tight_layout()
plt.show()
