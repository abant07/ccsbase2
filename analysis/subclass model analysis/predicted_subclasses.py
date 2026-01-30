import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "../CCSMLDatabase.db"
TABLE = "master_clean"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    f"""
    SELECT subclass
    FROM {TABLE}
    WHERE subclass LIKE '% (predicted)'
      AND subclass != 'NONE (predicted)'
    """,
    conn,
)
conn.close()

df["subclass_clean"] = df["subclass"].str.replace(" (predicted)", "", regex=False)

counts_all = df["subclass_clean"].value_counts()
top = counts_all.head(50)
other = counts_all.iloc[50:].sum()
counts = pd.concat([top, pd.Series({"Other": other})]).sort_values(ascending=False)

perc = (counts / counts.sum()) * 100

plt.figure(figsize=(max(8, 0.4 * len(perc)), 6))
bars = plt.bar(perc.index, perc.values)

for i, bar in enumerate(bars):
    bar.set_color(plt.cm.tab20(i % 20))

plt.ylabel("Percent of total datapoints (%)")
plt.xlabel("Predicted Subclasses")

plt.ylim(0, perc.max() * 1.1)
plt.xticks(rotation=45, ha="right", fontsize=8)

plt.tight_layout()
plt.show()
