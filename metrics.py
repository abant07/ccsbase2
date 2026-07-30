import resource
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


def macro_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return float(precision), float(recall), float(f1)


def softmax_entropy(proba, eps=1e-12):
    clipped_proba = np.clip(proba, eps, 1.0)
    return -np.sum(clipped_proba * np.log(clipped_proba), axis=1)


def mean_relative_error(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rel_err = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
    return float(np.mean(rel_err)) * 100


def median_relative_error(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rel_err = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
    return float(np.median(rel_err)) * 100


def peak_memory_usage_mb():
    peak_kb_or_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak_kb_or_bytes / (1024 ** 2) if sys.platform == "darwin" else peak_kb_or_bytes / 1024


def generate_metrics_table(csv_path: str, cv_score=None, output_image: str = "metrics_table.png"):
    df = pd.read_csv(csv_path)

    y_true = df["CCS_True"]
    y_pred = df["CCS_Pred"]

    abs_error = (y_true - y_pred).abs()
    rel_error = abs_error / y_true * 100

    mae = abs_error.mean()
    mdae = abs_error.median()
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    mre = rel_error.mean()
    mdre = rel_error.median()

    total = len(df)

    count_under_1pct = (rel_error < 1).sum()
    count_under_2pct = (rel_error < 2).sum()
    count_under_3pct = (rel_error < 3).sum()
    count_under_5pct = (rel_error < 5).sum()

    pct_under_1pct = count_under_1pct / total * 100
    pct_under_2pct = count_under_2pct / total * 100
    pct_under_3pct = count_under_3pct / total * 100
    pct_under_5pct = count_under_5pct / total * 100

    metric_specs = [
        ("MAE (Å)", f"{mae:.3f}", "mae", "Mean Absolute Error (test)"),
        ("MDAE (Å)", f"{mdae:.3f}", "mdae", "Median Absolute Error (test)"),
        ("RMSE (Å)", f"{rmse:.3f}", "rmse", "Root Mean Squared Error (test)"),
        ("MRE (%)", f"{mre:.2f}", "mre_pct", "Mean Relative Error (test)"),
        ("MDRE (%)", f"{mdre:.2f}", "mdre_pct", "Median Relative Error (test)"),
        ("R²", f"{r2:.3f}", "r2", "Coefficient of Determination (test)"),
    ]

    rows = []
    for label, value, cv_key, description in metric_specs:
        rows.append((label, value, description))
        if cv_score is not None:
            mean, std = cv_score[cv_key]
            rows.append((f"CV {label}", f"{mean} ± {std}", "5-fold CV mean ± std"))

    rows.append(("Total predictions (n)", f"{total:,}", ""))
    rows.append(("Predictions <1% RE", f"{count_under_1pct:,} ({pct_under_1pct:.1f}%)", "Extremely accurate predictions"))
    rows.append(("Predictions <2% RE", f"{count_under_2pct:,} ({pct_under_2pct:.1f}%)", "Accurate predictions"))
    rows.append(("Predictions <3% RE", f"{count_under_3pct:,} ({pct_under_3pct:.1f}%)", "Typically considered good"))
    rows.append(("Predictions <5% RE", f"{count_under_5pct:,} ({pct_under_5pct:.1f}%)", "Decent"))

    table_df = pd.DataFrame(rows, columns=["Metric", "Value", "Description / Note"])

    _, ax = plt.subplots(figsize=(10, 5.4))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.2)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(weight="bold", color="white")
        cell.set_edgecolor("#D3D3D3")
        cell.set_height(0.09)
        if j == 0:
            cell.get_text().set_weight("bold")

    plt.savefig(output_image, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def compute_adduct_metrics(df: pd.DataFrame, output_csv: str = None) -> pd.DataFrame:
    rows = []
    for adduct, group in df.groupby("Adduct"):
        y_true = group["CCS_True"].to_numpy(dtype=float)
        y_pred = group["CCS_Pred"].to_numpy(dtype=float)
        n = len(group)

        abs_error = np.abs(y_true - y_pred)
        rel_error = abs_error / y_true * 100

        mae = abs_error.mean()
        mdae = np.median(abs_error)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mre = rel_error.mean()
        mdre = np.median(rel_error)

        if n > 1:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        else:
            r2 = np.nan

        rows.append({
            "Adduct": adduct,
            "N": n,
            "MAE": round(float(mae), 4),
            "MDAE": round(float(mdae), 4),
            "RMSE": round(float(rmse), 4),
            "MRE (%)": round(float(mre), 4),
            "MDRE (%)": round(float(mdre), 4),
            "R2": round(float(r2), 4) if not np.isnan(r2) else np.nan,
        })

    result = pd.DataFrame(rows).sort_values("N", ascending=False).reset_index(drop=True)

    print(f"\n=== Per-Adduct Test Metrics ({len(result)} adducts) ===")
    print(result.to_string(index=False))

    if output_csv:
        result.to_csv(output_csv, index=False)

    return result
