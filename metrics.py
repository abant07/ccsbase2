import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Metrics:
    def generate_metrics_table(self, csv_path: str, cv_score=None, output_image: str = "metrics_table.png"):
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

        thresh_1 = (rel_error < 1).sum()
        thresh_2 = (rel_error < 2).sum()
        thresh_3 = (rel_error < 3).sum()
        thresh_5 = (rel_error < 5).sum()

        pct_1 = thresh_1 / total * 100
        pct_2 = thresh_2 / total * 100
        pct_3 = thresh_3 / total * 100
        pct_5 = thresh_5 / total * 100

        data = {
            "Metric": [
                "MAE (Å)",
                "CV MAE (Å)",
                "MDAE (Å)",
                "CV MDAE (Å)",
                "RMSE (Å)",
                "CV RMSE (Å)",
                "MRE (%)",
                "CV MRE (%)",
                "MDRE (%)",
                "CV MDRE (%)",
                "R²",
                "CV R²",
                "Total predictions (n)",
                "Predictions <1% RE",
                "Predictions <2% RE",
                "Predictions <3% RE",
                "Predictions <5% RE",
            ],
            "Value": [
                f"{mae:.3f}",
                f"{cv_score['mae'][0]} ± {cv_score['mae'][1]}",
                f"{mdae:.3f}",
                f"{cv_score['mdae'][0]} ± {cv_score['mdae'][1]}",
                f"{rmse:.3f}",
                f"{cv_score['rmse'][0]} ± {cv_score['rmse'][1]}",
                f"{mre:.2f}",
                f"{cv_score['mre_pct'][0]} ± {cv_score['mre_pct'][1]}",
                f"{mdre:.2f}",
                f"{cv_score['mdre_pct'][0]} ± {cv_score['mdre_pct'][1]}",
                f"{r2:.3f}",
                f"{cv_score['r2'][0]} ± {cv_score['r2'][1]}",
                f"{total:,}",
                f"{thresh_1:,} ({pct_1:.1f}%)",
                f"{thresh_2:,} ({pct_2:.1f}%)",
                f"{thresh_3:,} ({pct_3:.1f}%)",
                f"{thresh_5:,} ({pct_5:.1f}%)",
            ],
            "Description / Note": [
                "Mean Absolute Error (test)",
                "5-fold CV mean ± std",
                "Median Absolute Error (test)",
                "5-fold CV mean ± std",
                "Root Mean Squared Error (test)",
                "5-fold CV mean ± std",
                "Mean Relative Error (test)",
                "5-fold CV mean ± std",
                "Median Relative Error (test)",
                "5-fold CV mean ± std",
                "Coefficient of Determination (test)",
                "5-fold CV mean ± std",
                "",
                "Extremely accurate predictions",
                "Accurate predictions",
                "Typically considered good",
                "Decent",
            ],
        }

        table_df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(10, 5.4))
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

