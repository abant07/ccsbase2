import pandas as pd
import matplotlib.pyplot as plt

class Metrics:
    def generate_metrics_table(self, csv_path: str, cv_score=-1, output_image: str = "metrics_table.png"):
        df = pd.read_csv(csv_path)
        
        abs_error = abs(df["CCS_True"] - df["CCS_Pred"])
        rel_error = abs_error / df["CCS_True"] * 100
        
        mae = abs_error.mean()
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
                "MRE (%)",
                "MDRE (%)",
                "CV MDRE (%)",
                "Total predictions (n)",
                "Predictions <1% RE",
                "Predictions <2% RE",
                "Predictions <3% RE",
                "Predictions <5% RE"
            ],
            "Value": [
                f"{mae:.3f}",
                f"{mre:.2f}%",
                f"{mdre:.2f}%",
                f"{'NA' if cv_score == -1 else cv_score:.2f}",
                f"{total:,}",
                f"{thresh_1:,} ({pct_1:.1f}%)",
                f"{thresh_2:,} ({pct_2:.1f}%)",
                f"{thresh_3:,} ({pct_3:.1f}%)",
                f"{thresh_5:,} ({pct_5:.1f}%)"
            ],
            "Description / Note": [
                "Mean Absolute Error",
                "Mean Relative Error",
                "Median Relative Error",
                "CV Median Relative Error",
                "",
                "Extremely accurate predictions",
                "Accurate predictions",
                "Typically considered good",
                "Decent"
            ]
        }
        
        table_df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.axis("off")
        table = ax.table(cellText=table_df.values,
                        colLabels=table_df.columns,
                        cellLoc="center",
                        loc="center")
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.4)
        
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor("#4472C4")
                cell.set_text_props(weight="bold", color="white")
            cell.set_edgecolor("#D3D3D3")
            cell.set_height(0.1)
            if j == 0:
                cell.get_text().set_weight("bold")
        
        plt.savefig(output_image, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        
    
    def _make_table_with_legend(self, table_df, legend_map, title, output_image,
                            header_color="#455A64", header_text_color="white",
                            even_row_color="#ECEFF1", odd_row_color="#FFFFFF"):

        n_rows = len(table_df)
        fig_height = max(2.8, 0.42 * n_rows + 1.4)

        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc="center",
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.15, 1.55)

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor(header_color)
                cell.set_text_props(weight="bold", color=header_text_color)
            else:
                cell.set_facecolor(even_row_color if i % 2 == 0 else odd_row_color)
            cell.set_edgecolor("#CFD8DC")

        plt.title(title, fontsize=14, pad=16, weight="bold")

        legend_text = "\n".join(f"{i+1}: {name}" for i, name in enumerate(legend_map))
        plt.figtext(0.02, 0.01, legend_text, ha="left", va="bottom", fontsize=10)

        plt.savefig(output_image, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()


    def _make_table_no_legend(self, table_df, title, output_image,
                            header_color="#455A64", header_text_color="white",
                            even_row_color="#ECEFF1", odd_row_color="#FFFFFF"):

            n_rows = len(table_df)
            fig_height = max(2.5, 0.4 * n_rows + 1.3)

            fig, ax = plt.subplots(figsize=(10, fig_height))
            ax.axis("off")

            table = ax.table(
                cellText=table_df.values,
                colLabels=table_df.columns,
                cellLoc="center",
                loc="center"
            )

            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.15, 1.55)

            for (i, j), cell in table.get_celld().items():
                if i == 0:
                    cell.set_facecolor(header_color)
                    cell.set_text_props(weight="bold", color=header_text_color)
                else:
                    cell.set_facecolor(even_row_color if i % 2 == 0 else odd_row_color)
                cell.set_edgecolor("#CFD8DC")

            plt.title(title, fontsize=14, pad=16, weight="bold")

            plt.savefig(output_image, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()


    def generate_subclass_error_bin_tables(self, csv_path, output_prefix="subclass_error_bin_"):
            df = pd.read_csv(csv_path)

            df["AbsError"] = (df["CCS_True"] - df["CCS_Pred"]).abs()
            df["RelError"] = df["AbsError"] / df["CCS_True"] * 100

            bins = [
                ("lt1", df["RelError"] < 1, "Top 5 Subclasses for RelError < 1%"),
                ("1to2", (df["RelError"] > 1) & (df["RelError"] <= 2), "Top 5 Subclasses for 1% < RelError ≤ 2%"),
                ("2to3", (df["RelError"] > 2) & (df["RelError"] < 3), "Top 5 Subclasses for 2% < RelError < 3%"),
                ("3to5", (df["RelError"] > 3) & (df["RelError"] <= 5), "Top 5 Subclasses for 3% < RelError ≤ 5%"),
            ]

            for suffix, mask, title in bins:
                df_bin = df[mask]
                if df_bin.empty:
                    continue

                grouped = df_bin.groupby("Subclass").agg(
                    count=("CCS_True", "size"),
                    MAE=("AbsError", "mean"),
                    MDRE=("RelError", "median"),
                    MRE=("RelError", "mean")
                ).reset_index()

                grouped = grouped.sort_values("count", ascending=False).head(5)

                grouped["count"] = grouped["count"].astype(int)
                grouped["MAE"] = grouped["MAE"].round(3)
                grouped["MDRE"] = grouped["MDRE"].round(2)
                grouped["MRE"] = grouped["MRE"].round(2)

                grouped["ClassDisplay"] = grouped["Subclass"].astype(str).apply(
                    lambda s: s if len(s) <= 12 else s[:12] + "..."
                )

                legend_list = grouped["Subclass"].tolist()

                table_df = grouped[["ClassDisplay", "count", "MAE", "MDRE", "MRE"]].copy()
                table_df.columns = ["Subclass", "# of points", "MAE", "MDRE (%)", "MRE (%)"]

                output_image = f"{output_prefix}{suffix}.png"

                self._make_table_with_legend(
                    table_df,
                    legend_list,
                    title,
                    output_image,
                    header_color="#37474F"
                )


    def generate_adduct_error_table(self, csv_path, output_image="adduct_error_table.png"):
            df = pd.read_csv(csv_path)

            df["AbsError"] = (df["CCS_True"] - df["CCS_Pred"]).abs()
            df["RelError"] = df["AbsError"] / df["CCS_True"] * 100

            grouped = df.groupby("Adduct").agg(
                count=("CCS_True", "size"),
                MAE=("AbsError", "mean"),
                MDRE=("RelError", "median"),
                MRE=("RelError", "mean")
            ).reset_index()

            grouped = grouped.sort_values("count", ascending=False)

            grouped["count"] = grouped["count"].astype(int)
            grouped["MAE"] = grouped["MAE"].round(3)
            grouped["MDRE"] = grouped["MDRE"].round(2)
            grouped["MRE"] = grouped["MRE"].round(2)

            table_df = grouped[["Adduct", "count", "MAE", "MDRE", "MRE"]].copy()
            table_df.columns = ["Adduct", "# of points", "MAE", "MDRE (%)", "MRE (%)"]

            self._make_table_no_legend(
                table_df,
                "Error Metrics by Adduct (Full Test Set)",
                output_image,
                header_color="#00695C"
            )

