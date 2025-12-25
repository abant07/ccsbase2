import pandas as pd
import numpy as np
import joblib
import sqlite3

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, median_absolute_error, root_mean_squared_error, r2_score

from utils import Utils
from metrics import Metrics

class CCSBase2:
    
    def __init__(self, database_file, train_filename, test_filename, seed=None, use_metlin=True, subclass_frequency_threshold=None):
        self.database_file = database_file
        self.train_file = train_filename
        self.test_file = test_filename
        self.use_metlin = use_metlin
        self.subclass_frequency_threshold = subclass_frequency_threshold
        self.model = None
        self.cv_metrics = None
        self.metrics = Metrics()
        self.utils = Utils()
        self.seed = 26 if seed is None else seed

        conn = sqlite3.connect(database_file)
        query = "SELECT DISTINCT adduct FROM master_clean"
        adducts = sorted(pd.read_sql_query(query, conn).to_numpy().tolist())
        self.adducts = [adduct[0] for adduct in adducts]

        self.utils.train_test_split_custom (
            database_file=self.database_file,
            train_csv_path=self.train_file,
            test_csv_path=self.test_file,
            test_size=0.2,
            random_state=self.seed,
            use_metlin=use_metlin,
            subclass_frequency_threshold=subclass_frequency_threshold
        )
    
    def fit(self):
        X_list, y_list = [], []
        for _, row in pd.read_csv(self.train_file).iterrows():
            feat_values = self.utils.calculate_descriptors(
                row["smi"], row["mass"], row["z"], self.adducts, row["adduct"]
            )
            if feat_values is not None:
                X_list.append(feat_values)
                y_list.append(row["ccs"])

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=float)

        params = dict(
            objective="reg:squarederror",
            n_estimators=6000,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=30,
            min_child_weight=5,
            gamma=1,
            n_jobs=-1,
            tree_method="hist",
            verbosity=1,
        )

        # -------- 5-fold CV --------
        cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)

        fold_metrics = []
        for fold, (tr_idx, te_idx) in enumerate(cv.split(X), start=1):
            model = XGBRegressor(**params)
            model.fit(X[tr_idx], y[tr_idx])

            y_pred = model.predict(X[te_idx])
            y_true = y[te_idx]

            abs_err = np.abs(y_true - y_pred)
            mae_test = float(abs_err.mean())
            mdae_test = float(np.median(abs_err))
            rmse_test = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

            rel_err = abs_err / y_true * 100  # assumes y_true != 0
            mre_test = float(rel_err.mean())
            mdre_test = float(np.median(rel_err))

            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

            print(f"\nFold {fold}")
            print("MAE:", round(mae_test, 4))
            print("MDAE:", round(mdae_test, 4))
            print("RMSE:", round(rmse_test, 4))
            print("MRE (%):", round(mre_test, 4))
            print("MDRE (%):", round(mdre_test, 4))
            print("R2:", round(r2, 4))

            fold_metrics.append([mae_test, mdae_test, rmse_test, mre_test, mdre_test, r2])

        fold_metrics = np.array(fold_metrics, dtype=float)
        mean_metrics = np.nanmean(fold_metrics, axis=0)
        std_metrics = np.nanstd(fold_metrics, axis=0)

        print("\n=== 5-Fold CV (mean ± std) ===")
        print("MAE:",  round(mean_metrics[0], 4), "±", round(std_metrics[0], 4))
        print("MDAE:", round(mean_metrics[1], 4), "±", round(std_metrics[1], 4))
        print("RMSE:", round(mean_metrics[2], 4), "±", round(std_metrics[2], 4))
        print("MRE (%):",  round(mean_metrics[3], 4), "±", round(std_metrics[3], 4))
        print("MDRE (%):", round(mean_metrics[4], 4), "±", round(std_metrics[4], 4))
        print("R2:",   round(mean_metrics[5], 4), "±", round(std_metrics[5], 4))

        self.cv_metrics = {
            "mae": (round(mean_metrics[0], 4), round(std_metrics[0], 4)),
            "mdae": (round(mean_metrics[1], 4), round(std_metrics[1], 4)),
            "rmse": (round(mean_metrics[2], 4), round(std_metrics[2], 4)),
            "mre_pct": (round(mean_metrics[3], 4), round(std_metrics[3], 4)),
            "mdre_pct": (round(mean_metrics[4], 4), round(std_metrics[4], 4)),
            "r2": (round(mean_metrics[5], 4), round(std_metrics[5], 4)),
        }

        self.model = XGBRegressor(**params)
        self.model.fit(X, y)
        joblib.dump(self.model, "ccsbase2.joblib")

    def predict(self):
        print("Starting Prediction on Test Set")

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
        
        X_list = []
        y_list = []
        metadata = []
        for _, row in pd.read_csv(self.test_file).iterrows():
            feat_values = self.utils.calculate_descriptors(row['smi'], row['mass'], row['z'], self.adducts, row['adduct'])
            if feat_values is not None:
                X_list.append(feat_values)
                y_list.append(row['ccs'])
                metadata.append([row["tag"], row["subclass"], row["adduct"], row["name"], row["smi"], row["z"]])

        X_test = np.asarray(X_list)
        y_test = np.asarray(y_list)
        y_pred_test = self.model.predict(X_test)

        mae_test = mean_absolute_error(y_test, y_pred_test)
        mdae_test = median_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        rmse_test = root_mean_squared_error(y_test, y_pred_test)
        mre_test = mean_relative_error(y_test, y_pred_test)
        mdre_test = median_relative_error(y_test, y_pred_test)

        print("\n=== Test metrics ===")
        print("MAE:", round(mae_test, 4))
        print("MDAE:", round(mdae_test, 4))
        print("RMSE:", round(rmse_test, 4))
        print("MRE (%):", round(mre_test, 4))
        print("MDRE (%):", round(mdre_test, 4))
        print("R2:", round(r2, 4))


        # === Save predictions ===
        df_out = pd.DataFrame({
            "Tag": [m[0] for m in metadata],
            "Subclass": [m[1] for m in metadata],
            "Adduct": [m[2] for m in metadata],
            "Name": [m[3] for m in metadata],
            "SMILES": [m[4] for m in metadata],
            "Charge": [m[5] for m in metadata],
            "CCS_True": y_test,
            "CCS_Pred": y_pred_test
        })
        df_out.to_csv("ccs_predictions_test.csv", index=False)

        self.metrics.generate_metrics_table("ccs_predictions_test.csv", self.cv_metrics)
