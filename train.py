import time

import pandas as pd
import numpy as np
import joblib

from rdkit import Chem
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error, median_absolute_error, root_mean_squared_error, r2_score, make_scorer,
)
from scipy.stats import uniform, randint

from constants import ADDUCT_OFFSETS, XGB_INTEGER_HYPERPARAMS
from db import Database
from utils import calculate_charge
from training_utils import (
    train_test_split_custom,
    featurize_ccs_dataset,
    calculate_base_features,
    calculate_sparse_fingerprint,
    vectorize_sparse_fp,
    build_fingerprint_vocabulary,
    build_fingerprint_index,
    build_feature_matrix,
    normalize_hyperparam_range,
)
from metrics import (
    generate_metrics_table,
    compute_adduct_metrics,
    mean_relative_error,
    median_relative_error,
    peak_memory_usage_mb,
)

MODEL_PATH = "ccsbase2.joblib"
FP_VOCAB_PATH = "ccsbase2_fp_vocab.joblib"
ADDUCTS_PATH = "ccsbase2_adducts.joblib"


class CCSBase2:

    def __init__(self, database_file="CCSMLDatabase.db", train_filename="train_data.csv", test_filename="train_data.csv", 
                 n_estimators=600, max_depth=10,
                 learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, reg_lambda=30, min_child_weight=5, gamma=1,
                 seed=26, use_metlin=True, n_iter=20, fp_min_count=40):
        self.database_file = database_file
        self.train_file = train_filename
        self.test_file = test_filename
        self.use_metlin = use_metlin
        self.seed = seed
        self.model = None
        self.cv_metrics = None
        self.n_iter = n_iter
        self.fp_min_count = fp_min_count
        self.fp_vocab = None

        self.hyperparam_ranges = {
            "n_estimators": normalize_hyperparam_range(n_estimators),
            "max_depth": normalize_hyperparam_range(max_depth),
            "learning_rate": normalize_hyperparam_range(learning_rate),
            "subsample": normalize_hyperparam_range(subsample),
            "colsample_bytree": normalize_hyperparam_range(colsample_bytree),
            "reg_lambda": normalize_hyperparam_range(reg_lambda),
            "min_child_weight": normalize_hyperparam_range(min_child_weight),
            "gamma": normalize_hyperparam_range(gamma),
        }

        adducts_query = "SELECT adduct FROM master_clean GROUP BY adduct HAVING COUNT(*) >= 100 ORDER BY adduct"
        self.adducts = sorted(row[0] for row in Database(database_file).read(adducts_query))

        train_test_split_custom(
            database_file=self.database_file,
            train_csv_path=self.train_file,
            test_csv_path=self.test_file,
            test_size=0.2,
            random_state=self.seed,
            use_metlin=use_metlin,
        )

    def fit(self):
        train_df = pd.read_csv(self.train_file)
        base_features, fp_dicts, y, _ = featurize_ccs_dataset(train_df, self.adducts)

        self.fp_vocab = build_fingerprint_vocabulary(fp_dicts, self.fp_min_count)
        fp_index = build_fingerprint_index(self.fp_vocab)
        X = build_feature_matrix(base_features, fp_dicts, fp_index)

        print(f"Fingerprint vocabulary: {len(self.fp_vocab)} substructures kept (min_count={self.fp_min_count})")

        fixed_params = {}
        param_distributions = {}
        for name, (low, high) in self.hyperparam_ranges.items():
            is_integer = name in XGB_INTEGER_HYPERPARAMS
            if low == high:
                fixed_params[name] = int(low) if is_integer else low
            elif is_integer:
                param_distributions[name] = randint(int(low), int(high) + 1)
            else:
                param_distributions[name] = uniform(loc=low, scale=high - low)

        base_estimator = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            tree_method="hist",
            verbosity=1,
            random_state=self.seed,
            **fixed_params,
        )

        if not param_distributions:
            fit_start = time.perf_counter()
            self.model = base_estimator
            self.model.fit(X, y)
            fit_time = time.perf_counter() - fit_start
            self.cv_metrics = None
            print(f"No hyperparameters to search -- fit directly in {fit_time:.2f}s")
        else:
            scoring = {
                "mae": "neg_mean_absolute_error",
                "mdae": "neg_median_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "mre_pct": make_scorer(mean_relative_error, greater_is_better=False),
                "mdre_pct": make_scorer(median_relative_error, greater_is_better=False),
                "r2": "r2",
            }

            n_folds = 5
            print(
                f"Running random search over {list(param_distributions.keys())} "
                f"({self.n_iter} candidates x {n_folds} folds = {self.n_iter * n_folds} fits)"
            )

            search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_distributions,
                n_iter=self.n_iter,
                cv=KFold(n_splits=n_folds, shuffle=True, random_state=self.seed),
                scoring=scoring,
                refit="rmse",
                random_state=self.seed,
                n_jobs=1,
                verbose=3,
            )

            search_start = time.perf_counter()
            search.fit(X, y)
            search_time = time.perf_counter() - search_start

            print(f"Random search complete in {search_time:.2f}s | best params: {search.best_params_}")

            best_index = search.best_index_
            cv_results = search.cv_results_
            negated_metrics = {"mae", "mdae", "rmse", "mre_pct", "mdre_pct"}
            self.cv_metrics = {
                key: (
                    round(cv_results[f"mean_test_{key}"][best_index] * (-1 if key in negated_metrics else 1), 4),
                    round(cv_results[f"std_test_{key}"][best_index], 4),
                )
                for key in ("mae", "mdae", "rmse", "mre_pct", "mdre_pct", "r2")
            }
            print(f"CV metrics (winning candidate, mean ± std across folds): {self.cv_metrics}")

            self.model = search.best_estimator_

        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.fp_vocab, FP_VOCAB_PATH)
        joblib.dump(self.adducts, ADDUCTS_PATH)

        print(f"Trained on {len(X)} rows | Peak memory usage: {peak_memory_usage_mb():.1f} MB")

    def eval(self):
        print("Starting Evaluation on Test Set")

        test_df = pd.read_csv(self.test_file)
        base_features, fp_dicts, y_test, metadata = featurize_ccs_dataset(test_df, self.adducts)

        fp_index = build_fingerprint_index(self.fp_vocab)
        X_test = build_feature_matrix(base_features, fp_dicts, fp_index)

        predict_start = time.perf_counter()
        y_pred_test = self.model.predict(X_test)
        predict_time = time.perf_counter() - predict_start

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
        print(
            f"\nInference time: {predict_time:.4f}s for {len(X_test)} rows "
            f"({predict_time / max(len(X_test), 1) * 1000:.3f} ms/row) | "
            f"Peak memory usage: {peak_memory_usage_mb():.1f} MB"
        )

        df_out = pd.DataFrame({
            "Tag": [m[0] for m in metadata],
            "Subclass": [m[1] for m in metadata],
            "Adduct": [m[2] for m in metadata],
            "Name": [m[3] for m in metadata],
            "SMILES": [m[4] for m in metadata],
            "Charge": [m[5] for m in metadata],
            "CCS_True": y_test,
            "CCS_Pred": y_pred_test,
        })
        df_out.to_csv("testset_predictions.csv", index=False)

        generate_metrics_table("testset_predictions.csv", self.cv_metrics)
        compute_adduct_metrics(df_out, output_csv="adduct_metrics.csv")

    def predict(self, input_csv):
        """Deployed-model inference on an arbitrary CSV (columns: smi, adduct) -- loads from joblib artifacts."""
        print("Starting Inference")

        model = joblib.load(MODEL_PATH)
        fp_vocab = joblib.load(FP_VOCAB_PATH)
        adducts = joblib.load(ADDUCTS_PATH)
        fp_index = build_fingerprint_index(fp_vocab)

        df = pd.read_csv(input_csv)
        required_columns = ["smi", "adduct"]
        missing_columns = [c for c in required_columns if c not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {input_csv}: {missing_columns}")

        feature_rows, valid_idx = [], []
        for i, row in df.iterrows():
            smi, adduct = str(row["smi"]), str(row["adduct"])

            if adduct not in ADDUCT_OFFSETS:
                print(f"Skipping row {i}: adduct '{adduct}' not supported")
                continue
            if Chem.MolFromSmiles(smi) is None:
                print(f"Skipping row {i}: invalid SMILES '{smi}'")
                continue

            charge = calculate_charge(adduct)
            base = calculate_base_features(smi, charge, adducts, adduct)
            fp = calculate_sparse_fingerprint(smi)

            feature_rows.append(np.concatenate([base, vectorize_sparse_fp(fp, fp_index)]))
            valid_idx.append(i)

        if not feature_rows:
            raise RuntimeError("No rows could be featurized. Check SMILES/adduct values.")

        X = np.asarray(feature_rows, dtype=float)

        predict_start = time.perf_counter()
        predictions = model.predict(X)
        predict_time = time.perf_counter() - predict_start

        output_csv = input_csv[:-4] + "_predictions.csv"
        df["pred_ccs"] = np.nan
        df.loc[valid_idx, "pred_ccs"] = predictions
        df.to_csv(output_csv, index=False)

        print(f"Wrote: {output_csv}")
        print(f"Predicted rows: {len(valid_idx)} / {len(df)}")
        if len(valid_idx) != len(df):
            print("Some rows were skipped because of unsupported adducts or invalid SMILES (pred_ccs = NaN).")
        print(
            f"Inference time: {predict_time:.4f}s for {len(X)} rows "
            f"({predict_time / max(len(X), 1) * 1000:.3f} ms/row)"
        )


ccs_model = CCSBase2("CCSMLDatabase.db",
                       "train_data.csv",
                       "test_data.csv",
                       n_estimators=6000,
                       max_depth=10,
                       learning_rate=0.03,
                       subsample=0.9,
                       colsample_bytree=0.9,
                       reg_lambda=30,
                       min_child_weight=5,
                       gamma=1,
                       seed=26,
                       use_metlin=True,
                       fp_min_count=40
                    )
ccs_model.fit()
ccs_model.eval()