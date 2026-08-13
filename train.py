import time

import pandas as pd
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer

from constants import ADDUCT_OFFSETS, ADDUCT_STANDARDIZATION
from db import Database
from utils import (
    train_test_split_custom,
    featurize_ccs_dataset,
    calculate_base_features,
    calculate_sparse_fingerprint,
    load_or_build_fingerprint_vocabulary,
    build_fingerprint_index,
    build_feature_matrix_sparse,
)
from metrics import (
    compute_adduct_metrics,
    compute_ccs_metrics,
    mean_relative_error,
    median_relative_error,
    peak_memory_usage_mb,
)

MODEL_PATH = "ccsbase2.joblib"
ADDUCTS_PATH = "ccsbase2_adduct_list.joblib"


class CCSBase2:

    def __init__(self, database_file="CCSMLDatabase.db",
                 n_estimators=(6000,), max_depth=(10,),
                 learning_rate=(0.03,), subsample=(0.9,), colsample_bytree=(0.9,), reg_lambda=(30,),
                 min_child_weight=(5,), gamma=(1,),
                 seed=26, use_metlin=True, fp_vocab_file="ccsbase2_fp_vocab.joblib"):
        self.database_file = database_file
        self.use_metlin = use_metlin
        self.seed = seed
        self.model = None
        self.cv_metrics = None
        self.fp_vocab_file = fp_vocab_file
        self.fp_vocab = None

        self.param_grid = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
        }

        adducts_query = "SELECT adduct FROM master_clean GROUP BY adduct HAVING COUNT(*) >= 50 ORDER BY adduct"
        self.adducts = sorted(row[0] for row in Database(database_file).read(adducts_query))

        train_test_split_custom(
            database_file=self.database_file,
            test_size=0.2,
            random_state=self.seed,
            use_metlin=use_metlin,
        )

    def fit(self):
        train_df = pd.read_csv("train_data.csv")
        base_features, fp_dicts, y, _ = featurize_ccs_dataset(train_df, self.adducts)

        self.fp_vocab = load_or_build_fingerprint_vocabulary(self.database_file, self.fp_vocab_file)
        fp_index = build_fingerprint_index(self.fp_vocab)
        X = build_feature_matrix_sparse(base_features, fp_dicts, fp_index)

        print(f"Fingerprint vocabulary: {len(self.fp_vocab)} substructures (loaded from {self.fp_vocab_file})")

        base_estimator = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            tree_method="hist",
            verbosity=1,
            random_state=self.seed,
        )

        scoring = {
            "mae": "neg_mean_absolute_error",
            "mdae": "neg_median_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "mre_pct": make_scorer(mean_relative_error, greater_is_better=False),
            "mdre_pct": make_scorer(median_relative_error, greater_is_better=False),
            "r2": "r2",
        }

        n_folds = 5
        n_candidates = int(np.prod([len(values) for values in self.param_grid.values()]))
        print(
            f"Running grid search over {list(self.param_grid.keys())} "
            f"({n_candidates} candidates x {n_folds} folds = {n_candidates * n_folds} fits)"
        )

        search = GridSearchCV(
            estimator=base_estimator,
            param_grid=self.param_grid,
            cv=KFold(n_splits=n_folds, shuffle=True, random_state=self.seed),
            scoring=scoring,
            refit="rmse",
            n_jobs=1,
            verbose=3,
        )

        search_start = time.perf_counter()
        search.fit(X, y)
        search_time = time.perf_counter() - search_start

        print(f"Grid search complete in {search_time:.2f}s | best params: {search.best_params_}")

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
        joblib.dump(self.adducts, ADDUCTS_PATH)

        print(f"Trained on {X.shape[0]} rows | Peak memory usage: {peak_memory_usage_mb():.1f} MB")

    def eval(self):
        print("Starting Evaluation on Test Set")

        test_df = pd.read_csv("test_data.csv")
        base_features, fp_dicts, y_test, metadata = featurize_ccs_dataset(test_df, self.adducts)

        fp_index = build_fingerprint_index(self.fp_vocab)
        X_test = build_feature_matrix_sparse(base_features, fp_dicts, fp_index)

        predict_start = time.perf_counter()
        y_pred_test = self.model.predict(X_test)
        predict_time = time.perf_counter() - predict_start

        compute_ccs_metrics(y_test, y_pred_test, label="Test metrics")
        print(
            f"\nInference time: {predict_time:.4f}s for {X_test.shape[0]} rows "
            f"({predict_time / max(X_test.shape[0], 1) * 1000:.3f} ms/row) | "
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

        compute_adduct_metrics(df_out, output_csv="adduct_metrics.csv")

    def predict(self, input_csv: str, ccs_ground_truth_col=None):
        """Deployed-model inference on an arbitrary CSV (columns: smi, adduct) -- loads from joblib artifacts."""
        print("Starting Inference")

        model = joblib.load(MODEL_PATH)
        fp_vocab = joblib.load(self.fp_vocab_file)
        adducts = joblib.load(ADDUCTS_PATH)
        fp_index = build_fingerprint_index(fp_vocab)

        df = pd.read_csv(input_csv)
        required_columns = ["smi", "adduct"]
        missing_columns = [c for c in required_columns if c not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {input_csv}: {missing_columns}")
        if ccs_ground_truth_col is not None and ccs_ground_truth_col not in df.columns:
            raise ValueError(f"Ground truth column '{ccs_ground_truth_col}' not found in {input_csv}")

        base_features, fp_dicts, valid_idx = [], [], []
        for i, row in df.iterrows():
            smi, adduct = str(row["smi"]), str(row["adduct"])
            adduct = ADDUCT_STANDARDIZATION.get(adduct, adduct)

            if adduct not in ADDUCT_OFFSETS:
                print(f"Skipping row {i}: adduct '{adduct}' not supported. See constants.py for supported adducts.")
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"Skipping row {i}: invalid SMILES '{smi}'")
                continue

            neutral_mass = rdMolDescriptors.CalcExactMolWt(Chem.AddHs(mol))
            ion_mass = neutral_mass + ADDUCT_OFFSETS[adduct]
            base_features.append(calculate_base_features(smi, ion_mass, adducts, adduct))
            fp_dicts.append(calculate_sparse_fingerprint(smi))
            valid_idx.append(i)

        if not base_features:
            raise RuntimeError("No rows could be featurized. Check SMILES/adduct values.")

        X = build_feature_matrix_sparse(base_features, fp_dicts, fp_index)

        predict_start = time.perf_counter()
        predictions = model.predict(X)
        predict_time = time.perf_counter() - predict_start

        output_csv = input_csv[:-4] + "_predictions.csv"
        df["ccs_pred"] = np.nan
        df.loc[valid_idx, "ccs_pred"] = predictions
        df.to_csv(output_csv, index=False)

        print(f"Wrote: {output_csv}")
        print(f"Predicted rows: {len(valid_idx)} / {len(df)}")
        if len(valid_idx) != len(df):
            print("Some rows were skipped because of unsupported adducts or invalid SMILES (ccs_pred = NaN).")
        print(
            f"Inference time: {predict_time:.4f}s for {X.shape[0]} rows "
            f"({predict_time / max(X.shape[0], 1) * 1000:.3f} ms/row)"
        )

        if ccs_ground_truth_col is not None:
            scored = df.loc[valid_idx, ["ccs_pred", ccs_ground_truth_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(scored) == 0:
                print(f"\nNo rows with usable ground truth in '{ccs_ground_truth_col}' -- skipping metrics.")
            else:
                compute_ccs_metrics(
                    scored[ccs_ground_truth_col], scored["ccs_pred"],
                    label=f"Metrics vs '{ccs_ground_truth_col}'",
                )


ccs_model = CCSBase2("CCSMLDatabase.db",
                    n_estimators=[6000, 7000, 8000, 10000, 12000, 14000, 20000],
                    max_depth=[8, 10, 12, 13, 15],
                    learning_rate=[0.01, 0.02, 0.03],
                    subsample=[0.9],
                    colsample_bytree=[0.5, 0.9],
                    reg_lambda=[30],
                    min_child_weight=[1],
                    gamma=[1],
                    seed=26,
                    use_metlin=True,
                    fp_vocab_file="ccsbase2_fp_vocab.joblib"
                )
ccs_model.fit()
ccs_model.eval()
ccs_model.predict("./datasets/ood_testset.csv", "ccs")