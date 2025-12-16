import pandas as pd
import numpy as np
import joblib
import sqlite3

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, make_scorer

from utils import Utils
from metrics import Metrics

class CCSMLModel:
    
    def __init__(self, database_file, train_filename, test_filename, use_metlin=True, subclass_frequency_threshold=None):
        self.database_file = database_file
        self.train_file = train_filename
        self.test_file = test_filename
        self.use_metlin = use_metlin
        self.subclass_frequency_threshold = subclass_frequency_threshold
        self.model = None
        self.cv_score = None
        self.metrics = Metrics()
        self.utils = Utils()

        conn = sqlite3.connect(database_file)
        query = "SELECT DISTINCT adduct FROM master_clean"
        adducts = sorted(pd.read_sql_query(query, conn).to_numpy().tolist())
        self.adducts = [adduct[0] for adduct in adducts]

        self.utils.train_test_split_custom (
            database_file=self.database_file,
            train_csv_path=self.train_file,
            test_csv_path=self.test_file,
            test_size=0.2,
            random_state=26,
            use_metlin=use_metlin,
            subclass_frequency_threshold=subclass_frequency_threshold
        )
    
    def fit(self):
        def mdre(y_true, y_pred):
            # relative error, safe division
            denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
            rel = np.abs(y_true - y_pred) / denom
            return np.nanmedian(rel)
        
        mdre_scorer = make_scorer(mdre, greater_is_better=False)

        X_list = []
        y_list = []
        for _, row in pd.read_csv(self.train_file).iterrows():
            feat_values = self.utils.calculate_descriptors(row['smi'], row['mass'], row['z'], row['instrument'], self.adducts, row['adduct'])
            if feat_values is not None:
                X_list.append(feat_values)
                y_list.append(row['ccs'])

        X_train = np.array(X_list, dtype=float)
        y_train = np.array(y_list, dtype=float)

        # 5-fold CV
        cv = KFold(n_splits=5, shuffle=True, random_state=26)

        base_model = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            tree_method="hist",
            verbosity=1
        )

        # Use your params, but as a "grid" of single values
        param_grid = {
            "n_estimators":      [6000],
            "max_depth":         [10],
            "learning_rate":     [0.03],
            "subsample":         [0.9],
            "colsample_bytree":  [0.9],
            "reg_lambda":        [30],
            "min_child_weight":  [5],
            "gamma":             [1],
        }

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=mdre_scorer,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True,  # refit on full training data with best params
        )

        # Fit with cross-validation
        grid.fit(X_train, y_train)

        print("Best params:", grid.best_params_)
        print("Best CV score (MdRE):", -grid.best_score_ * 100)

        # Save the refit best model
        self.model = grid.best_estimator_
        self.cv_score = -grid.best_score_ * 100
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
            feat_values = self.utils.calculate_descriptors(row['smi'], row['mass'], row['z'], row['instrument'], self.adducts, row['adduct'])
            if feat_values is not None:
                X_list.append(feat_values)
                y_list.append(row['ccs'])
                metadata.append([row["tag"], row["subclass"], row["adduct"], row["name"], row["smi"], row["z"], row['instrument']])

        X_test = np.asarray(X_list)
        y_test = np.asarray(y_list)
        y_pred_test = self.model.predict(X_test)

        mae_test = mean_absolute_error(y_test, y_pred_test)
        mre_test = mean_relative_error(y_test, y_pred_test)
        mdre_test = median_relative_error(y_test, y_pred_test)

        print("\n=== Test metrics ===")
        print("MAE :", round(mae_test, 4))
        print("MRE (%):", round(mre_test, 4))
        print("MDRE (%):", round(mdre_test, 4))


        # === Save predictions ===
        df_out = pd.DataFrame({
            "Tag": [m[0] for m in metadata],
            "Subclass": [m[1] for m in metadata],
            "Adduct": [m[2] for m in metadata],
            "Name": [m[3] for m in metadata],
            "SMILES": [m[4] for m in metadata],
            "Charge": [m[5] for m in metadata],
            "Instrument": [m[6] for m in metadata],
            "CCS_True": y_test,
            "CCS_Pred": y_pred_test
        })
        df_out.to_csv("ccs_predictions_test.csv", index=False)

        self.metrics.generate_metrics_table("ccs_predictions_test.csv", self.cv_score)
        self.metrics.generate_subclass_error_bin_tables("ccs_predictions_test.csv")
        self.metrics.generate_adduct_error_table("ccs_predictions_test.csv")

ccs_model = CCSMLModel("CCSMLDB.db", 
                       "train_data.csv", 
                       "test_data.csv", 
                       use_metlin=False, 
                       subclass_frequency_threshold=40)
ccs_model.fit()
ccs_model.predict()
