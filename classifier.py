import numpy as np
import joblib

from scipy.stats import uniform, randint
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

from constants import XGB_INTEGER_HYPERPARAMS
from db import Database
from training_utils import (
    calculate_sparse_fingerprints,
    build_fingerprint_vocabulary,
    build_fingerprint_index,
    vectorize_fingerprints,
    normalize_hyperparam_range,
)

MODEL_PATH = "ccsbase2_classifier.joblib"
FP_VOCAB_PATH = "ccsbase2_classifier_fp_vocab.joblib"
NOVELTY_DETECTOR_PATH = "ccsbase2_classifier_novelty_detector.joblib"
ENCODER_PATH = "ccsbase2_classifier_encoder.joblib"


class SubclassClassifier:

    def __init__(self, database_file, seed, fp_min_count=5, n_iter=20, min_subclass_count=5,
                 n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9):
        self.database_file = database_file
        self.seed = seed
        self.fp_min_count = fp_min_count
        self.n_iter = n_iter
        self.min_subclass_count = min_subclass_count
        self.model = None
        self.encoder = None
        self.fp_vocab = None
        self.novelty_detector = None
        self.cv_metrics = None
        self.proxy_ood_features = None

        self.hyperparam_ranges = {
            "n_estimators": normalize_hyperparam_range(n_estimators),
            "max_depth": normalize_hyperparam_range(max_depth),
            "learning_rate": normalize_hyperparam_range(learning_rate),
            "subsample": normalize_hyperparam_range(subsample),
            "colsample_bytree": normalize_hyperparam_range(colsample_bytree),
        }

    def fit(self):
        db = Database(self.database_file)
        data = db.read_df(
            "SELECT smi, subclass FROM master_clean WHERE subclass NOT LIKE '%(predicted)' AND subclass IS NOT NULL"
        )
        data = data.drop_duplicates(subset=["smi"], keep="first")
        data["subclass"] = data["subclass"].astype(str)

        subclass_counts = data["subclass"].value_counts()
        training_classes = subclass_counts[subclass_counts >= self.min_subclass_count].index
        is_training_row = data["subclass"].isin(training_classes).to_numpy()

        train_data = data[is_training_row]
        proxy_ood_data = data[~is_training_row]

        print(
            f"Training rows: {len(train_data)} across {train_data['subclass'].nunique()} classes "
            f"(>= {self.min_subclass_count} members)"
        )
        print(
            f"Proxy OOD rows: {len(proxy_ood_data)} across {proxy_ood_data['subclass'].nunique()} classes "
            f"(< {self.min_subclass_count} members)"
        )

        train_fps = calculate_sparse_fingerprints(train_data["smi"])
        self.fp_vocab = build_fingerprint_vocabulary(train_fps, self.fp_min_count)
        fp_index = build_fingerprint_index(self.fp_vocab)
        X_train = vectorize_fingerprints(train_fps, fp_index)
        print(f"Fingerprint vocabulary: {len(self.fp_vocab)} substructures kept (min_count={self.fp_min_count})")

        self.encoder = LabelEncoder()
        y_train = self.encoder.fit_transform(train_data["subclass"].values)

        proxy_ood_fps = calculate_sparse_fingerprints(proxy_ood_data["smi"])
        if proxy_ood_fps:
            self.proxy_ood_features = vectorize_fingerprints(proxy_ood_fps, fp_index)
        else:
            self.proxy_ood_features = np.empty((0, len(self.fp_vocab)))

        self.novelty_detector = IsolationForest(contamination="auto", random_state=self.seed)
        self.novelty_detector.fit(X_train)

        param_distributions = {}
        for name, (low, high) in self.hyperparam_ranges.items():
            is_integer = name in XGB_INTEGER_HYPERPARAMS
            if low == high:
                param_distributions[name] = [int(low) if is_integer else low]
            elif is_integer:
                param_distributions[name] = randint(int(low), int(high) + 1)
            else:
                param_distributions[name] = uniform(loc=low, scale=high - low)

        n_folds = 5
        print(
            f"Running random search over {list(param_distributions.keys())} "
            f"({self.n_iter} candidates x {n_folds} folds = {self.n_iter * n_folds} fits)"
        )
        search = RandomizedSearchCV(
            estimator=XGBClassifier(
                objective="multi:softprob",
                tree_method="hist",
                n_jobs=-1,
                eval_metric="mlogloss",
                random_state=self.seed,
            ),
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed),
            scoring={
                "accuracy": "accuracy",
                "precision_macro": "precision_macro",
                "recall_macro": "recall_macro",
                "f1_macro": "f1_macro",
                "logloss": "neg_log_loss",
            },
            refit="logloss",
            random_state=self.seed,
        )
        search.fit(X_train, y_train)

        best_index = search.best_index_
        cv_results = search.cv_results_
        self.cv_metrics = {
            key: (
                round(cv_results[f"mean_test_{key}"][best_index] * (-1 if key == "logloss" else 1), 4),
                round(cv_results[f"std_test_{key}"][best_index], 4),
            )
            for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "logloss")
        }
        print(f"Best params: {search.best_params_}")
        print(f"CV metrics (winning candidate, mean ± std across folds): {self.cv_metrics}")

        self.model = search.best_estimator_

        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.fp_vocab, FP_VOCAB_PATH)
        joblib.dump(self.novelty_detector, NOVELTY_DETECTOR_PATH)
        joblib.dump(self.encoder, ENCODER_PATH)

    def eval(self):
        if len(self.proxy_ood_features) == 0:
            print("No proxy OOD rows to evaluate.")
            return

        is_novel = self.novelty_detector.predict(self.proxy_ood_features) == -1
        print(
            f"Proxy OOD set: {len(self.proxy_ood_features)} molecules from classes never included in training "
            f"(< {self.min_subclass_count} members)"
        )
        print(f"Flagged as novel by the gate: {is_novel.sum()} ({is_novel.mean() * 100:.1f}%)")

    def predict(self):
        print("Starting Inference")

        model = joblib.load(MODEL_PATH)
        fp_vocab = joblib.load(FP_VOCAB_PATH)
        novelty_detector = joblib.load(NOVELTY_DETECTOR_PATH)
        encoder = joblib.load(ENCODER_PATH)

        db = Database(self.database_file)
        data = db.read_df(
            "SELECT id, smi FROM master_clean WHERE subclass LIKE '%(predicted)' OR subclass IS NULL"
        )

        if len(data) == 0:
            print("No unlabeled rows.")
            return

        fp_index = build_fingerprint_index(fp_vocab)
        fps = calculate_sparse_fingerprints(data["smi"])
        X_test = vectorize_fingerprints(fps, fp_index)

        is_novel = novelty_detector.predict(X_test) == -1

        proba = model.predict_proba(X_test)
        pred_idx = np.argmax(proba, axis=1)
        pred_class = encoder.inverse_transform(pred_idx)

        predicted_class = np.where(is_novel, "NONE", pred_class)
        predicted_class = np.array([f"{c} (predicted)" for c in predicted_class], dtype=object)

        updates = list(zip(predicted_class, data["id"].astype(int)))
        db.write_many("UPDATE master_clean SET subclass = ? WHERE id = ?", updates)

        print(f"Updated {len(updates)} rows in master_clean.subclass")
        print(f"  Flagged NONE via novelty gate: {is_novel.sum()}")


classifier = SubclassClassifier(
    "CCSMLDatabase.db", seed=26, fp_min_count=40,
    n_iter=1, min_subclass_count=30,
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
)

classifier.fit()
classifier.eval()
classifier.predict()
