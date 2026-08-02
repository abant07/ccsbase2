import numpy as np
import joblib

from rdkit import DataStructs
from scipy.stats import uniform, randint
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from constants import XGB_INTEGER_HYPERPARAMS
from db import Database
from training_utils import (
    calculate_sparse_fingerprints,
    build_fingerprint_vocabulary,
    build_fingerprint_index,
    vectorize_fingerprints,
    normalize_hyperparam_range,
    to_rdkit_fingerprint,
)

MODEL_PATH = "ccsbase2_classifier.joblib"
FP_VOCAB_PATH = "ccsbase2_classifier_fp_vocab.joblib"
TRAIN_FINGERPRINTS_PATH = "ccsbase2_classifier_train_fingerprints.joblib"
ENCODER_PATH = "ccsbase2_classifier_encoder.joblib"


def _tag_predicted(values):
    return np.array([f"{v} (predicted)" if v is not None else None for v in values], dtype=object)


class SubclassClassifier:

    def __init__(self, database_file, seed, fp_min_count=5, n_iter=20, min_subclass_count=30,
                 novelty_threshold=0.7,
                 n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9):
        self.database_file = database_file
        self.seed = seed
        self.fp_min_count = fp_min_count
        self.n_iter = n_iter
        self.min_subclass_count = min_subclass_count
        self.novelty_threshold = novelty_threshold
        self.model = None
        self.encoder = None
        self.fp_vocab = None
        self.train_fingerprints = None
        self.cv_metrics = None
        self.proxy_ood_fingerprints = None

        self.hyperparam_ranges = {
            "n_estimators": normalize_hyperparam_range(n_estimators),
            "max_depth": normalize_hyperparam_range(max_depth),
            "learning_rate": normalize_hyperparam_range(learning_rate),
            "subsample": normalize_hyperparam_range(subsample),
            "colsample_bytree": normalize_hyperparam_range(colsample_bytree),
        }

        hierarchy = Database(database_file).read_df(
            "SELECT DISTINCT subclass, class, superclass FROM master_clean "
            "WHERE subclass NOT LIKE '%(predicted)' AND subclass IS NOT NULL"
        )
        self.subclass_to_class = dict(zip(hierarchy["subclass"], hierarchy["class"]))
        self.subclass_to_superclass = dict(zip(hierarchy["subclass"], hierarchy["superclass"]))

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
        self.proxy_ood_fingerprints = [to_rdkit_fingerprint(fp) for fp in proxy_ood_fps]

        # raw, non-vocabulary-restricted fingerprints -- the novelty gate needs to see rare
        # substructures that vocabulary filtering would otherwise zero out
        self.train_fingerprints = [to_rdkit_fingerprint(fp) for fp in train_fps]

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
        joblib.dump(self.train_fingerprints, TRAIN_FINGERPRINTS_PATH)
        joblib.dump(self.encoder, ENCODER_PATH)

    def eval(self):
        if len(self.proxy_ood_fingerprints) == 0:
            print("No proxy OOD rows to evaluate.")
            return

        nn_similarity = np.array([
            max(DataStructs.BulkTanimotoSimilarity(fp, self.train_fingerprints))
            for fp in self.proxy_ood_fingerprints
        ])
        is_novel = nn_similarity < self.novelty_threshold
        print(
            f"Proxy OOD set: {len(self.proxy_ood_fingerprints)} molecules from classes never included in training "
            f"(< {self.min_subclass_count} members)"
        )
        print(f"Flagged as novel by the gate: {is_novel.sum()} ({is_novel.mean() * 100:.1f}%)")

    def predict(self):
        print("Starting Inference")

        model = joblib.load(MODEL_PATH)
        fp_vocab = joblib.load(FP_VOCAB_PATH)
        train_fingerprints = joblib.load(TRAIN_FINGERPRINTS_PATH)
        encoder = joblib.load(ENCODER_PATH)

        db = Database(self.database_file)
        data = db.read_df(
            "SELECT id, smi, class, superclass FROM master_clean "
            "WHERE subclass LIKE '%(predicted)' OR subclass IS NULL"
        )

        if len(data) == 0:
            print("No unlabeled rows.")
            return

        fp_index = build_fingerprint_index(fp_vocab)
        fps = calculate_sparse_fingerprints(data["smi"])
        X_test = vectorize_fingerprints(fps, fp_index)

        nn_similarity = np.array([
            max(DataStructs.BulkTanimotoSimilarity(to_rdkit_fingerprint(fp), train_fingerprints))
            for fp in fps
        ])
        is_novel = nn_similarity < self.novelty_threshold

        proba = model.predict_proba(X_test)
        pred_idx = np.argmax(proba, axis=1)
        pred_class = encoder.inverse_transform(pred_idx)

        predicted_subclass = np.where(is_novel, None, pred_class)
        looked_up_class = np.array([self.subclass_to_class.get(c) for c in predicted_subclass], dtype=object)
        looked_up_superclass = np.array([self.subclass_to_superclass.get(c) for c in predicted_subclass], dtype=object)

        predicted_subclass = _tag_predicted(predicted_subclass)
        looked_up_class = _tag_predicted(looked_up_class)
        looked_up_superclass = _tag_predicted(looked_up_superclass)

        final_class = np.where(data["class"].isna(), looked_up_class, data["class"])
        final_superclass = np.where(data["superclass"].isna(), looked_up_superclass, data["superclass"])

        updates = list(zip(predicted_subclass, final_class, final_superclass, data["id"].astype(int)))
        db.write_many(
            "UPDATE master_clean SET subclass = ?, class = ?, superclass = ? WHERE id = ?", updates
        )

        print(f"Updated {len(updates)} rows in master_clean.subclass/class/superclass")
        print(f"  Flagged NONE via novelty gate: {is_novel.sum()}")


classifier = SubclassClassifier(
    "CCSMLDatabase.db", seed=26, fp_min_count=5,
    n_iter=1, min_subclass_count=30, novelty_threshold=0.7,
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
)

classifier.fit()
classifier.eval()
classifier.predict()
