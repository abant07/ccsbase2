import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from db import Database
from training_utils import (
    calculate_sparse_fingerprints,
    build_fingerprint_vocabulary,
    build_fingerprint_index,
    vectorize_fingerprints,
)
from metrics import macro_metrics, softmax_entropy


class SubclassClassifier:

    def __init__(self, database_file, min_class_count, seed, fp_min_count=5,
                 n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9):
        self.database_file = database_file
        self.min_class_count = min_class_count
        self.seed = seed
        self.fp_min_count = fp_min_count
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.model = None
        self.encoder = None
        self.entropy_threshold = None
        self.fp_vocab = None

    def fit(self):
        db = Database(self.database_file)
        data = db.read_df(
            "SELECT smi, subclass FROM master_clean WHERE subclass NOT LIKE '%(predicted)' AND subclass IS NOT NULL"
        )
        data = data.drop_duplicates(subset=["smi"], keep="first")

        data["subclass"] = data["subclass"].astype(str)
        class_counts = data["subclass"].value_counts()
        rare_classes = set(class_counts[class_counts < self.min_class_count].index)

        proxy_ood = data[data["subclass"].isin(rare_classes)].copy()
        train_df = data[~data["subclass"].isin(rare_classes)].copy()

        print(f"Rare-subclass rows held out: {len(proxy_ood)} (<{self.min_class_count}/subclass)")
        print(f"Training rows: {len(train_df)} across {train_df['subclass'].nunique()} classes")

        train_fps = calculate_sparse_fingerprints(train_df["smi"])
        self.fp_vocab = build_fingerprint_vocabulary(train_fps, self.fp_min_count)
        fp_index = build_fingerprint_index(self.fp_vocab)

        print(f"Fingerprint vocabulary: {len(self.fp_vocab)} substructures kept (min_count={self.fp_min_count})")

        X_train = vectorize_fingerprints(train_fps, fp_index)

        self.encoder = LabelEncoder()
        y_train = self.encoder.fit_transform(train_df["subclass"].astype(str).values)

        proxy_fps = calculate_sparse_fingerprints(proxy_ood["smi"])
        X_proxy_ood = vectorize_fingerprints(proxy_fps, fp_index)

        def build_classifier():
            return XGBClassifier(
                objective="multi:softprob",
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                tree_method="hist",
                n_jobs=-1,
                eval_metric="mlogloss",
                random_state=self.seed,
            )

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        n_classes = len(self.encoder.classes_)

        oof_proba = np.zeros((len(y_train), n_classes), dtype=np.float32)
        fold_acc, fold_p, fold_r, fold_f1 = [], [], [], []

        for fold, (train, test) in enumerate(kfold.split(X_train, y_train), 1):
            model = XGBClassifier(
                objective="multi:softprob",
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                tree_method="hist",
                n_jobs=-1,
                eval_metric="mlogloss",
                random_state=self.seed,
            )
            model.fit(X_train[train], y_train[train])

            proba = model.predict_proba(X_train[test])
            oof_proba[test] = proba
            y_pred = np.argmax(proba, axis=1)

            acc = accuracy_score(y_train[test], y_pred)
            p, r, f1 = macro_metrics(y_train[test], y_pred)

            fold_acc.append(acc)
            fold_p.append(p)
            fold_r.append(r)
            fold_f1.append(f1)

            print(f"Fold {fold} | acc={acc:.4f} | P={p:.4f} | R={r:.4f} | F1={f1:.4f}")

        print(
            "CV mean | "
            f"acc={np.mean(fold_acc):.4f} | "
            f"P={np.mean(fold_p):.4f} | "
            f"R={np.mean(fold_r):.4f} | "
            f"F1={np.mean(fold_f1):.4f}"
        )

        id_entropy = softmax_entropy(oof_proba)
        target_false_reject = 0.02
        self.entropy_threshold = float(np.quantile(id_entropy, 1.0 - target_false_reject))
        print(f"Entropy threshold (reject ~{target_false_reject * 100:.1f}% ID): {self.entropy_threshold:.6f}")

        self.model = build_classifier()
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, "ccsbase2_classifier.joblib")
        joblib.dump(self.fp_vocab, "ccsbase2_classifier_fp_vocab.joblib")

        if len(X_proxy_ood) > 0:
            proxy_entropy = softmax_entropy(self.model.predict_proba(X_proxy_ood))
            print(f"Proxy-OOD rejected: {(proxy_entropy > self.entropy_threshold).mean() * 100:.1f}%")

    def predict(self):
        print("Starting Inference")

        db = Database(self.database_file)
        data = db.read_df(
            "SELECT id, smi FROM master_clean WHERE subclass LIKE '%(predicted)' OR subclass IS NULL"
        )

        if len(data) == 0:
            print("No unlabeled rows.")
            return

        fp_index = build_fingerprint_index(self.fp_vocab)
        fps = calculate_sparse_fingerprints(data["smi"])
        X_test = vectorize_fingerprints(fps, fp_index)

        proba = self.model.predict_proba(X_test)
        entropy = softmax_entropy(proba)
        pred_idx = np.argmax(proba, axis=1)
        pred_class = self.encoder.inverse_transform(pred_idx)

        is_ood = entropy > self.entropy_threshold
        predicted_class = np.where(is_ood, "NONE", pred_class)
        predicted_class = np.array([f"{c} (predicted)" for c in predicted_class], dtype=object)

        updates = list(zip(predicted_class, data["id"].astype(int)))
        db.write_many("UPDATE master_clean SET subclass = ? WHERE id = ?", updates)

        print(f"Updated {len(updates)} rows in master_clean.subclass")
