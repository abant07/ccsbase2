import sqlite3
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier

from utils import Utils

class SubclassClassifier:

    def __init__(self, database_file, min_class_count, seed):
        self.database_file = database_file
        self.min_class_count = min_class_count
        self.seed = seed
        self.model = None
        self.encoder = None
        self.entropy_threshold = None
    
    def _build_model(self):
        return XGBClassifier(
            objective="multi:softprob",
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="mlogloss",
            random_state=self.seed, 
        )
    
    def _macro_metrics(self, y_true, y_pred):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        return float(p), float(r), float(f1)

    def _softmax_entropy(self, proba, eps=1e-12):
        p = np.clip(proba, eps, 1.0)
        return -np.sum(p * np.log(p), axis=1)
    
    def fit(self):
        conn = sqlite3.connect(self.database_file)
        query = "SELECT smi, subclass FROM master_clean WHERE subclass IS NOT NULL"
        data = pd.read_sql_query(query, conn)
        conn.close()

        data["subclass"] = data["subclass"].astype(str)
        class_counts = data["subclass"].value_counts()
        rare_classes = set(class_counts[class_counts < self.min_class_count].index)

        proxy_ood = data[data["subclass"].isin(rare_classes)].copy()
        train_df = data[~data["subclass"].isin(rare_classes)].copy()

        print(f"Rare-subclass rows held out: {len(proxy_ood)} (<{self.min_class_count}/subclass)")
        print(f"Training rows: {len(train_df)} across {train_df['subclass'].nunique()} classes")

        X_train = []
        for _, row in train_df.iterrows():
            features = Utils().calculate_classifier_descriptors(row['smi'])
            X_train.append(features)
        X_train = np.array(X_train)
        
        self.encoder = LabelEncoder()
        y_train = self.encoder.fit_transform(train_df["subclass"].astype(str).values)

        X_proxy_ood = []
        for _, row in proxy_ood.iterrows():
            features = Utils().calculate_classifier_descriptors(row['smi'])
            X_proxy_ood.append(features)
        X_proxy_ood = np.array(X_proxy_ood)
        
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        n_classes = len(self.encoder.classes_)

        oof_proba = np.zeros((len(y_train), n_classes), dtype=np.float32)
        fold_acc, fold_p, fold_r, fold_f1 = [], [], [], []

        for fold, (train, test) in enumerate(kfold.split(X_train, y_train), 1):
            model = self._build_model()
            model.fit(X_train[train], y_train[train])

            proba = model.predict_proba(X_train[test])
            oof_proba[test] = proba
            y_pred = np.argmax(proba, axis=1)

            acc = accuracy_score(y_train[test], y_pred)
            p, r, f1 = self._macro_metrics(y_train[test], y_pred)

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

        id_entropy = self._softmax_entropy(oof_proba)
        target_false_reject = 0.015
        self.entropy_threshold = float(np.quantile(id_entropy, 1.0 - target_false_reject))
        print(f"Entropy threshold (reject ~{target_false_reject*100:.1f}% ID): {self.entropy_threshold:.6f}")

        self.model = self._build_model()
        self.model.fit(X_train, y_train)

        if X_proxy_ood is not None and len(X_proxy_ood) > 0:
            proxy_entropy = self._softmax_entropy(self.model.predict_proba(X_proxy_ood))
            print(f"Proxy-OOD rejected: {(proxy_entropy > self.entropy_threshold).mean()*100:.1f}%")
        
    def predict(self):
        print ("Starting Inference")

        conn = sqlite3.connect(self.database_file)
        query = "SELECT id, smi FROM master_clean WHERE subclass IS NULL"
        data = pd.read_sql_query(query, conn)

        if len(data) == 0:
            conn.close()
            print("No unlabeled rows.")
            return


        X_test, keep_ids = [], []
        for _, row in data.iterrows():
            features = Utils().calculate_classifier_descriptors(row['smi'])
            X_test.append(features)
            keep_ids.append(int(row["id"]))
        X_test = np.array(X_test, dtype=np.float32)

        proba = self.model.predict_proba(X_test)
        entropy = self._softmax_entropy(proba)
        pred_idx = np.argmax(proba, axis=1)
        pred_class = self.encoder.inverse_transform(pred_idx)

        proba_unl = self.model.predict_proba(X_test)
        pred_idx = np.argmax(proba_unl, axis=1)
        pred_class = self.encoder.inverse_transform(pred_idx)

        is_ood = entropy > self.entropy_threshold
        predicted_class = np.where(is_ood, "NONE", pred_class)
        predicted_class = np.array([f"{c} (predicted)" for c in predicted_class], dtype=object)

        updates = [(predicted_class[i], keep_ids[i]) for i in range(len(keep_ids))]
        conn.executemany("UPDATE master_clean SET subclass = ? WHERE id = ?", updates)
        conn.commit()
        conn.close()

        print(f"Updated {len(updates)} rows in master_clean.subclass")
