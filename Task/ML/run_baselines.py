"""Train the repository's SVM/RF/MLP baselines on locally prepared K3b data.

The search grids and accuracy objective mirror Task/ML's original notebooks.
Hyperparameters are selected with stratified 5-fold CV on training only.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Dataset/processed/k3b/label45"
OUT_DIR = DATA_DIR / "baselines"
POSITIVE_LABEL = 4


def load_split(name: str):
    frame = pd.read_csv(DATA_DIR / "ml_features" / f"{name}.csv")
    if "label" not in frame or frame.empty:
        raise ValueError(f"Missing labels or empty feature split: {name}")
    return frame.drop(columns="label").to_numpy(dtype=np.float64), frame["label"].to_numpy(dtype=int)


def metrics(model, x, y):
    pred = model.predict(x)
    classes = list(model.classes_)
    pos_idx = classes.index(POSITIVE_LABEL)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, pos_idx]
    else:
        decision = np.asarray(model.decision_function(x))
        score = decision if classes[1] == POSITIVE_LABEL else -decision
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_class4": float(precision_score(y, pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall_class4": float(recall_score(y, pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1_class4": float(f1_score(y, pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "roc_auc_class4": float(roc_auc_score(y == POSITIVE_LABEL, score)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    candidates = {
        "svm": (Pipeline([("scale", StandardScaler()), ("model", SVC())]), {
            "model__C": [0.1, 1, 10, 50, 80, 100, 150],
            "model__gamma": ["scale", 0.01, 0.1, 1],
            "model__kernel": ["linear", "rbf"], "model__class_weight": [None, "balanced"],
        }),
        "random_forest": (Pipeline([("scale", StandardScaler()),
                                     ("model", RandomForestClassifier(random_state=42, n_jobs=1))]), {
            "model__n_estimators": [100, 200, 300], "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5, 10], "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        }),
        "mlp": (Pipeline([("scale", StandardScaler()),
                          ("model", MLPClassifier(random_state=42, early_stopping=False))]), {
            "model__hidden_layer_sizes": [(100,), (50, 50), (30, 30, 30)],
            "model__activation": ["relu", "tanh"], "model__solver": ["adam", "sgd"],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate": ["constant", "adaptive"],
            "model__max_iter": [200, 300, 500],
        }),
    }
    results = {"feature_columns": list(pd.read_csv(DATA_DIR / "ml_features/train.csv", nrows=0).columns[:-1]),
               "cv": {"method": "StratifiedKFold", "n_splits": 5, "shuffle": False,
                      "scoring": "accuracy", "note": "parameter grids mirror the original notebooks"},
               "positive_label": 4, "models": {}}
    for name, (estimator, grid) in candidates.items():
        print(f"[{name}] grid search on training split ({len(y_train)} trials)", flush=True)
        search = GridSearchCV(estimator, grid, scoring="accuracy", cv=cv,
                              n_jobs=-1, refit=True, return_train_score=False, verbose=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            search.fit(x_train, y_train)
        # Save the refitted best estimator from training data. No validation/test refit.
        model = search.best_estimator_
        model_path = OUT_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        model_metrics = {"best_cv_accuracy": float(search.best_score_),
                         "best_params": search.best_params_, "validation": metrics(model, x_val, y_val),
                         "test": metrics(model, x_test, y_test), "model_file": model_path.name}
        results["models"][name] = model_metrics
        print(json.dumps({name: model_metrics}, indent=2), flush=True)
    result_path = OUT_DIR / "results.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results: {result_path}", flush=True)


if __name__ == "__main__":
    main()
