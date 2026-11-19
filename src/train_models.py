"""Train and evaluate Renal AI models from the frozen participant split."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FEATURES = [
    "RIDAGEYR", "female", "INDFMPIR", "LBXGH", "BMXBMI",
    "BPXOSY1", "BPXODI1", "smoking", "insulin",
]
TARGET = "kidney_marker"
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]


def make_preprocessor() -> ColumnTransformer:
    numeric = ["RIDAGEYR", "INDFMPIR", "LBXGH", "BMXBMI", "BPXOSY1", "BPXODI1"]
    binary = ["female", "insulin"]
    categorical = ["smoking"]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("binary", binary_pipe, binary),
        ("categorical", categorical_pipe, categorical),
    ])


def main() -> None:
    train = pd.read_csv(DATA_DIR / "model_train.csv")
    test = pd.read_csv(DATA_DIR / "model_test.csv")
    x_train, y_train = train[FEATURES], train[TARGET].astype(int)
    x_test, y_test = test[FEATURES], test[TARGET].astype(int)

    if set(train.SEQN).intersection(test.SEQN):
        raise ValueError("Participant IDs overlap between train and test data")
    if len(train) != 1204 or len(test) != 302:
        raise ValueError("Unexpected split size; expected 1,204 train and 302 test rows")

    shared = make_preprocessor()
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1),
    }
    params = {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 5, 10, 15],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__class_weight": [None, "balanced"],
        },
        "XGBoost": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [2, 3, 4, 5],
            "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "classifier__subsample": [0.7, 0.9, 1.0],
            "classifier__colsample_bytree": [0.7, 0.9, 1.0],
        },
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fitted = {}
    rows = []

    for name in MODEL_NAMES:
        pipeline = Pipeline([
            ("preprocessor", shared),
            ("classifier", estimators[name]),
        ])
        search = RandomizedSearchCV(
            pipeline,
            params[name],
            n_iter=10,
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            refit=True,
        )
        print(f"Tuning {name}…", flush=True)
        search.fit(x_train, y_train)
        model = search.best_estimator_
        fitted[name] = model
        probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        predictions = (probabilities >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "Specificity": tn / (tn + fp),
            "F1-score": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "PR-AUC": average_precision_score(y_test, probabilities),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "CV ROC-AUC": search.best_score_,
            "Best parameters": json.dumps(search.best_params_, sort_keys=True),
        })
        print(f"  held-out ROC-AUC={rows[-1]['ROC-AUC']:.4f}; recall={rows[-1]['Recall']:.4f}", flush=True)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted, MODEL_DIR / "all_models.joblib")
    joblib.dump(fitted["Random Forest"], MODEL_DIR / "random_forest_pipeline.joblib")
    joblib.dump(fitted["Logistic Regression"], MODEL_DIR / "logistic_regression_pipeline.joblib")
    joblib.dump(fitted["XGBoost"], MODEL_DIR / "xgboost_pipeline.joblib")
    pd.DataFrame(rows).to_csv(REPORT_DIR / "model_evaluation.csv", index=False)
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps({
        "models": MODEL_NAMES,
        "features": FEATURES,
        "target": TARGET,
        "threshold": 0.5,
        "threshold_note": "Default threshold; not separately tuned.",
        "train_rows": len(train),
        "test_rows": len(test),
        "random_seed": 42,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved evaluation and three pipelines under {ROOT}", flush=True)


if __name__ == "__main__":
    main()
