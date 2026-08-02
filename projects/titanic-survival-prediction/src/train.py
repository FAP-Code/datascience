"""Cross-validate candidate models, select the best, and save it with its metrics."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

FEATURES_PATH = "data/processed/features.csv"
METRICS_PATH = "results/metrics.json"
MODEL_PATH = "results/model.joblib"
RANDOM_STATE = 42

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return X, y


def cross_validate_models(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in MODELS.items():
        scores = cross_validate(clone(model), X, y, cv=cv, scoring=SCORING)
        results[name] = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"])),
            }
            for metric in SCORING
        }
    return results


def evaluate_holdout(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def main():
    X, y = load_data()
    cv_results = cross_validate_models(X, y)

    best_model_name = max(cv_results, key=lambda name: cv_results[name]["accuracy"]["mean"])
    print(f"Best model by mean CV accuracy: {best_model_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    holdout_model = clone(MODELS[best_model_name])
    holdout_model.fit(X_train, y_train)
    holdout_metrics = evaluate_holdout(holdout_model, X_test, y_test)

    # Ship the holdout-trained model, not one refit on all data: evaluate.py
    # re-scores it against X_test, which would leak train data into the score
    # if the saved model had already seen it.
    joblib.dump(holdout_model, MODEL_PATH)

    output = {
        "random_state": RANDOM_STATE,
        "cross_validation": cv_results,
        "best_model": best_model_name,
        "holdout_metrics": holdout_metrics,
        "feature_columns": list(X.columns),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote metrics to {METRICS_PATH}")
    print(f"Saved final model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
