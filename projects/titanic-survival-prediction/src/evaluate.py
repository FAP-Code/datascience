"""Print a metrics summary and save diagnostic plots for the selected model."""
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

FEATURES_PATH = "data/processed/features.csv"
METRICS_PATH = "results/metrics.json"
MODEL_PATH = "results/model.joblib"
FIGURES_DIR = "results/figures"
RANDOM_STATE = 42


def print_summary(metrics: dict):
    print("Cross-validated accuracy (5-fold, mean +/- std):")
    for name, scores in metrics["cross_validation"].items():
        acc = scores["accuracy"]
        print(f"  {name:20s} {acc['mean']:.3f} +/- {acc['std']:.3f}")

    best = metrics["best_model"]
    print(f"\nSelected model: {best}")
    print("Held-out evaluation (20% stratified split):")
    for metric, value in metrics["holdout_metrics"].items():
        print(f"  {metric:10s} {value:.3f}")


def save_diagnostic_plots(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Did not survive", "Survived"], ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix (held-out set)")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/confusion_matrix.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve (held-out set)")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/roc_curve.svg")
    plt.close(fig)


def main():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    print_summary(metrics)

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    # Same split as train.py, recreated from the recorded random_state, so this
    # is the same held-out slice the saved model was scored against (and never trained on).
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=metrics["random_state"]
    )
    save_diagnostic_plots(model, X_test, y_test)
    print(f"\nSaved diagnostic plots to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
