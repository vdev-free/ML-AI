import os

import mlflow
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

os.makedirs("artifacts/day84", exist_ok=True)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("day84-mlflow-intro")

# print("Tracking URI:", mlflow.get_tracking_uri())

def run_once(n_estimators: int, max_depth: int | None, seed: int = 42):
    data = load_breast_cancer()
    X = data["data"]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    return float(auc), proba

with mlflow.start_run(run_name="rf-baseline"):
    n_estimators = 200
    max_depth = None

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    auc, proba = run_once(n_estimators=n_estimators, max_depth=max_depth)
    mlflow.log_metric("test_auc", auc)

    plt.figure(figsize=(7, 4))
    plt.hist(proba, bins=20)
    plt.tight_layout()

    plot_path = "artifacts/day84/proba_hist_rf-baseline.png"
    plt.savefig(plot_path)
    plt.close()

    mlflow.log_artifact(plot_path)
    print("Saved artifact:", plot_path)


    print("Logged run AUC:", round(auc, 4))
