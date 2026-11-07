"""Model evaluation and baseline comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def evaluate(model, X_test, y_test) -> dict:
    """Return accuracy, log-loss and the text classification report."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
        "y_pred": y_pred,
    }
    if hasattr(model, "predict_proba"):
        proba = np.clip(model.predict_proba(X_test), 1e-12, 1.0)
        metrics["log_loss"] = log_loss(y_test, proba, labels=model.classes_)
    return metrics


def compare_with_baselines(best_model, X_train, y_train, seed: int = 42) -> pd.DataFrame:
    """5-fold CV accuracy of the tuned KDE vs. a few standard baselines."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    models = {
        "KDE (tuned)": best_model,
        "GaussianNB": GaussianNB(var_smoothing=1e-2),
        "kNN (k=3)": KNeighborsClassifier(n_neighbors=3),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed),
    }
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        rows.append({"model": name, "cv_mean": scores.mean(), "cv_std": scores.std()})
    return (pd.DataFrame(rows)
            .sort_values("cv_mean", ascending=False)
            .reset_index(drop=True))


def most_confident_errors(model, X_test, y_test, k: int = 10) -> np.ndarray:
    """Indices of the k misclassified samples the model was most sure about."""
    proba = model.predict_proba(X_test)
    pred = model.classes_[proba.argmax(axis=1)]
    wrong = np.where(pred != y_test)[0]
    order = np.argsort(-proba[wrong].max(axis=1))
    return wrong[order][:k]
