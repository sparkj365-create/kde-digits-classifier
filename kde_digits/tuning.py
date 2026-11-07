"""Pipeline construction and hyper-parameter search."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .model import KDEClassifier


def build_pipeline(use_pca: bool = True) -> Pipeline:
    steps = []
    if use_pca:
        steps.append(("pca", PCA(random_state=0)))
    steps.append(("kde", KDEClassifier()))
    return Pipeline(steps)


def make_param_grid(use_pca: bool = True, n_bandwidths: int = 15) -> dict:
    grid = {"kde__bandwidth": np.geomspace(0.03, 0.8, n_bandwidths)}
    if use_pca:
        grid["pca__n_components"] = [10, 15, 20, 30]
    return grid


def tune(X, y, use_pca: bool = True, n_bandwidths: int = 15, cv_splits: int = 5,
         seed: int = 42, n_jobs: int = -1) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    search = GridSearchCV(
        build_pipeline(use_pca),
        make_param_grid(use_pca, n_bandwidths),
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        return_train_score=True,
    )
    return search.fit(X, y)
