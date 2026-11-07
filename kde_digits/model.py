"""Bayesian classifier built on per-class Kernel Density Estimates."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data
except ImportError:  # older versions
    def validate_data(estimator, *args, **kwargs):
        return estimator._validate_data(*args, **kwargs)


class KDEClassifier(ClassifierMixin, BaseEstimator):
    """Generative classifier: p(class | x) ∝ p(x | class) * p(class).

    p(x | class) is estimated with a KernelDensity model per class.

    Parameters
    ----------
    bandwidth : float
        Kernel bandwidth shared by all classes.
    kernel : str
        Any kernel supported by ``sklearn.neighbors.KernelDensity``.
    """

    def __init__(self, bandwidth: float = 0.1, kernel: str = "gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)

        self.models_ = []
        log_priors = []
        for label in self.classes_:
            X_class = X[y == label]
            model = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            model.fit(X_class)
            self.models_.append(model)
            log_priors.append(np.log(len(X_class) / len(X)))
        self.log_priors_ = np.asarray(log_priors)
        return self

    def _joint_log_likelihood(self, X) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return np.column_stack(
            [m.score_samples(X) for m in self.models_]
        ) + self.log_priors_

    def predict_log_proba(self, X) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        jll = self._joint_log_likelihood(X)  # raises NotFittedError if unfitted
        return self.classes_[np.argmax(jll, axis=1)]
