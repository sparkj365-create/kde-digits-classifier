import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import parametrize_with_checks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kde_digits.data import load_data
from kde_digits.evaluation import most_confident_errors
from kde_digits.model import KDEClassifier
from kde_digits.tuning import build_pipeline


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def fitted(data):
    return KDEClassifier(bandwidth=0.2).fit(data.X_train, data.y_train)


def test_proba_rows_sum_to_one(fitted, data):
    p = fitted.predict_proba(data.X_test)
    assert p.shape == (len(data.X_test), 10)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


def test_predict_matches_argmax_proba(fitted, data):
    p = fitted.predict_proba(data.X_test)
    assert np.array_equal(fitted.predict(data.X_test), fitted.classes_[p.argmax(1)])


def test_reasonable_accuracy(fitted, data):
    assert fitted.score(data.X_test, data.y_test) > 0.9


def test_pipeline_with_pca(data):
    pipe = build_pipeline(use_pca=True).set_params(pca__n_components=20,
                                                   kde__bandwidth=0.3)
    pipe.fit(data.X_train, data.y_train)
    assert pipe.score(data.X_test, data.y_test) > 0.9


def test_wrong_feature_count_raises(fitted):
    with pytest.raises(ValueError):
        fitted.predict(np.zeros((2, 5)))


def test_priors_follow_class_frequency():
    X = np.vstack([np.zeros((30, 2)), np.ones((10, 2))])
    y = np.array([0] * 30 + [1] * 10)
    m = KDEClassifier(bandwidth=0.5).fit(X, y)
    assert np.allclose(np.exp(m.log_priors_), [0.75, 0.25])


def test_clone_and_params():
    m = KDEClassifier(bandwidth=0.7, kernel="tophat")
    c = clone(m)
    assert c.get_params() == m.get_params()


def test_most_confident_errors_are_errors(fitted, data):
    idx = most_confident_errors(fitted, data.X_test, data.y_test, k=5)
    pred = fitted.predict(data.X_test)
    assert all(pred[i] != data.y_test[i] for i in idx)


@parametrize_with_checks([KDEClassifier()])
def test_sklearn_api(estimator, check):
    check(estimator)
