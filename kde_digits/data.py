"""Dataset loading and splitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass
class DigitsData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    target_names: np.ndarray


def load_data(test_size: float = 0.2, seed: int = 42) -> DigitsData:
    """Load the sklearn digits dataset, scale pixels to [0, 1], split stratified."""
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return DigitsData(X_train, X_test, y_train, y_test, digits.target_names)
