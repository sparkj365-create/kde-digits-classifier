"""All plotting helpers. Each function saves to `out_dir` when given."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _finish(fig, out_dir: Path | None, name: str, show: bool) -> None:
    fig.tight_layout()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / name, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_bandwidth_curve(search, out_dir=None, show=False) -> None:
    """CV accuracy vs bandwidth (best result over the other parameters)."""
    res = search.cv_results_
    bw = np.array(res["param_kde__bandwidth"].data, dtype=float)
    mean, std = res["mean_test_score"], res["std_test_score"]
    train = res["mean_train_score"]

    fig, ax = plt.subplots(figsize=(9, 5))
    uniq = np.unique(bw)
    best_idx = [np.where(bw == b)[0][np.argmax(mean[bw == b])] for b in uniq]
    ax.semilogx(uniq, mean[best_idx], "o-", label="CV accuracy")
    ax.semilogx(uniq, train[best_idx], "s--", alpha=0.6, label="Train accuracy")
    ax.fill_between(uniq, mean[best_idx] - std[best_idx],
                    mean[best_idx] + std[best_idx], alpha=0.2)
    ax.axvline(search.best_params_["kde__bandwidth"], ls=":", color="k",
               label="Best bandwidth")
    ax.set(xlabel="Bandwidth", ylabel="Accuracy", title="KDE performance vs bandwidth")
    ax.legend()
    ax.grid(alpha=0.3)
    _finish(fig, Path(out_dir) if out_dir else None, "bandwidth_curve.png", show)


def plot_confusion_matrix(y_true, y_pred, labels, out_dir=None, show=False) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("KDE confusion matrix")
    _finish(fig, Path(out_dir) if out_dir else None, "confusion_matrix.png", show)


def plot_predictions(X, y_true, y_pred, indices, title, fname,
                     out_dir=None, show=False) -> None:
    n = len(indices)
    cols = min(5, max(n, 1))
    rows = int(np.ceil(max(n, 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), indices):
        ax.imshow(X[i].reshape(8, 8), cmap="binary")
        ax.set_title(f"T:{y_true[i]} P:{y_pred[i]}",
                     color="green" if y_true[i] == y_pred[i] else "red")
    fig.suptitle(title)
    _finish(fig, Path(out_dir) if out_dir else None, fname, show)


def plot_baselines(df, out_dir=None, show=False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["model"][::-1], df["cv_mean"][::-1], xerr=df["cv_std"][::-1])
    ax.set(xlabel="CV accuracy", title="KDE vs baselines",
           xlim=(max(0.0, df["cv_mean"].min() - 0.05), 1.0))
    ax.grid(axis="x", alpha=0.3)
    _finish(fig, Path(out_dir) if out_dir else None, "baselines.png", show)
