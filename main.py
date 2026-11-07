"""Command-line entry point: tune, evaluate, plot and save the KDE classifier."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

from kde_digits.data import load_data
from kde_digits.evaluation import compare_with_baselines, evaluate, most_confident_errors
from kde_digits.tuning import tune


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-pca", action="store_true", help="disable the PCA step")
    p.add_argument("--bandwidths", type=int, default=15, help="grid size for bandwidth")
    p.add_argument("--cv", type=int, default=5, help="number of CV folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("outputs"), help="output directory")
    p.add_argument("--no-baselines", action="store_true", help="skip baseline comparison")
    p.add_argument("--show", action="store_true", help="also display figures on screen")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    from kde_digits import plots  # import after backend selection

    data = load_data(seed=args.seed)

    print("Tuning ...")
    search = tune(data.X_train, data.y_train, use_pca=not args.no_pca,
                  n_bandwidths=args.bandwidths, cv_splits=args.cv, seed=args.seed)
    best = search.best_estimator_
    print("Best parameters :", search.best_params_)
    print(f"CV accuracy     : {search.best_score_:.4f}")

    metrics = evaluate(best, data.X_test, data.y_test)
    print(f"Test accuracy   : {metrics['accuracy']:.4f}")
    print(f"Test log-loss   : {metrics['log_loss']:.4f}")
    print("\n" + metrics["report"])

    args.out.mkdir(parents=True, exist_ok=True)
    plots.plot_bandwidth_curve(search, args.out, args.show)
    plots.plot_confusion_matrix(data.y_test, metrics["y_pred"], data.target_names,
                                args.out, args.show)
    plots.plot_predictions(data.X_test, data.y_test, metrics["y_pred"], range(10),
                           "First 10 test predictions", "predictions.png",
                           args.out, args.show)
    errors = most_confident_errors(best, data.X_test, data.y_test)
    if len(errors):
        plots.plot_predictions(data.X_test, data.y_test, metrics["y_pred"], errors,
                               "Most confident mistakes", "errors.png",
                               args.out, args.show)

    if not args.no_baselines:
        print("Comparing with baselines ...")
        table = compare_with_baselines(best, data.X_train, data.y_train, args.seed)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        table.to_csv(args.out / "baselines.csv", index=False)
        plots.plot_baselines(table, args.out, args.show)

    joblib.dump(best, args.out / "kde_model.joblib")
    print(f"\nSaved figures and model to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
