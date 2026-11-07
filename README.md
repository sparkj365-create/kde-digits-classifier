# kde-digits

A Bayesian **Kernel Density Estimation classifier** for handwritten digits, built as a clean, tested, scikit-learn-compatible package.

For each class the model estimates `p(x | class)` with a Gaussian KDE and classifies with Bayes' rule: `p(class | x) ∝ p(x | class) · p(class)`.

## Results (sklearn digits, 80/20 split)

| Model | 5-fold CV accuracy |
|---|---|
| **KDE (PCA + tuned bandwidth)** | **0.988** |
| kNN (k=3) | 0.987 |
| Random Forest | 0.976 |
| Logistic Regression | 0.965 |
| Gaussian NB | 0.918 |

Held-out test accuracy: **0.983**, log-loss: **0.063**.
(The KDE row is slightly optimistic because its hyper-parameters were chosen on the same folds; the test accuracy is the unbiased figure.)

## Project structure

```
kde_digits/
├── kde_digits/
│   ├── model.py        # KDEClassifier (predict, predict_proba, predict_log_proba)
│   ├── data.py         # loading, scaling, stratified split
│   ├── tuning.py       # PCA + KDE pipeline and GridSearchCV
│   ├── evaluation.py   # metrics, baseline comparison, most-confident errors
│   └── plots.py        # bandwidth curve, confusion matrix, predictions, baselines
├── tests/test_model.py # unit tests + scikit-learn estimator API checks
├── main.py             # CLI entry point
└── .github/workflows/ci.yml
```

## Usage

```bash
pip install -r requirements.txt
python main.py                 # tune, evaluate, save figures + model to ./outputs
python main.py --no-pca        # plain KDE on raw pixels
python main.py --show          # also display figures
python main.py --help
python -m pytest -q tests
```

```python
from kde_digits import KDEClassifier

clf = KDEClassifier(bandwidth=0.3).fit(X_train, y_train)
clf.predict_proba(X_test)      # calibrated class probabilities
```

## What changed vs. the original script

- Split into modules instead of one file
- `predict_proba` / `predict_log_proba` computed stably with `logsumexp`
- Full scikit-learn API compliance (input validation, `NotFittedError`, `n_features_in_`), verified by `sklearn`'s own estimator checks
- PCA step tuned jointly with bandwidth (KDE struggles in 64 dimensions)
- Baseline comparison, log-loss, and a plot of the most confident mistakes
- CLI options, figures saved to disk (works headless), model saved with joblib
- Tests and CI

## License

MIT
