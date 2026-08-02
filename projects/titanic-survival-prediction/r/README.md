# R Implementation

An R reimplementation of the same pipeline as the project's Python version
(`../src/`): clean → engineer features → cross-validate three models → select
and evaluate the best one on a held-out split. Same dataset
(`../data/raw/titanic.csv`), same random seed (42), same methodology —
independent implementation, not a port of the Python code line-by-line.

## Design differences from the Python version

- **No manual one-hot encoding.** `features.R` keeps `Sex`, `Embarked`,
  `Title`, and `Pclass` as factors instead of dummy-encoding them: R's
  modelling functions (`glm`, `randomForest`, `gbm`) encode factor predictors
  natively, so a manual encoding step would just duplicate what the model
  formula already does.
- **Manual stratified cross-validation**, mirroring the Python version's
  approach rather than delegating to `caret::train()`'s built-in resampling,
  so per-fold accuracy/precision/recall/F1/ROC-AUC are computed the same way
  in both languages.
- Model fitting: `glm(family = binomial)` for logistic regression,
  `randomForest` for random forest, `gbm` (Bernoulli loss) for gradient
  boosting.

## Installation

```r
source("install_packages.R")
```

## Reproducing the results

Run from inside this `r/` directory:

```bash
Rscript src/clean.R      # ../data/raw/titanic.csv -> data/processed/cleaned.csv
Rscript src/features.R   # -> data/processed/features.csv
Rscript src/train.R      # -> results/model.rds, results/metrics.json
Rscript src/evaluate.R   # -> results/figures/confusion_matrix.svg, roc_curve.svg
```

`results/model.rds` is a local build artifact (not committed, see
`.gitignore`) — regenerate it with `src/train.R`.

## Results

5-fold cross-validation accuracy:

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 82.8% ± 4.0% | 79.1% ± 5.8% | 75.5% ± 8.4% | 0.869 ± 0.036 |
| **Random Forest (selected)** | **83.1% ± 3.3%** | 80.7% ± 6.4% | 74.0% ± 4.5% | 0.880 ± 0.035 |
| Gradient Boosting | 82.4% ± 3.3% | 80.4% ± 6.2% | 72.0% ± 5.9% | 0.881 ± 0.035 |

Held-out evaluation (20% stratified split):

| Metric | Value |
|---|---|
| Accuracy | 82.5% |
| Precision | 86.3% |
| Recall | 64.7% |
| F1 | 73.9% |
| ROC-AUC | 0.874 |

**Random Forest** won here, versus **Gradient Boosting** in the Python
version. This is an expected, honest difference, not a bug: the two runs use
independent RNG streams (R's and NumPy's random number generators don't
produce comparable sequences from the same seed) and each language's default
hyperparameters differ slightly between `sklearn` and R's `randomForest`/
`gbm`. All three models land within about 1 point of each other in both
languages, consistent with the Python finding that the data — not model
choice — is the binding constraint here.

## Verification

Both the cleaned dataset and this pipeline were checked against the Python
version before being committed: `clean.R`'s output matched `clean.py`'s
exactly across all 891 rows (Title, Age, Embarked, Fare, HasCabin — zero
mismatches), and the full pipeline was run end to end from a clean state to
confirm it's reproducible.
