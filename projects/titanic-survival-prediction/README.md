# Titanic Survival Prediction

## Type

Replication project.

## Objective

Given a Titanic passenger's attributes (class, sex, age, fare, family
composition, embarkation port), predict whether they survived, and identify
which factors are most associated with survival. This reproduces the
well-known Kaggle "Titanic: Machine Learning from Disaster" benchmark, and
served as the first project through this repository's structure and
conventions.

## Dataset

- **Source:** [datasciencedojo/datasets](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv),
  a public mirror of the Kaggle Titanic competition's `train.csv`. Originally
  compiled from the Titanic passenger manifest by Encyclopedia Titanica.
- **License:** Distributed for educational/non-commercial use, per Kaggle's
  Titanic competition rules.
- **Accessed:** 2026-08-02.
- **Description:** 891 labeled passengers, 12 raw columns. See
  `data/raw/titanic.csv`.

| Variable | Type | Description |
|---|---|---|
| PassengerId | Integer | Row identifier (not predictive) |
| Survived | Binary (0/1) | Target: 0 = did not survive, 1 = survived |
| Pclass | Ordinal (1/2/3) | Ticket class, a proxy for socio-economic status |
| Name | Text | Full name incl. title — source of the engineered `Title` feature |
| Sex | Categorical | male / female |
| Age | Float | ~20% missing; imputed |
| SibSp | Integer | Siblings/spouses aboard — used to derive `FamilySize` |
| Parch | Integer | Parents/children aboard — used to derive `FamilySize` |
| Ticket | Text | Ticket number; not used |
| Fare | Float | Fare paid |
| Cabin | Text | ~77% missing; reduced to a `HasCabin` flag |
| Embarked | Categorical | C / Q / S; 2 missing values, imputed with the mode |

## Methodology

1. **Clean** (`src/clean.py`): extract `Title` from `Name`, consolidate rare
   titles; impute `Age` by the median within each Title/Pclass group;
   impute `Embarked` with the mode and `Fare` with the Pclass median;
   reduce `Cabin` to a `HasCabin` presence flag.
2. **Engineer features** (`src/features.py`): derive `FamilySize`
   (`SibSp + Parch + 1`) and `IsAlone`; one-hot encode `Sex`, `Embarked`,
   `Title`.
3. **Explore** (`notebooks/01_eda.ipynb`): survival rate by sex, class, age
   band, fare, family size, and embarkation port.
4. **Train** (`src/train.py`): stratified 5-fold cross-validation over three
   candidate models, then a held-out evaluation of the best one.
5. **Evaluate** (`src/evaluate.py`): prints the metrics summary and saves a
   confusion matrix and ROC curve for the selected model as SVGs.

## Installation

```bash
pip install -r requirements.txt
```

## Reproducing the results

```bash
python src/clean.py      # data/raw/titanic.csv -> data/processed/cleaned.csv
python src/features.py   # -> data/processed/features.csv
python src/train.py      # -> results/model.joblib, results/metrics.json
python src/evaluate.py   # -> results/figures/confusion_matrix.svg, roc_curve.svg
```

All splits and models use `random_state = 42`, recorded in
`results/metrics.json`. `results/model.joblib` is a local build artifact
(the fitted `GradientBoostingClassifier`) — it's small to regenerate via
`src/train.py` and is not committed to the repository; `results/metrics.json`
and the figures below are the committed record of its performance.

## Results

Stratified 5-fold cross-validation accuracy on the full labeled set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.829 ± 0.008 | 0.790 ± 0.010 | 0.757 ± 0.031 | 0.773 ± 0.016 | 0.873 ± 0.021 |
| Random Forest | 0.828 ± 0.015 | 0.793 ± 0.024 | 0.749 ± 0.024 | 0.770 ± 0.020 | 0.878 ± 0.023 |
| **Gradient Boosting (selected)** | **0.836 ± 0.020** | 0.821 ± 0.035 | 0.734 ± 0.032 | 0.775 ± 0.028 | 0.884 ± 0.012 |

Full per-model CV metrics are in `results/metrics.json`.

**Held-out evaluation** (20% stratified split, unseen by the selected model):

| Metric | Value |
|---|---|
| Accuracy | 0.816 |
| Precision | 0.781 |
| Recall | 0.725 |
| F1 | 0.752 |
| ROC-AUC | 0.845 |

This lands in the target range from the project plan (~78–82% accuracy) and
is consistent with typical non-overfit scores on the public Titanic
leaderboard. See [`results/figures/confusion_matrix.svg`](results/figures/confusion_matrix.svg)
and [`roc_curve.svg`](results/figures/roc_curve.svg) for the held-out
diagnostics, and `notebooks/01_eda.ipynb` for the exploratory analysis
behind the feature choices.

Sex was the strongest single signal (women survived at a much higher rate
than men), followed by class/fare; children survived at a higher rate than
adults, and very large families (5+) survived at a lower rate than small
ones.

## Limitations

- Small dataset (891 labeled rows) limits model complexity and the
  reliability of subgroup conclusions.
- Reflects one specific historical event (evacuation procedures, 1912 social
  norms) — findings do not generalize beyond it.
- `Title` partially overlaps with information already in `Pclass`/`Sex`/`Fare`;
  it was kept because it measurably improved cross-validated accuracy over
  dropping it.
- As a replication project, the goal was a solid, documented reproduction of
  a known result, not a novel finding.

## Conclusions

A gradient boosting classifier on engineered features reaches ~84% mean
cross-validated accuracy and ~0.85 held-out ROC-AUC, in line with public
Titanic benchmarks. The exercise validated this repository's project
structure, templates, and reproducibility conventions end to end and is
ready to serve as the template for future replication and original-analysis
projects here.
