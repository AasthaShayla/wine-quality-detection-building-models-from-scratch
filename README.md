# Wine Quality Prediction - Decision Tree & Random Forest *from scratch*

Predicting the quality of red wine from its physicochemical properties, with the
**Decision Tree and Random Forest algorithms implemented from scratch in NumPy** -
no scikit-learn for the models. The goal is to *understand* how tree ensembles
actually work, not just call `.fit()`.

## Dataset

The UCI **red wine quality** dataset - 1,599 samples, 11 physicochemical features
(acidity, sugar, chlorides, sulphates, alcohol, …) and a `quality` score (3-8).

## What's built from scratch

Every core piece of the models is hand-implemented:

- **Train/test split** - custom index sampling (no `sklearn.model_selection`)
- **Entropy & information gain** - `-Σ p·log₂(p)`, weighted child entropy
- **Best-split search** - candidate midpoint thresholds per feature, chosen by
  lowest overall entropy
- **Decision tree** - recursive builder with purity checks, `minSampleSize`, and
  `maxDepth`
- **Random Forest** - **bootstrap** sampling + **random subspace** (random feature
  subsets) across a forest of trees, aggregated by **majority vote**
- **Metrics** - accuracy, precision, recall, F1 from a hand-computed confusion
  matrix

## Results

Random Forest on the held-out test set (dominant quality classes 5-6):

| Metric | Value |
|--------|-------|
| Accuracy | ~0.71 |
| Precision | ~0.76 |
| Recall | ~0.71 |

Single deep decision trees reach ~100% train accuracy (a clean demonstration of
overfitting), which the forest's bagging + random subspace helps counter.

## Tech stack

Python · NumPy · pandas · Matplotlib · seaborn · Jupyter Notebook.

## Run it

1. Download `winequality-red.csv` (UCI ML Repository / Kaggle) and update the path
   in the notebook.
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn
   ```
3. Open the notebook and run the cells top to bottom.

## Possible improvements

- Gini impurity as an alternative split criterion
- Binarize `quality` (good ≥ 7) for a cleaner, more balanced target
- Feature-importance from split frequency
- Benchmark the from-scratch models against scikit-learn
