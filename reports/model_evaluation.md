# Held-out model evaluation

The three pipelines were rebuilt with the project's fixed, stratified split and a five-fold stratified randomized search (seed 42). The training set has 1,204 participants and the held-out test set has 302, with no participant identifier overlap. Imputation, scaling, and categorical encoding are fitted inside each training fold. The classification threshold is 0.50.

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 68.5% | 66.1% | 56.5% | 77.8% | 0.609 | 0.734 | 0.705 |
| Random Forest | 67.2% | 61.9% | 63.4% | 70.2% | 0.626 | 0.718 | 0.687 |
| XGBoost | 66.9% | 63.7% | 55.0% | 76.0% | 0.590 | 0.718 | 0.697 |

Random Forest had the highest sensitivity and F1 at this threshold. Logistic Regression had the highest ROC-AUC; XGBoost had the highest specificity and the middle PR-AUC. No model dominates every measure. These are unweighted, sample-level results from one held-out split and do not show clinical utility or nationally representative performance. The complete precision values, confusion-matrix counts, selected parameters, and cross-validation ROC-AUC values are in `model_evaluation.csv`.

## Confusion matrices

Counts are ordered as true negatives, false positives, false negatives, true positives:

- Logistic Regression: 133, 38, 57, 74.
- Random Forest: 120, 51, 48, 83.
- XGBoost: 130, 41, 59, 72.

## Interpretation

The threshold was not selected for clinical use. Since the target is an operational, cross-sectional kidney-marker label and the test sample is modest, these results are exploratory. Further validation, uncertainty intervals, threshold assessment, and survey-design-aware analysis would be needed before considering any real-world use.
