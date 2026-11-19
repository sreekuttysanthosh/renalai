# Explainable Machine Learning for Screening Kidney Disease Markers Among Adults with Diabetes Using NHANES Clinical Data

## Abstract

This project develops a research and educational machine-learning workflow to
identify existing kidney abnormality markers among adults with diabetes using
NHANES 2017–March 2020 pre-pandemic data. The operational diabetes cohort
includes examined adults aged 20 or older with self-reported diabetes or HbA1c
>= 6.5%. Kidney-marker positivity is eGFR < 60 mL/min/1.73 m² or UACR >= 30
mg/g. The race-free 2021 CKD-EPI creatinine equation is used. A labeled cohort
of 1,506 participants is split into 1,204 training and 302 test participants.
Logistic Regression, Random Forest, and XGBoost are evaluated using stratified
cross-validation, hyperparameter search, and held-out metrics. SHAP explains
Random Forest behavior. The study is cross-sectional, unweighted, and not a
clinical diagnostic or future-risk prediction tool.

## Introduction and background

Kidney abnormalities are an important complication to assess among people with
diabetes. eGFR and urine albumin-to-creatinine ratio are complementary markers
used in kidney evaluation. Public NHANES data make it possible to explore
relationships between these markers and demographic, clinical, and lifestyle
features. Because the data are cross-sectional, the analysis describes existing
marker status and associations, not progression or causation.

The project uses the race-free 2021 CKD-EPI creatinine equation and SHAP
explanations. Methods and sources are detailed in
`reports/outcome_leakage_audit.md` and `reports/explainability.md`.

## Data and methods

Eight NHANES files were integrated by `SEQN`: demographics, diabetes
questionnaire, glycohemoglobin, biochemistry, urine albumin/creatinine,
anthropometry, blood pressure, and smoking questionnaire data. Data were
screened for duplicates, missingness, implausible values, and questionnaire
skip patterns. Survey design variables are retained, but not used for weighted
inference.

The agreed cohort definition is self-reported diabetes (`DIQ010=1`) OR HbA1c
(`LBXGH`) >= 6.5%, for adults aged at least 20 with an examination record;
pregnancy codes 1 and 3 are excluded. Medication use is analyzed descriptively
but is not an independent cohort criterion. The final eligible cohort has
1,601 participants. Of these, 1,506 have a classifiable outcome: 855 negative
and 651 positive; 95 have insufficient outcome measurements.

The label is positive when eGFR < 60 or NHANES UACR (`URDACT`) >= 30 mg/g.
Negative labels require both measures to be observed and normal. UACR was
cross-checked against `100 * URXUMA / URXUCR`; all 1,482 paired measurements
agree on classification at the 30 mg/g threshold, with maximum absolute
numeric difference 0.005 mg/g.

The nine predictors are age, female indicator, family income-to-poverty ratio,
HbA1c, BMI, systolic and diastolic blood pressure, smoking category, and insulin
status. Identifier `SEQN`, outcome measurements, eGFR, and the target are
excluded. HbA1c is both a cohort criterion and predictor, so a no-HbA1c
sensitivity experiment is included. Missing numerical values are imputed from
training data; categorical values are one-hot encoded; insulin missingness is
preserved with an indicator. Preprocessing is performed inside model pipelines.

An 80:20 stratified split yields 1,204 training and 302 test observations with
no participant overlap. Three models are compared using five-fold stratified
cross-validation and randomized parameter search. The classification
threshold remains the default 0.50 and is not clinically selected.

## Results

The saved pipelines were rebuilt from the supplied frozen split using five-fold stratified randomized search (seed 42). On the held-out set, Logistic Regression achieved accuracy 0.6854, ROC-AUC 0.7337, recall 0.5649, and F1 0.6091; Random Forest achieved accuracy 0.6722, ROC-AUC 0.7182, recall 0.6336, and F1 0.6264; XGBoost achieved accuracy 0.6689, ROC-AUC 0.7181, recall 0.5496, and F1 0.5902. Random Forest has the highest observed recall/F1 at this threshold; Logistic Regression has the highest ROC-AUC. The threshold remains 0.50 and is not clinically selected. The full comparison and confusion matrices are in `reports/model_evaluation.md` and `reports/model_evaluation.csv`.

Sensitivity analyses remove HbA1c, remove insulin, and replace BMI with waist
circumference. The notebook's recorded results are in
`reports/sensitivity_analysis.csv`; they predate the refreshed saved pipelines
and are exploratory, not reruns of the current models. An albuminuria-only
outcome analysis is not included in the supplied notebook.
Subgroup results by age, sex, and race/ethnicity are documented in
`reports/fairness_analysis.md`.

The modeling notebook's SHAP global analysis ranks insulin status, age, HbA1c, systolic blood
pressure, and the insulin missingness indicator highest in this fitted Random
Forest. Individual explanations are available in the dashboard and generated
report figures. SHAP attributions describe model behavior and do not establish
causal effects.

## Discussion

The models show moderate discrimination and a sensitivity/specificity trade-off
at the default threshold. Random Forest's higher recall may be relevant to
screening research, whereas Logistic Regression and XGBoost have higher
ranking metrics. Estimates are uncertain given the small held-out sample.
Results are exploratory and should not be used to prioritize or treat actual
patients.

## Ethics, privacy, fairness, and limitations

The source data are public NHANES survey data. The project does not use direct
clinical records or claim HIPAA/GDPR certification. It follows data-minimization
principles in the model by excluding participant ID and outcome-defining kidney
measurements. The dashboard is for synthetic/educational entries only; users
should not enter real patient data.

Subgroup metrics vary, but subgroup sizes and positive-case counts are limited.
The evaluation is unweighted and does not represent national prevalence.
Cross-sectional measurements cannot establish persistent CKD, future risk, or
causality. A single abnormal result does not confirm chronic kidney disease.
Missing measurements, one fixed split, default threshold, and the operational
diabetes definition limit generalizability.

## Conclusion

The project provides an end-to-end NHANES data workflow, three-model
comparison, SHAP explanations, subgroup audit, albuminuria-only sensitivity
analysis, and Streamlit demonstration. It supports educational exploration of
associations with existing kidney markers, not clinical diagnosis or
longitudinal prediction. Survey-weighted inference remains an optional
extension.

## References and data sources

- CDC/NCHS. NHANES 2017–March 2020 pre-pandemic data and documentation:
  <https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?cycle=2017-2020>
- Inker LA, et al. New Creatinine- and Cystatin C-Based Equations to Estimate
  GFR without Race. *N Engl J Med*. 2021;385:1737–1749.
  <https://doi.org/10.1056/NEJMoa2102953>
- Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions.
  NeurIPS 2017. <https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions>
- Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD 2016.
  <https://doi.org/10.1145/2939672.2939785>
