# Outcome and predictor leakage audit

The outcome is constructed from race-free 2021 CKD-EPI eGFR and NHANES UACR (`URDACT`). The model inputs are age, sex, income-to-poverty ratio, HbA1c, BMI, systolic and diastolic blood pressure, smoking, and insulin status. Participant ID (`SEQN`) is retained only for joins and split-overlap checks.

The model-ready `model_train.csv` and `model_test.csv` do not contain serum creatinine (`LBXSCR`), urine albumin (`URXUMA`), urine creatinine (`URXUCR`), reported UACR (`URDACT`), derived eGFR, or any additional outcome-defining measurement. The target (`kidney_marker`) is kept separate from the feature list. The training script checks that IDs do not overlap and constructs `X` using the explicit nine-feature allowlist.

HbA1c is an intended predictor and is also part of the operational cohort definition (`HbA1c >= 6.5%` OR self-reported diabetes). This can make the predictor partly informative about cohort selection; the notebook includes an HbA1c-removal sensitivity analysis. Results should therefore be interpreted as conditional on this study cohort definition.

The model is cross-sectional. This is not prospective prediction: its target is an existing marker status measured in the same NHANES cycle. Excluding direct kidney measurements from predictors prevents direct target leakage, but it does not remove all selection effects or establish temporal prediction.
