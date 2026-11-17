# Renal AI | Explainable Machine Learning for Kidney Health Screening

**An end-to-end Healthcare Analytics and Explainable AI project using NHANES clinical data**

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-purple)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)](https://streamlit.io/)

**Live Application:** https://renaliq-u.streamlit.app/

---

## 1. Project Overview

Renal AI is an explainable machine learning project that investigates whether routinely available demographic and clinical characteristics can help identify abnormal kidney markers among adults with diabetes.

The project uses publicly available health examination, laboratory and questionnaire data from the **National Health and Nutrition Examination Survey (NHANES), 2017–March 2020 pre-pandemic cycle**.

Multiple NHANES datasets are integrated to construct a diabetes cohort and identify kidney abnormalities using two complementary indicators:

- **Estimated Glomerular Filtration Rate (eGFR):** An estimate of kidney filtration function.
- **Urine Albumin-to-Creatinine Ratio (UACR):** An indicator of albumin leakage into urine.

Three machine learning algorithms are developed and compared. SHAP is incorporated to interpret individual model predictions, and an interactive Streamlit application, **Renaliq**, demonstrates the prediction and explanation workflow.

The project brings together healthcare data engineering, statistical analysis, predictive modelling and explainable artificial intelligence in one reproducible research workflow.

## 2. Problem Statement

Diabetes is an important risk factor associated with kidney complications. Identifying abnormal kidney markers is an important part of monitoring kidney health among people with diabetes.

However, kidney abnormalities cannot always be understood through a single clinical measurement. Reduced kidney filtration and increased urinary albumin represent different aspects of kidney health.

This project investigates the following question:

**Can demographic, metabolic, clinical and lifestyle characteristics be used by machine learning models to identify adults with diabetes who have existing abnormal kidney markers?**

The challenge involves integrating heterogeneous healthcare datasets, defining a reliable outcome from laboratory measurements, preventing target leakage, evaluating predictive performance and making model predictions interpretable.

## 3. Proposed Solution

An end-to-end machine learning pipeline was developed to address the research problem.

The workflow consists of:

1. Integrating eight NHANES datasets using the participant identifier `SEQN`.
2. Cleaning the data and performing data-quality assessments.
3. Identifying eligible adults with diabetes using predefined criteria.
4. Constructing a binary kidney-marker outcome using eGFR and UACR.
5. Performing exploratory data analysis and missing-data assessment.
6. Selecting predictors and auditing potential target leakage.
7. Building preprocessing pipelines for numerical and categorical variables.
8. Training and comparing Logistic Regression, Random Forest and XGBoost.
9. Evaluating performance using multiple classification metrics.
10. Applying SHAP to explain individual predictions.
11. Developing an interactive Streamlit application for model demonstration.

The final application, Renaliq, allows users to enter demographic and clinical characteristics, obtain a model prediction and inspect the features contributing to that individual prediction.

## 4. Key Project Objectives

- Integrate demographic, examination, laboratory and questionnaire data from NHANES.
- Construct a well-defined study cohort of adults with diabetes.
- Identify abnormal kidney markers using eGFR and UACR thresholds.
- Conduct exploratory data analysis and assess data quality.
- Engineer a clinically relevant binary outcome.
- Prevent target leakage through appropriate predictor selection.
- Implement reproducible preprocessing and machine learning pipelines.
- Compare multiple classification algorithms.
- Evaluate classification performance and the trade-off between false negatives and false positives.
- Apply explainable AI to interpret individual model predictions.
- Build an interactive research prototype using Streamlit.
- Examine limitations, subgroup performance and potential generalizability.

## 5. Dataset and Cohort Construction

**Data source:** National Health and Nutrition Examination Survey (NHANES), CDC/NCHS.

**Study period:** 2017–March 2020 pre-pandemic cycle.

Eight data components were integrated using `SEQN`, the NHANES participant identifier.

| Data component | Information extracted |
|---|---|
| Demographics | Age, sex and socioeconomic information |
| Diabetes questionnaire | Self-reported diabetes and treatment information |
| Glycohemoglobin | HbA1c measurements |
| Biochemistry | Serum creatinine |
| Urine laboratory data | Urine albumin and creatinine |
| Anthropometry | BMI and body measurements |
| Blood pressure | Systolic and diastolic blood pressure |
| Smoking questionnaire | Smoking information |

### Cohort selection

The study focuses on adults aged 20 years and above with diabetes, identified using the project's operational eligibility criteria.

The cohort was constructed using self-reported diabetes and HbA1c-based criteria, followed by the predefined data-quality and eligibility checks.

**Cohort summary:**

| Stage | Records |
|---|---:|
| Merged dataset | 15,560 |
| Eligible records | 1,601 |
| Classifiable kidney-marker outcomes | 1,506 |
| Positive kidney-marker cases | 651 |
| Negative kidney-marker cases | 855 |
| Excluded due to insufficient outcome measurements | 95 |

## 6. Kidney Abnormality Outcome Definition

A binary target variable, `kidney_marker`, was constructed using two kidney-related measurements.

**Positive kidney-marker classification:**

- eGFR < 60 mL/min/1.73 m², OR
- UACR ≥ 30 mg/g.

**Negative kidney-marker classification:**

Both eGFR and UACR must be observed and within the study's normal ranges.

eGFR was calculated using the race-free 2021 CKD-EPI creatinine equation.

The NHANES-provided UACR measurement was used and checked against its underlying urine albumin and creatinine measurements.

This combined outcome captures two complementary indicators of kidney abnormality.

**Clinical interpretation:** A single abnormal eGFR or UACR measurement does not establish chronic kidney disease. CKD diagnosis requires clinical assessment and evidence that abnormalities persist over time.

## 7. Predictive Features

Nine demographic, socioeconomic, clinical and lifestyle features were selected for model development.

| Feature | Description | Domain |
|---|---|---|
| Age | Participant age | Demographic |
| Sex | Female indicator | Demographic |
| INDFMPIR | Family income-to-poverty ratio | Socioeconomic |
| HbA1c | Glycated haemoglobin | Metabolic |
| BMI | Body mass index | Anthropometric |
| Systolic BP | Systolic blood pressure | Clinical |
| Diastolic BP | Diastolic blood pressure | Clinical |
| Smoking | Smoking category | Lifestyle |
| Insulin | Insulin status | Treatment-related |

Direct kidney outcome measurements, including eGFR, UACR, serum creatinine and urine albumin measurements, were excluded from model predictors to prevent target leakage.

The participant identifier `SEQN` was also excluded from model features.

## 8. Technical Methodology

### Data engineering and preprocessing

- Multi-file integration using participant identifiers.
- Data cleaning and duplicate assessment.
- Missing-value and data-quality analysis.
- Numerical and categorical feature handling.
- Training-data-based imputation.
- Categorical encoding.
- Numerical feature scaling where required.
- Missingness indicator handling for insulin status.
- Scikit-learn preprocessing pipelines.

All preprocessing transformations were fitted using training data to reduce information leakage.

### Machine learning models

Three classification algorithms were implemented:

| Algorithm | Role in the project |
|---|---|
| Logistic Regression | Interpretable linear baseline classifier |
| Random Forest | Ensemble learning using multiple decision trees |
| XGBoost | Gradient-boosted decision-tree classifier |

Hyperparameter tuning was performed using randomized search with stratified five-fold cross-validation on the training dataset.

A stratified 80:20 train-test split was used.

### Model evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrices

Particular attention was given to recall because missed positive kidney-marker cases are an important consideration in screening-oriented research.

## 9. Model Performance

The following results were obtained on the held-out test dataset using the default classification threshold of 0.50.

| Model | ROC-AUC | Recall | F1-score | PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7301 | 0.5496 | 0.5902 | 0.700 |
| Random Forest | 0.7182 | 0.6260 | 0.6212 | 0.684 |
| XGBoost | 0.7272 | 0.5420 | 0.5868 | 0.717 |

All three models achieved approximately 66.9% accuracy.

**Model selection:** Random Forest was selected for individual-level SHAP analysis because it achieved the highest recall and F1-score at the default threshold.

Logistic Regression achieved the highest ROC-AUC, while XGBoost achieved the highest PR-AUC. Therefore, model selection was based on the project's evaluation priorities rather than a claim that one algorithm performed best across every metric.

The results indicate moderate predictive performance and require further validation.

## 10. Explainable Artificial Intelligence

SHAP (SHapley Additive exPlanations) was used to investigate how model features influence predictions.

The project incorporates explainability to improve the transparency of machine learning outputs.

**Individual-level explainability:**

- SHAP waterfall plots.
- Positive and negative feature contributions.
- Feature-level interpretation for individual predictions.

The research analysis also examines global feature importance and SHAP-based feature patterns.

SHAP explanations describe the behaviour of the trained model. They do not establish causal relationships or demonstrate that a feature is biologically responsible for kidney abnormalities.

## 11. Renaliq – Interactive Research Application

**Live application:** https://renaliq-u.streamlit.app/

Renaliq is a Streamlit-based interactive research prototype developed to demonstrate the practical application of the trained machine learning pipeline.

The application provides:

- Demographic and clinical feature inputs.
- Model-based kidney-marker classification.
- Estimated model prediction probability.
- Individual-level SHAP explanations.
- Interactive interpretation of prediction contributions.
- Research-use disclaimer and limitations.

The dashboard connects machine learning outputs with an accessible user interface, demonstrating how predictive modelling and explainable AI can be presented through an interactive application.

The application is not connected to a hospital information system and is not intended for real patient diagnosis or treatment decisions.

## 12. Technology Stack

| Technology | Application |
|---|---|
| Python | Core programming and model development |
| Pandas | Dataset integration and data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data visualization |
| Scikit-learn | Preprocessing, pipelines, classification and evaluation |
| XGBoost | Gradient-boosted machine learning |
| SHAP | Explainable AI and individual prediction interpretation |
| Joblib | Model serialization and loading |
| Streamlit | Interactive dashboard development |
| Jupyter Notebook | Exploratory analysis and experimentation |
| Git and GitHub | Version control and project management |

## 13. Skills Demonstrated and Key Learning Outcomes

This project provided practical experience across the healthcare analytics and machine learning lifecycle.

**Healthcare data analytics**
- Integrating heterogeneous health datasets.
- Understanding clinical and laboratory variables.
- Constructing research cohorts.
- Defining outcomes using clinical measurement thresholds.
- Assessing missingness and data quality.

**Machine learning**
- Building classification pipelines.
- Implementing Logistic Regression, Random Forest and XGBoost.
- Hyperparameter optimization.
- Cross-validation and stratified train-test splitting.
- Evaluating classification performance and threshold-dependent metrics.

**Explainable AI**
- Understanding SHAP values.
- Interpreting feature contributions.
- Analysing individual model predictions.
- Distinguishing model associations from causal explanations.

**Software development and deployment**
- Saving and loading trained machine learning models.
- Integrating models into a Streamlit application.
- Creating interactive prediction workflows.
- Managing project files and dependencies.
- Using Git and GitHub for version control.

**Research and critical analysis**
- Translating a healthcare question into a machine learning problem.
- Identifying target leakage.
- Evaluating model limitations.
- Interpreting subgroup performance.
- Understanding generalizability and external validation requirements.

## 14. Sensitivity Analysis and Limitations

Sensitivity analyses explored the impact of alternative modelling choices, including:

- Removing HbA1c.
- Removing insulin.
- Replacing BMI with waist circumference.
- Considering an albuminuria-only outcome among participants with observed UACR.

Subgroup performance was also explored across age, sex and race/ethnicity groups.

The following limitations are acknowledged:

- NHANES is a cross-sectional survey.
- The analysis is unweighted and does not estimate national prevalence.
- A single abnormal measurement does not establish CKD.
- Model performance may be affected by missing data and sample size.
- Subgroup results require cautious interpretation.
- Findings may not generalize to Indian or other populations.
- No independent hospital-based external validation has been conducted.
- The application is a research prototype, not a clinically validated decision-support system.

## 15. Future Scope

Potential future extensions include:

- Validation using independent clinical or hospital-based datasets.
- Investigation of Indian healthcare datasets for population-specific evaluation.
- Longitudinal analysis to investigate persistent kidney abnormalities and future outcomes.
- Further evaluation of missing-data strategies and classification thresholds.
- Calibration and external validation of predicted probabilities.
- Expanded fairness and subgroup performance analysis.
- Additional explainability and model-monitoring capabilities.
- Further development of the interactive application.

## 16. Repository Structure

```text
renaliq-main/
│
├── data/
│   └── README.md
│
├── models/
│   ├── all_models.joblib
│   └── model_metadata.json
│
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── exploratory_data_analysis.ipynb
│   └── modeling.ipynb
│
├── src/
│   └── ...
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
└── README.md
```

*Note: This is the intended project organization. Filenames should be updated to reflect the actual repository contents.*

## 17. Running the Application Locally

### Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd renaliq-main
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the dashboard

```bash
python -m streamlit run dashboard/app.py
```

The application will open in your local browser.

## 18. Data Privacy and Responsible Use

This project uses publicly available NHANES research data.

- Raw participant-level datasets should not be committed to a public repository.
- Participant identifiers are excluded from model predictors.
- Appropriate data handling and privacy practices should be followed.
- Predictions must not be used as clinical diagnoses or treatment recommendations.

## 19. Disclaimer

Renal AI is developed exclusively for academic, research and educational purposes.

The model estimates the likelihood of an abnormal kidney-marker classification based on the patterns learned from the NHANES training data. It does not diagnose chronic kidney disease, predict future disease progression or replace professional medical assessment.

## 20. Acknowledgements

Data used in this project were obtained from the National Health and Nutrition Examination Survey (NHANES), conducted by the US Centers for Disease Control and Prevention (CDC) and the National Center for Health Statistics (NCHS).

---

**Project:** Renal AI / Renaliq  
**Domain:** Healthcare Analytics | Machine Learning | Explainable AI  
**Application:** https://renaliq-u.streamlit.app/  
**Status:** Research prototype and machine learning implementation developed; ongoing refinement and documentation.

---
## 21. Contributors
1. SREEKUTTY SANTHOSH
2. THEERTHA VIJYACHANDRAN 
3. YADHU KRISHNA VB 



