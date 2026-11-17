
# Renal AI | Explainable Machine Learning for Kidney Health Screening

### An End-to-End Healthcare Analytics, Machine Learning and Explainable AI Project

<p align="center">
  <a href="https://renaliq-u.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Renaliq-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Explainable%20AI-SHAP-purple?style=for-the-badge" />
</p>

**Live Application:** https://renaliq-u.streamlit.app/

---

## 1. Project Overview

Renal AI is an explainable machine learning project that investigates whether demographic, clinical, metabolic and lifestyle characteristics can help identify existing abnormal kidney markers among adults with diabetes.

The project uses publicly available data from the **National Health and Nutrition Examination Survey (NHANES), 2017–March 2020 pre-pandemic cycle**, conducted by the US Centers for Disease Control and Prevention (CDC).

Multiple NHANES datasets are integrated to construct a study cohort and identify kidney abnormalities using two complementary indicators:

- **Estimated Glomerular Filtration Rate (eGFR):** An estimate of kidney filtration function.
- **Urine Albumin-to-Creatinine Ratio (UACR):** An indicator of albumin leakage into urine.

Three machine learning algorithms are developed and evaluated:

- Logistic Regression
- Random Forest
- XGBoost

SHAP (SHapley Additive exPlanations) is used to interpret individual model predictions. An interactive Streamlit application, **Renaliq**, demonstrates the prediction and explainability workflow.

The project demonstrates an end-to-end healthcare analytics workflow, combining healthcare data integration, exploratory analysis, predictive modelling, model evaluation, explainable AI and application development.

## 2. Problem Statement

Diabetes is an important risk factor associated with kidney complications.

Kidney abnormalities can involve reduced filtration function or increased albumin leakage into urine. These changes are assessed using measurements such as eGFR and UACR.

However, identifying individuals with abnormal kidney markers requires analysing multiple types of healthcare information rather than relying on a single demographic or clinical characteristic.

This project investigates:

**Can routinely available demographic and clinical characteristics be used by machine learning models to identify adults with diabetes who have existing abnormal kidney markers?**

The project addresses challenges involving heterogeneous healthcare data integration, cohort construction, outcome definition, missing-data handling, target leakage prevention, predictive modelling and model interpretability.

## 3. Proposed Solution

An end-to-end machine learning pipeline was developed to investigate the research problem.

The workflow includes:

1. Integration of eight NHANES datasets using the participant identifier `SEQN`.
2. Data cleaning and quality assessment.
3. Identification of eligible adults with diabetes.
4. Construction of a binary kidney-marker outcome using eGFR and UACR.
5. Exploratory data analysis and missing-data assessment.
6. Predictor selection and target leakage auditing.
7. Missing-value imputation and feature preprocessing.
8. Development of three machine learning classification models.
9. Model evaluation and comparison.
10. SHAP-based individual prediction explanations.
11. Development of an interactive Streamlit application.

The final prototype, Renaliq, demonstrates how machine learning predictions and their feature-level explanations can be presented through an interactive interface.

## 4. Key Project Objectives

- Integrate demographic, examination, laboratory and questionnaire data from NHANES.
- Construct a study cohort of adults with diabetes.
- Define kidney abnormalities using eGFR and UACR thresholds.
- Perform exploratory data analysis and data-quality assessment.
- Engineer a clinically relevant binary outcome.
- Prevent target leakage during feature selection and model development.
- Implement reproducible preprocessing pipelines.
- Develop and compare multiple classification algorithms.
- Evaluate predictive performance using appropriate classification metrics.
- Apply SHAP to interpret individual model predictions.
- Develop an interactive Streamlit research prototype.
- Examine model limitations, subgroup performance and generalizability.

## 5. Dataset and Cohort Construction

**Dataset:** National Health and Nutrition Examination Survey (NHANES)

**Study cycle:** 2017–March 2020 pre-pandemic cycle

**Source:** CDC / National Center for Health Statistics (NCHS)

Eight NHANES data components were integrated using `SEQN`, the participant identifier.

| Dataset component | Information extracted |
|---|---|
| Demographics | Age, sex and socioeconomic information |
| Diabetes questionnaire | Self-reported diabetes and treatment information |
| Glycohemoglobin | HbA1c measurements |
| Biochemistry | Serum creatinine |
| Urine laboratory data | Urine albumin and creatinine |
| Anthropometry | BMI and body measurements |
| Blood pressure | Systolic and diastolic blood pressure |
| Smoking questionnaire | Smoking information |

### Cohort Selection

Adults aged 20 years and above were selected using the project's predefined diabetes eligibility criteria.

The cohort was constructed using self-reported diabetes and HbA1c-based criteria, followed by the required eligibility and data-quality checks.

### Cohort Summary

| Stage | Records |
|---|---:|
| Merged dataset | 15,560 |
| Eligible records | 1,601 |
| Classifiable kidney-marker outcomes | 1,506 |
| Positive kidney-marker cases | 651 |
| Negative kidney-marker cases | 855 |
| Insufficient outcome measurements | 95 |

## 6. Kidney Abnormality Definition

A binary target variable named `kidney_marker` was constructed using two kidney-related measurements.

### Positive Classification

A participant was classified as positive if either condition was satisfied:

- eGFR < 60 mL/min/1.73 m²
- UACR ≥ 30 mg/g

### Negative Classification

A participant was classified as negative only when both measurements were available and within the normal ranges defined by the study.

eGFR was calculated using the race-free 2021 CKD-EPI creatinine equation.

The NHANES-provided UACR measurement was used and validated against its underlying urine albumin and creatinine measurements.

**Important:** A single abnormal eGFR or UACR measurement does not establish chronic kidney disease (CKD). Clinical diagnosis requires further assessment and evidence of persistence over time.

## 7. Predictive Features

Nine demographic, socioeconomic, clinical and lifestyle variables were selected as model predictors.

| Feature | Description | Category |
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

Direct kidney outcome measurements, including eGFR, UACR, serum creatinine and urine albumin measurements, were excluded from the predictor variables to prevent target leakage.

The participant identifier `SEQN` was also excluded from model predictors.

## 8. Methodology

### Data Engineering and Preprocessing

- Multi-file dataset integration using participant identifiers.
- Data cleaning and duplicate assessment.
- Missing-value and data-quality analysis.
- Numerical and categorical feature handling.
- Numerical imputation using training data.
- Categorical encoding.
- Numerical feature scaling where required.
- Missingness indicator handling for insulin status.
- Scikit-learn preprocessing pipelines.

All preprocessing transformations were fitted using training data to reduce information leakage.

### Train-Test Split

- Stratified 80:20 train-test split.
- Training records: 1,204.
- Testing records: 302.
- Stratified five-fold cross-validation on the training data.
- Randomized hyperparameter search.

### Machine Learning Algorithms

| Algorithm | Description |
|---|---|
| Logistic Regression | Baseline linear classification model |
| Random Forest | Ensemble learning using multiple decision trees |
| XGBoost | Gradient-boosted decision-tree classifier |

## 9. Model Evaluation and Results

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrices

### Model Comparison

| Model | ROC-AUC | Recall | F1-score | PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7301 | 0.5496 | 0.5902 | 0.700 |
| Random Forest | 0.7182 | 0.6260 | 0.6212 | 0.684 |
| XGBoost | 0.7272 | 0.5420 | 0.5868 | 0.717 |

All three models achieved approximately 66.9% accuracy.

### Model Selection

Random Forest was selected for individual-level SHAP analysis because it achieved the highest recall and F1-score at the default classification threshold of 0.50.

Logistic Regression achieved the highest ROC-AUC, while XGBoost achieved the highest PR-AUC.

Model selection was therefore based on the project's evaluation priorities rather than performance on a single metric.

The results indicate moderate predictive performance and require further validation.

## 10. Explainable Artificial Intelligence (SHAP)

SHAP (SHapley Additive exPlanations) was implemented to understand how individual features influence model predictions.

The project uses individual-level explanations to investigate the contribution of different input features to a participant's prediction.

### Explainability Components

- Individual SHAP waterfall plots.
- Positive and negative feature contributions.
- Feature-level interpretation of individual predictions.
- Global feature importance and SHAP analysis in the research evaluation.

SHAP improves model interpretability by explaining how features contribute to model outputs.

However, SHAP values explain model behaviour and do not establish causal relationships or biological mechanisms.

## 11. Renaliq – Interactive Streamlit Application

**Live Demo:** https://renaliq-u.streamlit.app/

Renaliq is an interactive research prototype developed using Streamlit.

It demonstrates how a trained machine learning pipeline can be integrated into a user-facing application.

### Dashboard Capabilities

- Demographic and clinical feature inputs.
- Model-based kidney-marker classification.
- Estimated prediction probability.
- Individual-level SHAP explanations.
- Feature contribution interpretation.
- Research-use disclaimer and limitations.

The application connects predictive modelling with explainable AI through an interactive interface.

The dashboard is not connected to hospital information systems and is not intended for real patient diagnosis or treatment decisions.

## 12. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming and model development |
| Pandas | Data integration and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data visualization |
| Scikit-learn | Preprocessing, pipelines, classification and evaluation |
| XGBoost | Gradient-boosted classification |
| SHAP | Explainable AI |
| Joblib | Model serialization and loading |
| Streamlit | Interactive application development |
| Jupyter Notebook | Data exploration and experimentation |
| Git and GitHub | Version control and project management |

## 13. Skills Demonstrated and Key Learning Outcomes

This project provided practical experience across the healthcare analytics and machine learning lifecycle.

### Healthcare Data Analytics
- Integrating heterogeneous healthcare datasets.
- Understanding clinical and laboratory measurements.
- Constructing research cohorts.
- Defining outcomes using clinical thresholds.
- Assessing missing data and data quality.

### Machine Learning
- Building classification pipelines.
- Implementing Logistic Regression, Random Forest and XGBoost.
- Hyperparameter optimization.
- Cross-validation and stratified data splitting.
- Evaluating classification metrics and threshold-dependent performance.

### Explainable AI
- Understanding SHAP values.
- Interpreting feature contributions.
- Analysing individual predictions.
- Distinguishing model associations from causal explanations.

### Software Development and Deployment
- Saving and loading trained machine learning models.
- Integrating machine learning pipelines into Streamlit.
- Building interactive prediction workflows.
- Managing project dependencies.
- Using Git and GitHub for version control.

### Research and Critical Analysis
- Translating a healthcare problem into a machine learning task.
- Identifying target leakage.
- Evaluating model limitations.
- Interpreting subgroup performance.
- Understanding external validation and generalizability.

## 14. Sensitivity Analysis and Subgroup Evaluation

Sensitivity analyses explored the effect of alternative modelling choices:

- Removing HbA1c.
- Removing insulin.
- Replacing BMI with waist circumference.
- Considering an albuminuria-only outcome among participants with observed UACR.

Subgroup performance was also explored across age, sex and race/ethnicity groups.

These analyses were intended to examine the robustness of model findings and identify areas requiring further investigation.

## 15. Limitations

- NHANES is a cross-sectional survey.
- The analysis is unweighted and does not estimate national prevalence.
- A single abnormal kidney-marker measurement does not establish CKD.
- Model performance may be affected by missing data and sample size.
- Subgroup estimates require cautious interpretation.
- Findings may not generalize to other populations, including Indian populations.
- No independent hospital-based external validation has been performed.
- The application is an academic research prototype, not a clinically validated diagnostic system.

## 16. Future Scope

Potential extensions include:

- External validation using independent clinical datasets.
- Investigation of Indian healthcare datasets for population-specific evaluation.
- Longitudinal analysis of kidney-function changes.
- Evaluation of additional missing-data strategies.
- Probability calibration and threshold optimization.
- Expanded subgroup and fairness analysis.
- Further development of the interactive dashboard.
- Integration with validated healthcare decision-support workflows, subject to appropriate clinical and regulatory evaluation.

## 17. Repository Structure

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

*Note: This represents the intended repository organization. Update filenames and folders to match the actual repository.*

## 18. Installation and Usage

### Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd renaliq-main
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment (Windows)

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
python -m streamlit run dashboard/app.py
```

The application will open in your browser.

## 19. Data Privacy and Responsible Use

This project uses publicly available NHANES research data.

- Raw participant-level datasets should not be committed to a public repository.
- Participant identifiers are excluded from model predictors.
- Appropriate data-handling and privacy practices should be followed.
- Predictions must not be interpreted as clinical diagnoses or treatment recommendations.

## 20. Disclaimer

Renal AI is developed exclusively for academic, research and educational purposes.

The model estimates the likelihood of an abnormal kidney-marker classification based on patterns learned from NHANES data.

It does not diagnose chronic kidney disease, predict future disease progression or replace professional medical assessment.

## 21. Acknowledgements

Data used in this project were obtained from the National Health and Nutrition Examination Survey (NHANES), conducted by the US Centers for Disease Control and Prevention (CDC) and the National Center for Health Statistics (NCHS).

---

<p align="center">
  <b>Renal AI | Renaliq</b><br>
  Healthcare Analytics • Machine Learning • Explainable AI
  <br><br>
  <a href="https://renaliq-u.streamlit.app/">Explore the Live Application</a>
</p>
