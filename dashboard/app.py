
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Renal AI | Kidney Marker Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. CUSTOM UI DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #F7FAFA;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #155E59 !important;
    }

    h1 {
        font-weight: 750 !important;
        letter-spacing: -1px;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCEBE8;
        padding: 18px;
        border-radius: 12px;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background-color: #16877C;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 0.55rem 1.5rem;
        font-weight: 600;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #116A62;
        color: white;
        border: none;
    }

    [data-testid="stExpander"] {
        background-color: white;
        border: 1px solid #DCEBE8;
        border-radius: 10px;
    }

    [data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid #DCEBE8;
        padding: 22px;
        border-radius: 14px;
    }

    hr {
        border-color: #DCEBE8;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. PROJECT PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "all_models.joblib"

TEST_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_test.csv"
)

TRAIN_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_train.csv"
)

SHAP_PATH = (
    PROJECT_ROOT / "reports" / "shap_global_importance.csv"
)

THRESHOLD = 0.50

MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
]

FEATURE_NAMES = [
    "RIDAGEYR",
    "female",
    "INDFMPIR",
    "LBXGH",
    "BMXBMI",
    "BPXOSY1",
    "BPXODI1",
    "smoking",
    "insulin",
]


# ============================================================
# 4. LOAD ALL THREE MODELS
# ============================================================

@st.cache_resource
def load_models():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "all_models.joblib was not found inside the models folder."
        )

    models = joblib.load(MODEL_PATH)

    if not isinstance(models, dict):
        raise ValueError(
            "The saved model file must contain a dictionary of three pipelines."
        )

    if set(models.keys()) != set(MODEL_NAMES):
        raise ValueError(
            "The saved file does not contain all three expected models."
        )

    for name, model in models.items():

        if not hasattr(model, "named_steps"):
            raise ValueError(
                f"{name} is not a complete preprocessing pipeline."
            )

        if list(model.feature_names_in_) != FEATURE_NAMES:
            raise ValueError(
                f"{name} has unexpected input features."
            )

    return models


# ============================================================
# 5. LOAD HELD-OUT TEST DATA
# ============================================================

@st.cache_data
def load_test_data():

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "data/processed/model_test.csv was not found."
        )

    data = pd.read_csv(TEST_DATA_PATH)

    required_columns = set(FEATURE_NAMES) | {
        "SEQN",
        "kidney_marker",
    }

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            "Missing test-data columns: "
            + ", ".join(sorted(missing))
        )

    if len(data) != 302:
        raise ValueError(
            f"Expected 302 test participants, found {len(data)}."
        )

    if data["SEQN"].duplicated().any():
        raise ValueError(
            "Duplicate participant IDs found in test data."
        )

    if data["kidney_marker"].isna().any():
        raise ValueError(
            "Missing kidney_marker values found in test data."
        )

    X_test = data[FEATURE_NAMES].copy()

    y_test = data["kidney_marker"].astype(int)

    return X_test, y_test


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

def evaluate_models(models, X_test, y_test):

    results = []

    predictions = {}

    for name, model in models.items():

        positive_index = list(model.classes_).index(1)

        probabilities = model.predict_proba(X_test)[
            :, positive_index
        ]

        predicted = (
            probabilities >= THRESHOLD
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predicted,
            labels=[0, 1],
        ).ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(
                    y_test, predicted
                ),
                "Precision": precision_score(
                    y_test, predicted, zero_division=0
                ),
                "Sensitivity": recall_score(
                    y_test, predicted, zero_division=0
                ),
                "Specificity": specificity,
                "F1-score": f1_score(
                    y_test, predicted, zero_division=0
                ),
                "ROC-AUC": roc_auc_score(
                    y_test, probabilities
                ),
                "PR-AUC": average_precision_score(
                    y_test, probabilities
                ),
            }
        )

        predictions[name] = {
            "probabilities": probabilities,
            "predictions": predicted,
        }

    results_df = pd.DataFrame(results)

    return results_df, predictions


# ============================================================
# 7. SHAP EXPLANATION
# ============================================================

def explain_prediction(model_name, model, input_row):

    preprocessor = model.named_steps["preprocessor"]

    classifier = model.named_steps["classifier"]

    transformed_input = preprocessor.transform(input_row)

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    # Tree models
    if model_name in ["Random Forest", "XGBoost"]:

        explainer = shap.TreeExplainer(classifier)

    # Logistic Regression
    else:

        if TRAIN_DATA_PATH.exists():

            background_data = pd.read_csv(
                TRAIN_DATA_PATH
            )[FEATURE_NAMES].copy()

        else:

            X_test, _ = load_test_data()

            background_data = X_test.copy()

        # Use a small representative background sample.
        background_data = background_data.sample(
            n=min(50, len(background_data)),
            random_state=42,
        )

        transformed_background = preprocessor.transform(
            background_data
        )

        if hasattr(transformed_background, "toarray"):
            transformed_background = (
                transformed_background.toarray()
            )

        explainer = shap.LinearExplainer(
            classifier,
            transformed_background,
        )

    explanation = explainer(transformed_input)

    values = np.asarray(explanation.values)

    base_values = np.asarray(explanation.base_values)

    positive_index = list(model.classes_).index(1)

    if values.ndim == 3:

        local_values = values[0, :, positive_index]

        local_base = base_values.reshape(
            -1, len(model.classes_)
        )[0, positive_index]

    else:

        local_values = values[0]

        local_base = base_values.reshape(-1)[0]

    feature_names = preprocessor.get_feature_names_out()

    waterfall = shap.Explanation(
        values=local_values,
        base_values=local_base,
        data=np.asarray(transformed_input)[0],
        feature_names=feature_names,
    )

    return waterfall



# ============================================================
# 8. HEADER
# ============================================================

st.markdown(
    """
<div style="background: linear-gradient(110deg, #E4F3F0, #F8FBFA);
            padding: 30px;
            border-radius: 16px;
            border-left: 6px solid #16877C;
            margin-bottom: 20px;">

    <div style="color: #155E59;
                font-size: 36px;
                font-weight: 750;
                letter-spacing: -1px;">
        Renal AI
    </div>

    <div style="color: #426C68;
                font-size: 18px;
                margin-top: 8px;">
        Explainable Machine Learning for Kidney Marker Screening
    </div>

    <div style="color: #58716E;
                font-size: 13px;
                margin-top: 12px;">
        NHANES 2017–March 2020 | Healthcare Analytics | Research Prototype
    </div>

</div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Research and educational demonstration only. "
    "This application estimates existing study-defined kidney markers. "
    "It does not diagnose kidney disease or predict future disease."
)

st.divider()


# ============================================================
# 9. LOAD MODELS
# ============================================================

try:

    models = load_models()

except Exception as error:

    st.error(f"Model loading failed: {error}")

    st.stop()


# ============================================================
# 10. MODEL SELECTION
# ============================================================

st.header("Prediction workspace")

st.write(
    "Select a machine learning model and enter the participant's "
    "demographic and clinical information."
)

selected_model_name = st.selectbox(
    "Select machine learning model",
    MODEL_NAMES,
    index=1,
)

selected_model = models[selected_model_name]

st.caption(
    f"Selected model: {selected_model_name}"
)

st.divider()


# ============================================================
# 11. PARTICIPANT INPUT FORM
# ============================================================

st.subheader("Participant information")

st.write(
    "Enter the nine features used by the trained model."
)

with st.form("patient_information_form"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=20,
            max_value=80,
            value=45,
        )

        sex = st.selectbox(
            "Sex",
            ["Female", "Male"],
        )

        income = st.number_input(
            "Family income-to-poverty ratio",
            min_value=0.0,
            value=1.0,
            step=0.1,
        )

        hba1c = st.number_input(
            "HbA1c (%)",
            min_value=0.0,
            value=6.0,
            step=0.1,
        )

        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=0.0,
            value=25.0,
            step=0.1,
        )

    with col2:

        systolic = st.number_input(
            "Systolic blood pressure (mmHg)",
            min_value=0,
            value=120,
        )

        diastolic = st.number_input(
            "Diastolic blood pressure (mmHg)",
            min_value=0,
            value=80,
        )

        smoking = st.selectbox(
            "Smoking status",
            ["never", "former", "current", "unknown"],
        )

        insulin_choice = st.selectbox(
            "Insulin status",
            ["Yes", "No", "Unknown"],
        )

    submitted = st.form_submit_button(
        "Generate prediction",
        use_container_width=True,
    )


# ============================================================
# 12. GENERATE PREDICTION
# ============================================================

if submitted:

    input_row = pd.DataFrame(
        [
            {
                "RIDAGEYR": age,
                "female": 1 if sex == "Female" else 0,
                "INDFMPIR": income,
                "LBXGH": hba1c,
                "BMXBMI": bmi,
                "BPXOSY1": systolic,
                "BPXODI1": diastolic,
                "smoking": smoking,
                "insulin": {
                    "Yes": 1,
                    "No": 0,
                    "Unknown": np.nan,
                }[insulin_choice],
            }
        ]
    )

    input_row = input_row[FEATURE_NAMES]

    try:

        positive_index = list(
            selected_model.classes_
        ).index(1)

        probability = float(
            selected_model.predict_proba(
                input_row
            )[0, positive_index]
        )

        predicted_positive = probability >= THRESHOLD

        st.divider()

        st.header("Prediction result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Estimated probability",
                f"{probability:.1%}",
            )

        with result_col2:

            st.metric(
                "Classification threshold",
                f"{THRESHOLD:.0%}",
            )

        if predicted_positive:

            st.warning(
                "Positive for the study-defined kidney marker."
            )

            st.write(
                "The model classified this input as belonging "
                "to the kidney-marker-positive category."
            )

        else:

            st.success(
                "Negative for the study-defined kidney marker."
            )

            st.write(
                "The model classified this input as belonging "
                "to the kidney-marker-negative category."
            )

        st.caption(
            f"Prediction generated using {selected_model_name}. "
            "The 0.50 threshold is the default and has not been separately tuned."
        )

        st.info(
            "This probability is a model estimate, not a clinical "
            "risk calculation or a confirmed diagnosis."
        )

        # ----------------------------------------------------
        # INDIVIDUAL SHAP EXPLANATION
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Why did the model make this estimate?"
        )

        try:

            waterfall = explain_prediction(
                selected_model_name,
                selected_model,
                input_row,
            )

            shap.plots.waterfall(
                waterfall,
                max_display=10,
                show=False,
            )

            st.pyplot(
                plt.gcf(),
                clear_figure=True,
            )

            plt.close()

            st.caption(
                "Features pushing the model output higher increase "
                "its estimated positive-class output. Features pushing "
                "it lower decrease that output. These are model "
                "contributions, not causes."
            )

        except Exception as error:

            st.warning(
                "The prediction was generated, but the SHAP "
                f"explanation could not be displayed: {error}"
            )

        with st.expander("View submitted model inputs"):

            st.dataframe(
                input_row,
                hide_index=True,
                use_container_width=True,
            )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ============================================================
# 13. HELD-OUT MODEL EVALUATION
# ============================================================

st.divider()

st.header("Model performance evaluation")

st.write(
    "Comparison of Logistic Regression, Random Forest and XGBoost "
    "on the separate 302-participant held-out test set."
)

try:

    X_test, y_test = load_test_data()

    results_df, all_predictions = evaluate_models(
        models,
        X_test,
        y_test,
    )

    display_df = results_df.copy()

    percentage_columns = [
        "Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "F1-score",
    ]

    for column in percentage_columns:

        display_df[column] = display_df[column].map(
            lambda value: f"{value:.1%}"
        )

    for column in ["ROC-AUC", "PR-AUC"]:

        display_df[column] = display_df[column].map(
            lambda value: f"{value:.3f}"
        )

    st.subheader("Comparison across all three models")

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # SELECTED MODEL METRICS
    # --------------------------------------------------------

    selected_predictions = all_predictions[
        selected_model_name
    ]

    test_probabilities = selected_predictions[
        "probabilities"
    ]

    test_predictions = selected_predictions[
        "predictions"
    ]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    st.subheader(
        f"{selected_model_name} – Held-out test performance"
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Accuracy",
        f"{accuracy_score(y_test, test_predictions):.1%}",
    )

    metric2.metric(
        "Precision",
        f"{precision_score(y_test, test_predictions, zero_division=0):.1%}",
    )

    metric3.metric(
        "Sensitivity",
        f"{recall_score(y_test, test_predictions, zero_division=0):.1%}",
    )

    metric4.metric(
        "Specificity",
        f"{specificity:.1%}",
    )

    metric5, metric6, metric7 = st.columns(3)

    metric5.metric(
        "F1-score",
        f"{f1_score(y_test, test_predictions, zero_division=0):.1%}",
    )

    metric6.metric(
        "ROC-AUC",
        f"{roc_auc_score(y_test, test_probabilities):.3f}",
    )

    metric7.metric(
        "PR-AUC",
        f"{average_precision_score(y_test, test_probabilities):.3f}",
    )

    # --------------------------------------------------------
    # EVALUATION PLOTS
    # --------------------------------------------------------

    st.subheader("Performance visualizations")

    figure, axes = plt.subplots(
        1, 3, figsize=(16, 4)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        labels=[0, 1],
        display_labels=["No marker", "Marker"],
        cmap="Blues",
        ax=axes[0],
        colorbar=False,
    )

    axes[0].set_title("Confusion matrix")

    RocCurveDisplay.from_predictions(
        y_test,
        test_probabilities,
        ax=axes[1],
        name=selected_model_name,
    )

    axes[1].set_title("ROC curve")

    PrecisionRecallDisplay.from_predictions(
        y_test,
        test_probabilities,
        ax=axes[2],
        name=selected_model_name,
    )

    axes[2].set_title("Precision-recall curve")

    figure.tight_layout()

    st.pyplot(
        figure,
        clear_figure=True,
    )

    plt.close(figure)

    st.caption(
        "These are unweighted sample-level results from one held-out "
        "split. They are not estimates of national NHANES prevalence "
        "or evidence of clinical utility."
    )

except Exception as error:

    st.warning(
        f"Model evaluation could not be displayed: {error}"
    )


# ============================================================
# 14. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

st.divider()

st.header("Global explainability")

st.write(
    "Global feature importance summarizes how strongly features "
    "contribute to model predictions across observations."
)

if SHAP_PATH.exists():

    try:

        importance = pd.read_csv(SHAP_PATH)

        required_columns = {
            "feature",
            "mean_absolute_shap",
        }

        if not required_columns.issubset(importance.columns):

            raise ValueError(
                "The SHAP file is missing required columns."
            )

        importance = importance.sort_values(
            "mean_absolute_shap",
            ascending=False,
        ).head(10)

        st.subheader(
            "Global SHAP feature importance – Random Forest"
        )

        chart_data = importance.set_index("feature")

        st.bar_chart(
            chart_data["mean_absolute_shap"],
            horizontal=True,
        )

        st.caption(
            "This is the saved Random Forest global SHAP analysis. "
            "It is not automatically recalculated when another model "
            "is selected."
        )

    except Exception as error:

        st.warning(
            f"Global SHAP importance could not be displayed: {error}"
        )

else:

    st.info(
        "The saved global SHAP report was not found. "
        "Individual explanations can still be generated."
    )


# ============================================================
# 15. ABOUT THE PROJECT
# ============================================================

st.divider()

st.header("About Renal AI")

st.write(
    """
    Renal AI is an explainable machine learning research prototype
    developed to identify study-defined kidney abnormality markers
    among adults with diabetes.

    The project uses publicly available NHANES 2017–March 2020
    pre-pandemic cycle data.

    The kidney-marker outcome is defined as either:

    - eGFR below 60 mL/min/1.73 m², OR
    - UACR of at least 30 mg/g.

    The models use nine demographic and clinical predictors.
    Outcome-defining kidney measurements are excluded from the
    predictor variables to prevent target leakage.
    """
)


# ============================================================
# 16. LIMITATIONS AND DISCLAIMER
# ============================================================

st.divider()

st.header("Limitations and responsible use")

st.warning(
    """
    This application is an academic research and educational prototype,
    not a medical diagnostic tool.

    It does not replace laboratory tests, clinical assessment,
    or advice from qualified healthcare professionals.

    The displayed probability is a model estimate for an existing
    cross-sectional study label. It is not a diagnosis and does not
    predict future kidney disease.

    A single abnormal eGFR or UACR measurement does not establish
    chronic kidney disease, which requires evidence of persistence.

    The model has not been clinically validated. Test-set results
    are unweighted, and NHANES survey weights were not used to
    estimate national prevalence.
    """
)

st.caption(
    "Renal AI | F6 Capstone Project | Group H"
)

