from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Renal AI | Kidney Marker Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. RENAL AI THEME
# No custom HTML is used in this application.
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

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #DCEBE8;
        padding: 18px;
        border-radius: 12px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #16877C;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        min-height: 42px;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #116A62;
        color: white;
        border: none;
    }

    [data-testid="stExpander"] {
        background-color: white;
        border: 1px solid #DCEBE8;
        border-radius: 10px;
    }

    hr {
        border-color: #DCEBE8;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "all_models.joblib"

THRESHOLD = 0.50

FEATURE_NAMES = [
    "RIDAGEYR",
    "female",
    "INDFMPIR",
    "LBXGH",
    "BMXBMI",
    "BPXOSY1",
    "BPXODI1",
    "smoking",
    "insulin"
]

MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]


# ============================================================
# 4. LOAD ALL THREE SAVED MODEL PIPELINES
# ============================================================

@st.cache_resource
def load_models():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    saved_models = joblib.load(MODEL_PATH)

    if not isinstance(saved_models, dict):
        raise ValueError(
            "all_models.joblib must contain a dictionary "
            "of the three fitted model pipelines."
        )

    missing_models = [
        name for name in MODEL_NAMES
        if name not in saved_models
    ]

    if missing_models:
        raise ValueError(
            "Missing models: " + ", ".join(missing_models)
        )

    for name in MODEL_NAMES:

        model = saved_models[name]

        if not hasattr(model, "named_steps"):
            raise ValueError(
                f"{name} is not a complete preprocessing pipeline."
            )

        if not hasattr(model, "predict_proba"):
            raise ValueError(
                f"{name} does not support probability prediction."
            )

        if list(model.feature_names_in_) != FEATURE_NAMES:
            raise ValueError(
                f"{name} does not expect the correct nine features."
            )

    return saved_models


try:
    models = load_models()

except Exception as error:

    st.error(f"Could not load the model pipelines: {error}")

    st.info(
        "Check that models/all_models.joblib exists and contains "
        "the three verified fitted pipelines."
    )

    st.stop()


# ============================================================
# 5. HEADER
# Native Streamlit components prevent HTML from appearing
# as visible code in the header.
# ============================================================

st.title("🩺 Renal AI")

st.subheader(
    "Explainable Machine Learning for Kidney Marker Screening"
)

st.caption("NHANES 2017–March 2020")

st.info(
    "Renal AI explores the use of machine learning to identify "
    "existing study-defined kidney abnormality markers among "
    "adults with diabetes. It is an academic research prototype, "
    "not a medical diagnostic tool."
)

st.divider()


# ============================================================
# 6. MODEL SELECTION
# ============================================================

st.header("Prediction workspace")

st.write(
    "Choose one of the three trained classification models "
    "to generate a prediction."
)

selected_model_name = st.selectbox(
    "Select machine learning model",
    MODEL_NAMES,
    index=1
)

selected_model = models[selected_model_name]

st.caption(
    f"Selected model: {selected_model_name}"
)

st.divider()


# ============================================================
# 7. PATIENT INFORMATION FORM
# ============================================================

st.header("Participant information")

st.write(
    "Enter the nine demographic and clinical features "
    "used by the trained model."
)

with st.form("renal_ai_input_form"):

    left, right = st.columns(2)

    with left:

        age = st.number_input(
            "Age (years)",
            min_value=20,
            max_value=80,
            value=45,
            step=1
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        income = st.number_input(
            "Family income-to-poverty ratio",
            min_value=0.0,
            value=1.02,
            step=0.1
        )

        hba1c = st.number_input(
            "HbA1c (%)",
            min_value=0.0,
            value=6.01,
            step=0.1
        )

        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=0.0,
            value=25.01,
            step=0.1
        )

    with right:

        systolic = st.number_input(
            "Systolic blood pressure (mmHg)",
            min_value=0,
            value=120,
            step=1
        )

        diastolic = st.number_input(
            "Diastolic blood pressure (mmHg)",
            min_value=0,
            value=80,
            step=1
        )

        smoking = st.selectbox(
            "Smoking status",
            ["never", "former", "current", "unknown"]
        )

        insulin_choice = st.selectbox(
            "Insulin status",
            ["Yes", "No", "Unknown"]
        )

    submitted = st.form_submit_button(
        "Generate prediction",
        use_container_width=True
    )


# ============================================================
# 8. CREATE MODEL INPUT
# ============================================================

if submitted:

    insulin_value = {
        "Yes": 1,
        "No": 0,
        "Unknown": np.nan
    }[insulin_choice]

    input_data = pd.DataFrame(
        [{
            "RIDAGEYR": age,
            "female": 1 if sex == "Female" else 0,
            "INDFMPIR": income,
            "LBXGH": hba1c,
            "BMXBMI": bmi,
            "BPXOSY1": systolic,
            "BPXODI1": diastolic,
            "smoking": smoking,
            "insulin": insulin_value
        }]
    )

    input_data = input_data[FEATURE_NAMES]


    # ========================================================
    # 9. GENERATE PREDICTION
    # ========================================================

    try:

        positive_class_index = list(
            selected_model.classes_
        ).index(1)

        probability = float(
            selected_model.predict_proba(
                input_data
            )[0, positive_class_index]
        )

        prediction = int(
            probability >= THRESHOLD
        )

        st.divider()

        st.header("Prediction result")

        result_left, result_right = st.columns(2)

        with result_left:

            st.metric(
                "Estimated probability",
                f"{probability * 100:.1f}%"
            )

        with result_right:

            st.metric(
                "Classification threshold",
                f"{THRESHOLD * 100:.0f}%"
            )

        if prediction == 1:

            st.warning(
                "Positive for the study-defined kidney marker."
            )

            st.write(
                "The model assigned this input to the "
                "kidney-marker-positive category."
            )

        else:

            st.success(
                "Negative for the study-defined kidney marker."
            )

            st.write(
                "The model assigned this input to the "
                "kidney-marker-negative category."
            )

        st.caption(
            f"Prediction generated using {selected_model_name}. "
            "The 0.50 threshold is the default and has not "
            "been separately tuned."
        )

        st.info(
            "The probability is a model estimate based on "
            "patterns learned from NHANES training data. "
            "It is not a confirmed clinical diagnosis."
        )


        # ====================================================
        # 10. DISPLAY INPUT VALUES
        # ====================================================

        with st.expander("View submitted model inputs"):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # 11. INDIVIDUAL SHAP EXPLANATION
        # ====================================================

        st.divider()

        st.header("Why did the model make this estimate?")

        st.write(
            "SHAP explains how input features contribute "
            "to the selected model's output."
        )

        try:

            preprocessor = (
                selected_model.named_steps["preprocessor"]
            )

            classifier = (
                selected_model.named_steps["classifier"]
            )

            transformed_input = preprocessor.transform(
                input_data
            )

            if hasattr(transformed_input, "toarray"):

                transformed_input = (
                    transformed_input.toarray()
                )

            feature_names = (
                preprocessor.get_feature_names_out()
            )

            if selected_model_name in [
                "Random Forest",
                "XGBoost"
            ]:

                explainer = shap.TreeExplainer(
                    classifier
                )

                shap_values = explainer(
                    transformed_input
                )

            else:

                training_path = (
                    PROJECT_ROOT
                    / "data"
                    / "processed"
                    / "model_train.csv"
                )

                if not training_path.exists():

                    st.warning(
                        "The Logistic Regression SHAP explanation "
                        "needs the training background data. "
                        "The prediction itself was generated successfully."
                    )

                    st.stop()

                training_data = pd.read_csv(
                    training_path
                )[FEATURE_NAMES]

                background = (
                    preprocessor.transform(training_data)
                )

                if hasattr(background, "toarray"):

                    background = background.toarray()

                explainer = shap.LinearExplainer(
                    classifier,
                    background
                )

                shap_values = explainer(
                    transformed_input
                )

            values = np.asarray(
                shap_values.values
            )

            base_values = np.asarray(
                shap_values.base_values
            )

            class_index = positive_class_index

            if values.ndim == 3:

                local_values = values[
                    0, :, class_index
                ]

                local_base = base_values.reshape(
                    -1, len(selected_model.classes_)
                )[0, class_index]

            else:

                local_values = values[0]

                local_base = base_values.reshape(-1)[0]

            waterfall = shap.Explanation(
                values=local_values,
                base_values=local_base,
                data=np.asarray(transformed_input)[0],
                feature_names=feature_names
            )

            shap.plots.waterfall(
                waterfall,
                max_display=10,
                show=False
            )

            st.pyplot(
                plt.gcf(),
                clear_figure=True
            )

            plt.close()

            st.caption(
                "Features pushing the model output higher increase "
                "the estimated positive-class output. Features "
                "pushing it lower decrease that output. These are "
                "model contributions, not causes."
            )

        except Exception as shap_error:

            st.warning(
                "The prediction was generated, but its SHAP "
                f"explanation could not be displayed: {shap_error}"
            )

    except Exception as prediction_error:

        st.error(
            f"Prediction failed: {prediction_error}"
        )


# ============================================================
# 12. MODEL COMPARISON
# Values below are from the evaluation table shown
# in your latest dashboard screenshot.
# Verify against the final notebook before publication.
# ============================================================

st.divider()

st.header("Held-out model evaluation")

st.write(
    "Comparison of the three models on the separate "
    "302-participant held-out test set."
)

comparison_data = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "Accuracy": [
            "66.9%",
            "66.9%",
            "66.9%"
        ],
        "Precision": [
            "63.7%",
            "61.7%",
            "64.0%"
        ],
        "Sensitivity": [
            "55.0%",
            "62.6%",
            "54.2%"
        ],
        "Specificity": [
            "76.0%",
            "70.2%",
            "76.6%"
        ],
        "F1-score": [
            "59.0%",
            "62.1%",
            "58.7%"
        ],
        "ROC-AUC": [
            "0.730",
            "0.718",
            "0.727"
        ],
        "PR-AUC": [
            "0.700",
            "0.684",
            "0.717"
        ]
    }
)

st.dataframe(
    comparison_data,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Reported held-out test metrics. These are unweighted "
    "sample-level results, not estimates of national NHANES "
    "prevalence or evidence of clinical utility."
)



# ============================================================
# 14. ABOUT THE PROJECT
# ============================================================

st.divider()

st.header("About the project")

about_left, about_right = st.columns(2)

with about_left:

    st.subheader("Study population")

    st.write(
        """
        Adults aged 20 years or older meeting the project's
        operational diabetes definition, using NHANES
        2017–March 2020 pre-pandemic cycle data.
        """
    )

with about_right:

    st.subheader("Kidney-marker definition")

    st.write(
        """
        Positive kidney marker:

        - eGFR below 60 mL/min/1.73 m², OR
        - UACR of at least 30 mg/g.
        """
    )

st.subheader("Model predictors")

st.write(
    """
    Age, sex, family income-to-poverty ratio, HbA1c, BMI,
    systolic blood pressure, diastolic blood pressure,
    smoking status and insulin status.
    """
)


# ============================================================
# 15. LIMITATIONS AND DISCLAIMER
# ============================================================

st.divider()

st.header("Limitations and responsible use")

st.warning(
    """
    This application is an academic research and educational
    prototype, not a medical diagnostic tool.

    It does not replace laboratory tests, clinical assessment
    or advice from qualified healthcare professionals.

    The displayed probability represents a model estimate for
    an existing cross-sectional study label. It does not
    diagnose chronic kidney disease or predict future disease.

    A single abnormal eGFR or UACR measurement does not establish
    chronic kidney disease, which requires evidence of persistence.

    The evaluation results are unweighted sample-level results.
    NHANES survey weights were not used to estimate national
    prevalence. The model has not been clinically validated.
    """
)

st.caption(
    "Renal AI | F6 Capstone Project | Group H"
)
