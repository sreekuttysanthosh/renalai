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


st.set_page_config(page_title="Renal AI", page_icon="🩺")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_pipeline.joblib"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_test.csv"
THRESHOLD = 0.5  # The project metadata says this is the default, untuned threshold.


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_shap_explainer():
    model = load_model()
    return shap.TreeExplainer(model.named_steps["classifier"])


@st.cache_data
def load_test_data():
    test_data = pd.read_csv(TEST_DATA_PATH)
    required_columns = set(load_model().feature_names_in_) | {"SEQN", "kidney_marker"}
    missing_columns = required_columns.difference(test_data.columns)
    if missing_columns:
        raise ValueError(
            "The held-out test data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if len(test_data) != 302:
        raise ValueError(f"Expected 302 held-out participants, found {len(test_data)}.")
    if test_data["SEQN"].duplicated().any():
        raise ValueError("The held-out test data contains duplicate participant IDs.")
    if test_data["kidney_marker"].isna().any():
        raise ValueError("The held-out test data contains missing target values.")

    features = list(load_model().feature_names_in_)
    return test_data[features], test_data["kidney_marker"].astype(int)


@st.cache_data
def load_model_comparison():
    comparison = pd.read_csv(PROJECT_ROOT / "reports" / "model_comparison.csv")
    expected_models = {"Logistic Regression", "Random Forest", "XGBoost"}
    if set(comparison["Model"]) != expected_models:
        raise ValueError("The model comparison report does not contain all three models.")
    return comparison


@st.cache_data
def load_global_shap_importance():
    importance = pd.read_csv(PROJECT_ROOT / "reports" / "shap_global_importance.csv")
    if not {"feature", "mean_absolute_shap"}.issubset(importance.columns):
        raise ValueError("The global SHAP report is missing required columns.")
    return importance


st.title("Renal AI")
st.write("A student project that explores kidney-related screening markers.")
st.caption(
    "Research and educational demonstration only. This model identifies an existing "
    "study-defined marker; it does not predict future disease or diagnose kidney disease."
)

if MODEL_PATH.exists():
    model = load_model()
else:
    model = None
    st.error(
        "The Random Forest pipeline was not found. Check that "
        "models/random_forest_pipeline.joblib exists in the project folder."
    )

st.header("Patient information")
st.write("Enter the nine features used by the project model.")

age = st.number_input("Age in years", min_value=20, max_value=80, value=45)
sex = st.selectbox("Sex", ["Female", "Male"])
income = st.number_input("Family income-to-poverty ratio", min_value=0.0, value=1.0)
hba1c = st.number_input("HbA1c", min_value=0.0, value=6.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
systolic = st.number_input("Systolic blood pressure", min_value=0, value=120)
diastolic = st.number_input("Diastolic blood pressure", min_value=0, value=80)
smoking = st.selectbox("Smoking status", ["never", "former", "current", "unknown"])
insulin_choice = st.selectbox("Insulin status", ["Yes", "No", "Unknown"])

if st.button("Predict", disabled=model is None):
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
                "insulin": {"Yes": 1, "No": 0, "Unknown": np.nan}[insulin_choice],
            }
        ]
    )

    try:
        positive_class_index = list(model.classes_).index(1)
        positive_probability = float(
            model.predict_proba(input_row)[0, positive_class_index]
        )
        predicted_positive = positive_probability >= THRESHOLD

        st.subheader("Project model result")
        st.metric("Estimated probability for the study-defined kidney marker", f"{positive_probability:.1%}")
        st.write(
            "Model classification: **positive** for the study-defined kidney marker."
            if predicted_positive
            else "Model classification: **negative** for the study-defined kidney marker."
        )
        st.caption(f"Classification threshold: {THRESHOLD:.0%} (default; not separately tuned).")

        st.subheader("Why did the model make this estimate?")
        try:
            preprocessor = model.named_steps["preprocessor"]
            transformed_input = preprocessor.transform(input_row)
            if hasattr(transformed_input, "toarray"):
                transformed_input = transformed_input.toarray()

            shap_explanation = load_shap_explainer()(transformed_input)
            shap_values = shap_explanation.values
            base_values = np.asarray(shap_explanation.base_values)

            if shap_values.ndim == 3:
                local_values = shap_values[0, :, positive_class_index]
                local_base_value = base_values.reshape(-1, len(model.classes_))[
                    0, positive_class_index
                ]
            else:
                local_values = shap_values[0]
                local_base_value = base_values.reshape(-1)[0]

            waterfall = shap.Explanation(
                values=local_values,
                base_values=local_base_value,
                data=np.asarray(transformed_input)[0],
                feature_names=preprocessor.get_feature_names_out(),
            )
            shap.plots.waterfall(waterfall, max_display=10, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close()
            st.caption(
                "Features pushing right increase the model output for the study-defined "
                "kidney marker; features pushing left decrease it. These are model "
                "contributions, not causes."
            )
        except (
            ValueError,
            TypeError,
            IndexError,
            KeyError,
            AttributeError,
            RuntimeError,
        ) as error:
            st.warning(f"The prediction worked, but its SHAP explanation could not be created: {error}")

        with st.expander("See the model inputs"):
            st.dataframe(input_row, hide_index=True)
    except (ValueError, TypeError, IndexError, KeyError) as error:
        st.error(f"Prediction failed: {error}")

st.header("Held-out model evaluation")
st.write(
    "The following results evaluate the saved Random Forest pipeline on the separate "
    "302-participant test set. The 0.50 threshold is the project's default and was not "
    "tuned on this test set."
)
try:
    comparison = load_model_comparison()
    st.subheader("Held-out comparison across all models")
    comparison_display = comparison[
        [
            "Model",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_specificity",
            "test_f1",
            "test_roc_auc",
            "test_pr_auc",
        ]
    ].rename(
        columns={
            "test_accuracy": "Accuracy",
            "test_precision": "Precision",
            "test_recall": "Sensitivity",
            "test_specificity": "Specificity",
            "test_f1": "F1",
            "test_roc_auc": "ROC-AUC",
            "test_pr_auc": "PR-AUC",
        }
    )
    for column in ("Accuracy", "Precision", "Sensitivity", "Specificity", "F1"):
        comparison_display[column] = comparison_display[column].map(
            lambda value: f"{value:.1%}"
        )
    for column in ("ROC-AUC", "PR-AUC"):
        comparison_display[column] = comparison_display[column].map(
            lambda value: f"{value:.3f}"
        )
    st.dataframe(comparison_display, hide_index=True)

    test_features, test_target = load_test_data()
    positive_class_index = list(model.classes_).index(1)
    test_probabilities = model.predict_proba(test_features)[:, positive_class_index]
    test_predictions = (test_probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        test_target, test_predictions, labels=[0, 1]
    ).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0

    metric_columns = st.columns(4)
    metric_columns[0].metric("Accuracy", f"{accuracy_score(test_target, test_predictions):.1%}")
    metric_columns[1].metric(
        "Precision", f"{precision_score(test_target, test_predictions, zero_division=0):.1%}"
    )
    metric_columns[2].metric(
        "Sensitivity / recall",
        f"{recall_score(test_target, test_predictions, zero_division=0):.1%}",
    )
    metric_columns[3].metric("Specificity", f"{specificity:.1%}")

    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "F1-score", f"{f1_score(test_target, test_predictions, zero_division=0):.1%}"
    )
    metric_columns[1].metric("ROC-AUC", f"{roc_auc_score(test_target, test_probabilities):.3f}")
    metric_columns[2].metric(
        "PR-AUC", f"{average_precision_score(test_target, test_probabilities):.3f}"
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    ConfusionMatrixDisplay.from_predictions(
        test_target,
        test_predictions,
        labels=[0, 1],
        display_labels=["No marker", "Marker"],
        cmap="Blues",
        ax=axes[0],
        colorbar=False,
    )
    axes[0].set_title("Confusion matrix")
    RocCurveDisplay.from_predictions(
        test_target, test_probabilities, ax=axes[1], name="Random Forest"
    )
    axes[1].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(
        test_target, test_probabilities, ax=axes[2], name="Random Forest"
    )
    axes[2].set_title("Precision-recall curve")
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
    st.caption(
        "These are unweighted sample-level results from one held-out split, not estimates "
        "of national NHANES prevalence or evidence of clinical utility."
    )
    st.subheader("Global SHAP feature importance")
    global_importance = load_global_shap_importance().head(10).set_index("feature")
    st.bar_chart(global_importance["mean_absolute_shap"])
except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Held-out evaluation could not be displayed: {error}")

st.header("About")
st.write(
    "The study cohort includes adults with diabetes from NHANES 2017–March 2020. "
    "The study label is eGFR below 60 mL/min/1.73 m² or UACR at least 30 mg/g. "
    "Outcome-defining kidney measurements are excluded from model predictors."
)

st.header("Limitations")
st.warning(
    "This is an academic project, not a medical diagnostic tool. "
    "It does not replace laboratory tests or advice from healthcare professionals. "
    "The displayed probability is a model estimate for a cross-sectional study label, "
    "not a diagnosis or a prediction of future disease. A single abnormal measurement "
    "does not confirm chronic kidney disease. The test-set metrics are unweighted, "
    "and NHANES survey weights were not used to estimate national prevalence."
)
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


st.set_page_config(page_title="Renal AI", page_icon="🩺")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_pipeline.joblib"
THRESHOLD = 0.5  # The project metadata says this is the default, untuned threshold.


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_shap_explainer():
    model = load_model()
    return shap.TreeExplainer(model.named_steps["classifier"])


st.title("Renal AI")
st.write("A student project that explores kidney-related screening markers.")

if MODEL_PATH.exists():
    model = load_model()
else:
    model = None
    st.error(
        "The Random Forest pipeline was not found. Check that "
        "models/random_forest_pipeline.joblib exists in the project folder."
    )

st.header("Patient information")
st.write("Enter the nine features used by the project model.")

age = st.number_input("Age in years", min_value=20, max_value=80, value=45)
sex = st.selectbox("Sex", ["Female", "Male"])
income = st.number_input("Family income-to-poverty ratio", min_value=0.0, value=1.0)
hba1c = st.number_input("HbA1c", min_value=0.0, value=6.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
systolic = st.number_input("Systolic blood pressure", min_value=0, value=120)
diastolic = st.number_input("Diastolic blood pressure", min_value=0, value=80)
smoking = st.selectbox("Smoking status", ["never", "former", "current", "unknown"])
insulin_choice = st.selectbox("Insulin status", ["Yes", "No", "Unknown"])

if st.button("Predict", disabled=model is None):
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
                "insulin": {"Yes": 1, "No": 0, "Unknown": np.nan}[insulin_choice],
            }
        ]
    )

    try:
        positive_class_index = list(model.classes_).index(1)
        positive_probability = float(
            model.predict_proba(input_row)[0, positive_class_index]
        )
        predicted_positive = positive_probability >= THRESHOLD

        st.subheader("Project model result")
        st.metric("Estimated probability for the study-defined kidney marker", f"{positive_probability:.1%}")
        st.write(
            "Model classification: **positive** for the study-defined kidney marker."
            if predicted_positive
            else "Model classification: **negative** for the study-defined kidney marker."
        )
        st.caption(f"Classification threshold: {THRESHOLD:.0%} (default; not separately tuned).")

        st.subheader("Why did the model make this estimate?")
        try:
            preprocessor = model.named_steps["preprocessor"]
            transformed_input = preprocessor.transform(input_row)
            if hasattr(transformed_input, "toarray"):
                transformed_input = transformed_input.toarray()

            shap_explanation = load_shap_explainer()(transformed_input)
            shap_values = shap_explanation.values
            base_values = np.asarray(shap_explanation.base_values)

            if shap_values.ndim == 3:
                local_values = shap_values[0, :, positive_class_index]
                local_base_value = base_values.reshape(-1, len(model.classes_))[
                    0, positive_class_index
                ]
            else:
                local_values = shap_values[0]
                local_base_value = base_values.reshape(-1)[0]

            waterfall = shap.Explanation(
                values=local_values,
                base_values=local_base_value,
                data=np.asarray(transformed_input)[0],
                feature_names=preprocessor.get_feature_names_out(),
            )
            shap.plots.waterfall(waterfall, max_display=10, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close()
            st.caption(
                "Features pushing right increase the model output for the study-defined "
                "kidney marker; features pushing left decrease it. These are model "
                "contributions, not causes."
            )
        except Exception as error:
            st.warning(f"The prediction worked, but its SHAP explanation could not be created: {error}")

        with st.expander("See the model inputs"):
            st.dataframe(input_row, hide_index=True)
    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.header("About")
st.write("This project uses NHANES data for academic analysis.")

st.header("Limitations")
st.warning(
    "This is an academic project, not a medical diagnostic tool. "
    "It does not replace laboratory tests or advice from healthcare professionals. "
    "The displayed probability is a model estimate for the study label, not a diagnosis."
)
