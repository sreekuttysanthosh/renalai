
    try:

        shap_report = pd.read_csv(
            shap_report_path
        )

        required = {
            "feature",
            "mean_absolute_shap"
        }

        if not required.issubset(
            shap_report.columns
        ):

            raise ValueError(
                "Expected columns: feature and mean_absolute_shap."
            )

        shap_report = shap_report.sort_values(
            "mean_absolute_shap",
            ascending=True
        ).tail(10)

        st.subheader(
            "Random Forest – Global SHAP importance"
        )

        chart_data = shap_report.set_index(
            "feature"
        )

        st.bar_chart(
            chart_data["mean_absolute_shap"],
            horizontal=True
        )

        st.caption(
            "This saved global SHAP analysis belongs to Random Forest. "
            "It does not change when a different prediction model is selected."
        )

    except Exception as error:

        st.warning(
            f"Global SHAP chart could not be displayed: {error}"
        )

else:

    st.info(
        "The global SHAP summary file is not available in the "
        "repository. Individual SHAP explanations can still "
        "be generated for supported models."
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
    Age, sex, HbA1c, BMI, systolic blood pressure,
    diastolic blood pressure, smoking status and insulin status.
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
