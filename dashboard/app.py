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
# 13. ABOUT THE PROJECT
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
# 14. LIMITATIONS AND DISCLAIMER
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
