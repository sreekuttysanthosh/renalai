# Model explainability

The Random Forest model is the dashboard's default and SHAP is used for individual explanations. The dashboard's global view computes mean absolute SHAP values and a beeswarm plot over a fixed, reproducible sample of up to 200 training participants. For the refreshed Random Forest, the leading features in that sample were insulin status (0.0583), age (0.0538), HbA1c (0.0438), systolic blood pressure (0.0358), and the insulin missingness indicator (0.0188). These values describe this fitted model's behavior; they are not causal effects or clinical importance rankings.

The dashboard computes a local SHAP explanation for the selected model after a prediction. Positive contributions move the model output toward the positive class and negative contributions move it away. SHAP explanations do not establish why a participant has a marker, and they do not turn the output into a diagnosis.
