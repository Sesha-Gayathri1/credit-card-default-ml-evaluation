import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

EXPECTED_FEATURES = [f"X{i}" for i in range(1, 24)]

TARGET_COLUMN = "Y"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_model(model_path):
    """
    Load a trained machine learning model from a joblib file.
    """
    return joblib.load(model_path)


def validate_features(df):
    """
    Validate that the uploaded dataset contains X1 through X23.
    """

    missing_columns = [
        col for col in EXPECTED_FEATURES
        if col not in df.columns
    ]

    if missing_columns:
        return False, missing_columns

    return True, []


def prepare_features(df):
    """
    Select the 23 required features in the exact order expected
    by the trained models.

    If the target column Y exists, preserve it separately.
    """

    X = df[EXPECTED_FEATURES].copy()

    for column in EXPECTED_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    if X.isnull().any().any():
        invalid_columns = X.columns[
            X.isnull().any()
        ].tolist()

        raise ValueError(
            "Missing or non-numeric values were found in "
            f"these feature columns: {invalid_columns}"
        )

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "The dataset contains infinite or invalid numeric values."
        )

    return X


def calculate_metrics(y_true, y_pred, y_score=None):
    """
    Calculate classification metrics.
    """

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    if y_score is not None:
        try:
            metrics["AUC"] = roc_auc_score(
                y_true, y_score
            )
        except (ValueError, TypeError):
            metrics["AUC"] = np.nan
    else:
        metrics["AUC"] = np.nan

    return metrics


def get_model_score(model, X):
    """
    Return prediction scores for ROC-AUC when supported.

    Uses positive-class probabilities when available.
    Otherwise, uses the decision function when available.
    """

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        if probabilities.shape[1] >= 2:
            return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return None


def display_confusion_matrix(y_true, y_pred, model_name):
    """
    Display a confusion matrix.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=["No Default (0)", "Default (1)"],
        yticklabels=["No Default (0)", "Default (1)"],
        ax=ax
    )

    ax.set_title(
        f"{model_name} — Confusion Matrix"
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("💳 Credit Card Default Prediction")

st.markdown(
    """
    This application predicts whether a credit card customer
    is likely to default on their payment using trained
    machine learning classification models.

    **Available models:** Logistic Regression, Decision Tree,
    KNN, Naive Bayes, and Support Vector Machine.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Configuration")

selected_model_name = st.sidebar.selectbox(
    "Select a Machine Learning Model",
    list(MODEL_FILES.keys())
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Input requirements**

    - CSV format
    - 23 required feature columns: X1–X23
    - Optional target column: Y
    - Numeric feature values
    """
)


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

model_filename = MODEL_FILES[selected_model_name]

model_path = os.path.join(
    MODEL_DIR,
    model_filename
)

st.subheader("Selected Model")

st.write(f"**Model:** {selected_model_name}")

if not os.path.isfile(model_path):
    st.error(
        f"Model file not found: `{model_path}`. "
        "Ensure that the model directory and saved model "
        "files are included in the project."
    )
    st.stop()

try:
    model = load_model(model_path)

except Exception as e:
    st.error(
        f"Unable to load the selected model: {e}"
    )
    st.stop()


# ============================================================
# CSV UPLOAD
# ============================================================

st.subheader("📂 Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV containing X1 through X23",
    type=["csv"],
    help=(
        "The CSV must contain all 23 features used during "
        "model training. The target column Y is optional."
    )
)

if uploaded_file is None:
    st.info(
        "Upload a test CSV to generate customer default predictions."
    )
    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:
    df = pd.read_csv(uploaded_file)

    # Remove accidental spaces from column names.
    df.columns = df.columns.astype(str).str.strip()

except Exception as e:
    st.error(f"Unable to read the uploaded CSV: {e}")
    st.stop()


if df.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()


# ============================================================
# VALIDATE DATASET
# ============================================================

st.subheader("🔎 Dataset Validation")

st.write("Uploaded dataset dimensions:", df.shape)

is_valid, missing_columns = validate_features(df)

if not is_valid:
    st.error(
        "The uploaded CSV is missing required feature columns: "
        f"{missing_columns}"
    )

    st.warning(
        "Your CSV must contain every feature from X1 to X23. "
        "A Revenue column is not required."
    )

    st.stop()


st.success(
    "All 23 required feature columns (X1–X23) are present."
)


# ============================================================
# PREVIEW DATA
# ============================================================

with st.expander("Preview Uploaded Dataset", expanded=True):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.write(
        f"**Rows:** {df.shape[0]} | "
        f"**Columns:** {df.shape[1]}"
    )


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

try:
    X = prepare_features(df)

except ValueError as e:
    st.error(str(e))
    st.stop()


# Check expected model input dimensions when available.
expected_n_features = getattr(
    model,
    "n_features_in_",
    None
)

if expected_n_features is None and hasattr(model, "steps"):
    final_estimator = model.steps[-1][1]

    expected_n_features = getattr(
        final_estimator,
        "n_features_in_",
        None
    )

if (
    expected_n_features is not None
    and expected_n_features != X.shape[1]
):
    st.error(
        f"The selected model expects {expected_n_features} "
        f"features, but the application supplied {X.shape[1]}."
    )
    st.stop()


# ============================================================
# PREDICTION
# ============================================================

st.subheader("🎯 Generate Predictions")

st.write(
    f"Ready to predict defaults for **{len(X)} customers** "
    f"using **{selected_model_name}**."
)

if st.button(
    "Predict Defaults",
    type="primary",
    use_container_width=True
):

    try:
        with st.spinner("Generating predictions..."):

            predictions = model.predict(X)

            prediction_results = df.copy()

            prediction_results["Predicted_Default"] = (
                predictions.astype(int)
            )

            prediction_results["Prediction_Label"] = (
                prediction_results["Predicted_Default"]
                .map({
                    0: "No Default",
                    1: "Default"
                })
                .fillna("Unknown")
            )

            scores = get_model_score(model, X)

            if scores is not None:
                prediction_results[
                    "Default_Score"
                ] = scores

        st.session_state["prediction_results"] = (
            prediction_results
        )

        st.session_state["predictions"] = predictions

        st.session_state["selected_model_name"] = (
            selected_model_name
        )

        st.session_state["prediction_scores"] = scores

        st.success("Predictions generated successfully!")

    except Exception as e:
        st.error(
            f"Prediction failed. Please verify that the uploaded "
            f"features use the same format and encoding as the "
            f"training data. Details: {e}"
        )


# ============================================================
# DISPLAY PREDICTION RESULTS
# ============================================================

if "prediction_results" in st.session_state:

    if (
        st.session_state["selected_model_name"]
        == selected_model_name
    ):

        results = st.session_state["prediction_results"]

        predictions = st.session_state["predictions"]

        scores = st.session_state.get(
            "prediction_scores"
        )

        st.divider()

        st.subheader("📊 Prediction Summary")

        total_customers = len(predictions)

        default_count = int(
            np.sum(predictions == 1)
        )

        non_default_count = int(
            np.sum(predictions == 0)
        )

        default_percentage = (
            default_count / total_customers * 100
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

        col2.metric(
            "Predicted Defaults",
            f"{default_count:,}"
        )

        col3.metric(
            "Predicted Non-Defaults",
            f"{non_default_count:,}"
        )

        col4.metric(
            "Default Rate",
            f"{default_percentage:.2f}%"
        )


        # ----------------------------------------------------
        # RESULTS TABLE
        # ----------------------------------------------------

        st.subheader("Customer-Level Predictions")

        st.dataframe(
            results,
            use_container_width=True,
            height=400
        )


        # ----------------------------------------------------
        # DOWNLOAD RESULTS
        # ----------------------------------------------------

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_data,
            file_name="credit_card_default_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )


        # ----------------------------------------------------
        # PREDICTION DISTRIBUTION
        # ----------------------------------------------------

        st.subheader("Prediction Distribution")

        distribution = pd.Series(
            predictions
        ).value_counts().reindex(
            [0, 1],
            fill_value=0
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            ["No Default", "Default"],
            [
                distribution.loc[0],
                distribution.loc[1]
            ]
        )

        ax.set_ylabel("Number of Customers")
        ax.set_title("Predicted Customer Default Distribution")

        st.pyplot(fig)
        plt.close(fig)


        # ----------------------------------------------------
        # MODEL EVALUATION
        # ----------------------------------------------------

        st.divider()

        st.subheader("📈 Model Evaluation")

        if TARGET_COLUMN in df.columns:

            st.write(
                "The uploaded dataset contains the actual target "
                "column Y. Model evaluation metrics can therefore "
                "be calculated."
            )

            try:
                y_true = pd.to_numeric(
                    df[TARGET_COLUMN],
                    errors="raise"
                ).to_numpy()

                if pd.isna(y_true).any():
                    raise ValueError(
                        "The target column Y contains missing values."
                    )

                if not np.isin(y_true, [0, 1]).all():
                    raise ValueError(
                        "The target column Y must contain only 0 and 1."
                    )

                y_pred = predictions

                metrics = calculate_metrics(
                    y_true,
                    y_pred,
                    scores
                )

                m1, m2, m3 = st.columns(3)
                m4, m5, m6 = st.columns(3)

                m1.metric(
                    "Accuracy",
                    f"{metrics['Accuracy']:.4f}"
                )

                m2.metric(
                    "Precision",
                    f"{metrics['Precision']:.4f}"
                )

                m3.metric(
                    "Recall",
                    f"{metrics['Recall']:.4f}"
                )

                m4.metric(
                    "F1 Score",
                    f"{metrics['F1 Score']:.4f}"
                )

                m5.metric(
                    "AUC",
                    (
                        f"{metrics['AUC']:.4f}"
                        if not np.isnan(metrics["AUC"])
                        else "N/A"
                    )
                )

                m6.metric(
                    "MCC",
                    f"{metrics['MCC']:.4f}"
                )


                # Confusion matrix.
                display_confusion_matrix(
                    y_true,
                    y_pred,
                    selected_model_name
                )


                # Classification report.
                st.subheader("Classification Report")

                report = classification_report(
                    y_true,
                    y_pred,
                    labels=[0, 1],
                    target_names=[
                        "No Default (0)",
                        "Default (1)"
                    ],
                    output_dict=True,
                    zero_division=0
                )

                report_df = pd.DataFrame(
                    report
                ).transpose()

                st.dataframe(
                    report_df.round(4),
                    use_container_width=True
                )


                # Explain important metrics.
                st.info(
                    """
                    **Metric interpretation**

                    - **Accuracy:** Overall proportion of correct predictions.
                    - **Precision:** Of customers predicted to default,
                      the proportion who actually defaulted.
                    - **Recall:** Of actual defaulters, the proportion
                      correctly identified.
                    - **F1 Score:** Harmonic mean of precision and recall.
                    - **AUC:** Ability to rank positive cases above
                      negative cases using prediction scores.
                    - **MCC:** A balanced measure of binary
                      classification performance.
                    """
                )

            except (ValueError, TypeError) as e:
                st.warning(
                    f"Could not calculate evaluation metrics: {e}"
                )

        else:

            st.info(
                """
                The uploaded CSV does not contain the actual target
                column Y.

                Predictions are available, but accuracy, precision,
                recall, F1 score, AUC, MCC, and the confusion matrix
                cannot be calculated without known actual labels.
                """
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Card Default Prediction | "
    "Machine Learning Classification Project"
)
