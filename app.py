import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Online Shoppers Purchase Prediction",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [f"X{i}" for i in range(1, 24)]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

# =========================================================
# 2. CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        div[data-testid="stMetric"] {
            background-color: #1e293b;
            border: 1px solid #334155;
            padding: 18px;
            border-radius: 12px;
        }

        h1, h2, h3 {
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model(model_path, modified_time):
    return joblib.load(model_path)


def find_model_file(filename):
    """
    Support both 'model' and 'models' directories.
    """

    possible_paths = [
        os.path.join("model", filename),
        os.path.join("models", filename),
    ]

    for path in possible_paths:
        if os.path.isfile(path):
            return path

    return None


def get_selected_model(model_name):
    filename = MODEL_FILES[model_name]
    model_path = find_model_file(filename)

    if model_path is None:
        st.error(
            f"Model file not found: {filename}. "
            "Check whether the file exists in the model/ "
            "or models/ directory."
        )
        return None

    try:
        modified_time = os.path.getmtime(model_path)

        model = load_model(
            model_path,
            modified_time,
        )

        return model

    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        return None


# =========================================================
# 4. DATASET VALIDATION
# =========================================================

def validate_dataset(df):

    if df.empty:
        return False, "The uploaded CSV is empty."

    # Clean column names.
    df.columns = df.columns.astype(str).str.strip()

    # Check for X1-X23.
    missing_columns = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing_columns:
        return (
            False,
            f"Missing required feature columns: {missing_columns}",
        )

    # Convert feature values to numeric.
    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Reject missing or non-numeric values.
    if df[FEATURES].isna().any().any():

        invalid_columns = (
            df[FEATURES]
            .columns[df[FEATURES].isna().any()]
            .tolist()
        )

        return (
            False,
            f"Missing or non-numeric values in: {invalid_columns}",
        )

    # Reject infinite values.
    if not np.isfinite(
        df[FEATURES].to_numpy(dtype=float)
    ).all():

        return (
            False,
            "Feature columns contain infinite values.",
        )

    # Validate optional target column.
    if "Y" in df.columns:

        if df["Y"].isna().any():
            return False, "The Y column contains missing values."

        normalized_y = df["Y"].astype(str).str.strip().str.lower()

        valid_labels = {
            "0", "1",
            "0.0", "1.0",
            "true", "false",
            "yes", "no",
            "purchase", "no purchase",
        }

        if not normalized_y.isin(valid_labels).all():
            return (
                False,
                "The Y column must contain binary labels such as "
                "0/1, True/False, or Purchase/No Purchase.",
            )

    return True, "Dataset validation successful."


def normalize_target(series):
    """
    Convert common binary target formats to 0 and 1.

    0 = No Purchase
    1 = Purchase
    """

    mapping = {
        "0": 0,
        "0.0": 0,
        "false": 0,
        "no": 0,
        "no purchase": 0,
        "1": 1,
        "1.0": 1,
        "true": 1,
        "yes": 1,
        "purchase": 1,
    }

    normalized = series.astype(str).str.strip().str.lower()

    return normalized.map(mapping).astype(int)


# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Configuration")

    selected_model_name = st.selectbox(
        "Select a Machine Learning Model",
        options=list(MODEL_FILES.keys()),
        index=1,
    )

    st.divider()

    st.markdown(
        """
        ### 📋 Input Requirements

        - CSV format
        - 23 feature columns: X1-X23
        - Optional target column: Y
        - Numeric feature values
        - Target labels: 0/1 or equivalent binary labels

        ### 🤖 Available Models

        - Logistic Regression
        - Decision Tree
        - KNN
        - Naive Bayes
        - Support Vector Machine
        """
    )

# =========================================================
# 6. MAIN APPLICATION
# =========================================================

st.title("🛍️ Online Shoppers Purchase Prediction")

st.write(
    "Upload a test dataset, select a machine learning model, "
    "generate customer-level predictions, and evaluate model "
    "performance when actual target labels are available."
)

st.markdown(
    f"**Selected Model:** {selected_model_name}"
)

# ---------------------------------------------------------
# Upload CSV
# ---------------------------------------------------------

st.subheader("📂 Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV containing X1 through X23",
    type=["csv"],
)

if uploaded_file is None:

    st.info(
        "Upload your test CSV to begin prediction and evaluation."
    )

    st.stop()

# ---------------------------------------------------------
# Read CSV
# ---------------------------------------------------------

try:

    df = pd.read_csv(uploaded_file)

except Exception as exc:

    st.error(f"Unable to read the uploaded CSV: {exc}")
    st.stop()

# ---------------------------------------------------------
# Validate Dataset
# ---------------------------------------------------------

st.subheader("🔎 Dataset Validation")

st.write(
    f"**Uploaded dataset dimensions:** "
    f"`{df.shape}`"
)

valid, validation_message = validate_dataset(df)

if not valid:

    st.error(validation_message)
    st.stop()

st.success(
    "All 23 required feature columns (X1-X23) are present "
    "and valid."
)

if "Y" in df.columns:

    st.success(
        "Target column Y detected. Model evaluation is available."
    )

else:

    st.info(
        "No Y column detected. The app will generate predictions, "
        "but evaluation metrics cannot be calculated without "
        "actual labels."
    )

# ---------------------------------------------------------
# Preview Data
# ---------------------------------------------------------

st.subheader("📊 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True,
)

with st.expander("View dataset statistics"):

    st.write(df[FEATURES].describe().T)

# =========================================================
# 7. LOAD MODEL AND GENERATE PREDICTIONS
# =========================================================

st.subheader("🤖 Customer-Level Predictions")

if st.button(
    "Predict Purchases",
    type="primary",
    use_container_width=False,
):

    model = get_selected_model(selected_model_name)

    if model is None:
        st.stop()

    X = df[FEATURES].copy()

    try:

        # Generate predictions.
        predictions = np.asarray(
            model.predict(X)
        ).reshape(-1)

        if len(predictions) != len(df):
            st.error(
                "The model returned a different number of "
                "predictions than the number of uploaded rows."
            )
            st.stop()

        # Convert predicted labels to 0 and 1.
        predicted_labels = normalize_target(
            pd.Series(predictions)
        ).to_numpy()

        # Get prediction probabilities when supported.
        probabilities = None

        if hasattr(model, "predict_proba"):

            try:

                probabilities = model.predict_proba(X)

                # Locate the class corresponding to label 1.
                if hasattr(model, "classes_"):

                    classes = list(model.classes_)

                    positive_class_index = None

                    for index, label in enumerate(classes):

                        normalized_label = str(label).strip().lower()

                        if normalized_label in {
                            "1", "1.0", "true", "yes", "purchase"
                        }:
                            positive_class_index = index
                            break

                    if positive_class_index is not None:

                        probabilities = probabilities[
                            :, positive_class_index
                        ]

                    else:
                        probabilities = None

                else:
                    probabilities = None

            except Exception:
                probabilities = None

        # Save predictions.
        results = df.copy()

        results["Predicted_Y"] = predicted_labels

        results["Predicted_Purchase"] = (
            results["Predicted_Y"]
            .map({
                0: "No Purchase",
                1: "Purchase",
            })
        )

        if probabilities is not None:

            results["Purchase_Probability"] = probabilities

        st.session_state["prediction_results"] = results
        st.session_state["predicted_labels"] = predicted_labels
        st.session_state["selected_model_name"] = selected_model_name

        # -------------------------------------------------
        # SUMMARY METRICS
        # -------------------------------------------------

        total_customers = len(results)

        purchase_count = int(
            (predicted_labels == 1).sum()
        )

        no_purchase_count = int(
            (predicted_labels == 0).sum()
        )

        purchase_percentage = (
            purchase_count / total_customers * 100
            if total_customers else 0
        )

        st.success("Predictions generated successfully.")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Customers",
            f"{total_customers:,}",
        )

        col2.metric(
            "Predicted Purchases",
            f"{purchase_count:,}",
        )

        col3.metric(
            "Predicted No Purchases",
            f"{no_purchase_count:,}",
        )

        col4.metric(
            "Purchase Rate",
            f"{purchase_percentage:.2f}%",
        )

        # -------------------------------------------------
        # CUSTOMER-LEVEL PREDICTIONS TABLE
        # -------------------------------------------------

        st.subheader("Customer-Level Predictions")

        st.dataframe(
            results,
            use_container_width=True,
            height=450,
        )

        # -------------------------------------------------
        # DOWNLOAD PREDICTIONS
        # -------------------------------------------------

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions CSV",
            data=csv_data,
            file_name="purchase_predictions.csv",
            mime="text/csv",
        )

        # =================================================
        # 8. MODEL EVALUATION
        # =================================================

        st.divider()

        st.header("📈 Model Evaluation")

        if "Y" not in df.columns:

            st.warning(
                "Evaluation metrics are unavailable because "
                "the uploaded CSV does not contain the actual "
                "target column Y. Upload a labeled test dataset "
                "to view the confusion matrix and classification "
                "report."
            )

        else:

            y_true = normalize_target(df["Y"])

            y_pred = predicted_labels

            # Ensure both actual classes are represented
            # in the confusion matrix.
            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            )

            accuracy = accuracy_score(
                y_true,
                y_pred,
            )

            precision = precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            mcc = matthews_corrcoef(
                y_true,
                y_pred,
            )

            # ---------------------------------------------
            # EVALUATION SUMMARY
            # ---------------------------------------------

            st.subheader("Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{accuracy:.4f}",
            )

            col2.metric(
                "Precision",
                f"{precision:.4f}",
            )

            col3.metric(
                "Recall",
                f"{recall:.4f}",
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "F1-Score",
                f"{f1:.4f}",
            )

            col2.metric(
                "Matthews Correlation Coefficient",
                f"{mcc:.4f}",
            )

            col3.metric(
                "Test Samples",
                f"{len(y_true):,}",
            )

            # ---------------------------------------------
            # CONFUSION MATRIX
            # ---------------------------------------------

            st.subheader("Confusion Matrix")

            cm_df = pd.DataFrame(
                cm,
                index=[
                    "Actual No Purchase",
                    "Actual Purchase",
                ],
                columns=[
                    "Predicted No Purchase",
                    "Predicted Purchase",
                ],
            )

            st.dataframe(
                cm_df,
                use_container_width=True,
            )

            # ---------------------------------------------
            # CLASSIFICATION REPORT
            # ---------------------------------------------

            st.subheader("Classification Report")

            report = classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=[
                    "No Purchase",
                    "Purchase",
                ],
                output_dict=True,
                zero_division=0,
            )

            report_df = pd.DataFrame(report).T

            st.dataframe(
                report_df.style.format(
                    {
                        "precision": "{:.4f}",
                        "recall": "{:.4f}",
                        "f1-score": "{:.4f}",
                        "support": "{:.0f}",
                    },
                    na_rep="-",
                ),
                use_container_width=True,
            )

            # ---------------------------------------------
            # CONFUSION MATRIX VISUALIZATION
            # ---------------------------------------------

            st.subheader("Confusion Matrix Visualization")

            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay

            fig, ax = plt.subplots(figsize=(7, 5))

            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=[
                    "No Purchase",
                    "Purchase",
                ],
            )

            display.plot(
                ax=ax,
                cmap="Blues",
                values_format="d",
                colorbar=False,
            )

            ax.set_title(
                f"Confusion Matrix - {selected_model_name}"
            )

            fig.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            # ---------------------------------------------
            # EVALUATION INTERPRETATION
            # ---------------------------------------------

            st.subheader("Evaluation Interpretation")

            st.write(
                f"""
                - **Accuracy:** {accuracy:.2%} of all predictions
                  were correct.
                - **Precision:** {precision:.2%} of predicted
                  purchases were actual purchases.
                - **Recall:** {recall:.2%} of actual purchases
                  were correctly identified.
                - **F1-score:** {f1:.4f}, the harmonic mean
                  of precision and recall.
                - **MCC:** {mcc:.4f}, a measure of classification
                  quality that considers all four confusion
                  matrix outcomes. A value of +1 indicates
                  perfect prediction, 0 indicates no better
                  than chance in the correlation sense, and
                  -1 indicates completely inverse prediction.
                """
            )

    except Exception as exc:

        st.error(
            f"Prediction or evaluation failed: {exc}"
        )

# =========================================================
# 9. FOOTER
# =========================================================

st.divider()

st.caption(
    "Online Shoppers Purchase Prediction | "
    "Machine Learning Model Evaluation"
)
