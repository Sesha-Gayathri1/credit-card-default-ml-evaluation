import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Default Prediction")
st.write(
    "Upload test data and select a machine learning model "
    "to predict credit card defaults."
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #020b16;
            color: #e6edf7;
        }
        .metric-card {
            padding: 0.15rem 0;
            margin-bottom: 0.5rem;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #aab7c8;
            margin-bottom: 0.2rem;
            font-weight: 500;
        }
        .metric-value {
            font-size: 1.85rem;
            color: #f2f6fb;
            font-weight: 500;
            line-height: 1.2;
        }
        .section-title {
            color: #f2f6fb;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_evaluation_metrics(metrics):
    metric_labels = [
        ("Accuracy", metrics["Accuracy"]),
        ("AUC", metrics["AUC"]),
        ("Precision", metrics["Precision"]),
        ("Recall", metrics["Recall"]),
        ("F1 Score", metrics["F1 Score"]),
        ("MCC", metrics["MCC"]),
    ]

    first_row = st.columns(3)
    second_row = st.columns(3)

    for idx, (label, value) in enumerate(metric_labels[:3]):
        with first_row[idx]:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value:.4f}</div></div>",
                unsafe_allow_html=True,
            )

    for idx, (label, value) in enumerate(metric_labels[3:]):
        with second_row[idx]:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value:.4f}</div></div>",
                unsafe_allow_html=True,
            )


def compute_model_metrics(y_true, y_pred, y_score=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score) if y_score is not None else 0.0,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics

# Model paths
MODEL_DIR = Path("model")

MODEL_PATHS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "KNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Support Vector Machine": MODEL_DIR / "support_vector_machine.joblib",
}

# Model selection
selected_model = st.selectbox(
    "Select a Model",
    list(MODEL_PATHS.keys())
)

# CSV upload
uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")
        st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
        st.dataframe(data.head())

        # Remove target if included in the uploaded dataset
        target_column = "default.payment.next.month"

        if target_column in data.columns:
            X = data.drop(columns=[target_column])
        elif "Y" in data.columns:
            X = data.drop(columns=["Y"])
        else:
            X = data.copy()

        # Validate the expected feature count
        if X.shape[1] != 23:
            st.error(
                f"Expected 23 feature columns, but received {X.shape[1]}. "
                "Please upload a compatible dataset."
            )
            st.stop()

        if not all(
            pd.api.types.is_numeric_dtype(X[col])
            for col in X.columns
        ):
            st.error("All 23 feature columns must be numeric.")
            st.stop()

        if X.isnull().any().any():
            st.error("The uploaded dataset contains missing values.")
            st.stop()

        if st.button("Predict Defaults", type="primary"):
            model_path = MODEL_PATHS[selected_model]

            if not model_path.exists():
                st.error(f"Model file not found: {model_path}")
                st.stop()

            model = joblib.load(model_path)

            # Preserve the feature order used during model training
            expected_features = getattr(
                model, "feature_names_in_", None
            )

            if expected_features is not None:
                if set(X.columns) != set(expected_features):
                    st.error(
                        "The uploaded feature names do not match "
                        "the selected model's expected features."
                    )
                    st.stop()

                X = X[list(expected_features)]

            predictions = model.predict(X)

            results = data.copy()
            results["Predicted_Default"] = predictions

            # Display prediction summary
            default_count = int((predictions == 1).sum())
            non_default_count = int((predictions == 0).sum())

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Customers", len(predictions))
            col2.metric("Predicted Defaults", default_count)
            col3.metric("Predicted Non-Defaults", non_default_count)

            target_label = "default.payment.next.month" if "default.payment.next.month" in data.columns else "Y" if "Y" in data.columns else None

            if target_label is not None:
                y_true = data[target_label].astype(int).to_numpy()
                y_pred = predictions.astype(int)

                y_score = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X)
                    classes = list(model.classes_)
                    if 1 in classes:
                        default_index = classes.index(1)
                        y_score = probabilities[:, default_index]

                metrics = compute_model_metrics(y_true, y_pred, y_score)

                st.subheader("Model Evaluation")
                render_evaluation_metrics(metrics)

                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                confusion_df = pd.DataFrame(
                    cm,
                    index=["Actual 0", "Actual 1"],
                    columns=["Pred 0", "Pred 1"],
                )

                st.subheader("Confusion Matrix")
                st.dataframe(confusion_df, use_container_width=True)

            st.subheader("Prediction Results")
            st.dataframe(results, use_container_width=True)

            # Show probabilities only when the model supports them
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)
                classes = list(model.classes_)

                if 1 in classes:
                    default_index = classes.index(1)

                    results["Default_Probability"] = (
                        probabilities[:, default_index]
                    )

                    st.subheader("Default Probability")
                    st.dataframe(
                        results[["Predicted_Default", "Default_Probability"]],
                        use_container_width=True
                    )

            csv_data = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Predictions CSV",
                data=csv_data,
                file_name="credit_card_default_predictions.csv",
                mime="text/csv"
            )

    except Exception as exc:
        st.error(f"Unable to process the uploaded dataset: {exc}")
