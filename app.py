import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

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

# Model paths
MODEL_DIR = Path("models")

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
