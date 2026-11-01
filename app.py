from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURE_COLUMNS = [f"X{i}" for i in range(1, 24)]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


# --------------------------------------------------
# 2. Load saved models and evaluation results
# --------------------------------------------------

@st.cache_resource
def load_models():
    loaded = {}

    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        loaded[name] = joblib.load(path)

    return loaded


@st.cache_data
def load_comparison():
    path = OUTPUT_DIR / "model_comparison.csv"

    if not path.exists():
        return None

    return pd.read_csv(path, index_col=0)


@st.cache_data
def load_confusion_matrices():
    path = OUTPUT_DIR / "confusion_matrix_results.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


try:
    models = load_models()
except Exception as exc:
    st.error(f"Unable to load models: {exc}")
    st.stop()

comparison_df = load_comparison()
confusion_df = load_confusion_matrices()


# --------------------------------------------------
# 3. Sidebar navigation
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Dashboard",
        "Model Comparison",
        "Confusion Matrices",
        "Customer Prediction",
    ]
)


# --------------------------------------------------
# 4. Dashboard
# --------------------------------------------------

if page == "Dashboard":

    st.title("💳 Credit Card Default Prediction")
    st.markdown(
        """
        An interactive machine-learning application for
        exploring credit card default classification models.
        """
    )

    st.info(
        "Dataset: Default of Credit Card Clients "
        "— Taiwan."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Observations", "30,000")
    col2.metric("Input Features", "23")
    col3.metric("Trained Models", len(models))

    st.subheader("Target Distribution")

    target_counts = pd.Series(
        {
            "No Default (0)": 23364,
            "Default (1)": 6636,
        }
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    target_counts.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title("Distribution of Target Classes")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Customers")
    ax.tick_params(axis="x", rotation=0)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
        ### About the project

        The objective is to compare five classification
        algorithms for predicting whether a customer will
        default on their credit card payment next month.

        **Algorithms evaluated**
        - Logistic Regression
        - Decision Tree
        - K-Nearest Neighbors
        - Gaussian Naive Bayes
        - Support Vector Machine

        **Evaluation metrics**
        Accuracy, AUC, Precision, Recall, F1 Score,
        and Matthews Correlation Coefficient (MCC).
        """
    )


# --------------------------------------------------
# 5. Model comparison
# --------------------------------------------------

elif page == "Model Comparison":

    st.title("📊 Model Performance Comparison")

    if comparison_df is None:
        st.warning(
            "model_comparison.csv was not found in "
            "the outputs folder."
        )

    else:
        required_metrics = [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC",
        ]

        missing = [
            metric
            for metric in required_metrics
            if metric not in comparison_df.columns
        ]

        if missing:
            st.error(
                f"Missing metric columns: {missing}"
            )

        else:
            st.subheader("Evaluation Results")

            st.dataframe(
                comparison_df[required_metrics].round(4),
                use_container_width=True,
            )

            selected_metrics = st.multiselect(
                "Select metrics to compare",
                required_metrics,
                default=[
                    "Accuracy",
                    "AUC",
                    "F1 Score",
                ],
            )

            if selected_metrics:
                st.subheader("Performance Chart")

                fig, ax = plt.subplots(figsize=(11, 5))

                comparison_df[selected_metrics].plot(
                    kind="bar",
                    ax=ax,
                )

                ax.set_xlabel("Model")
                ax.set_ylabel("Score")
                ax.set_title(
                    "Classification Model Comparison"
                )
                ax.set_ylim(-1, 1)
                ax.tick_params(
                    axis="x",
                    rotation=25,
                )

                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

            st.caption(
                "Metrics are loaded from the saved notebook "
                "evaluation results."
            )


# --------------------------------------------------
# 6. Confusion matrices
# --------------------------------------------------

elif page == "Confusion Matrices":

    st.title("🔍 Confusion Matrix Analysis")

    if confusion_df is None:
        st.warning(
            "confusion_matrix_results.csv was not found "
            "in the outputs folder."
        )

    else:
        required_columns = [
            "Model",
            "TN",
            "FP",
            "FN",
            "TP",
        ]

        if not set(required_columns).issubset(
            confusion_df.columns
        ):
            st.error(
                "The confusion matrix CSV must contain "
                "Model, TN, FP, FN, and TP columns."
            )

        else:
            selected_model = st.selectbox(
                "Select a classifier",
                confusion_df["Model"].tolist(),
            )

            row = confusion_df.loc[
                confusion_df["Model"] == selected_model
            ].iloc[0]

            cm = [
                [int(row["TN"]), int(row["FP"])],
                [int(row["FN"]), int(row["TP"])],
            ]

            st.subheader(selected_model)

            fig, ax = plt.subplots(figsize=(6, 5))

            image = ax.imshow(cm)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

            ax.set_xticklabels(
                ["No Default (0)", "Default (1)"]
            )
            ax.set_yticklabels(
                ["No Default (0)", "Default (1)"]
            )

            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix")

            for i in range(2):
                for j in range(2):
                    ax.text(
                        j,
                        i,
                        f"{cm[i][j]:,}",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

            fig.colorbar(image, ax=ax)
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            col1, col2 = st.columns(2)

            col1.metric("True Negatives", f"{cm[0][0]:,}")
            col2.metric("False Positives", f"{cm[0][1]:,}")

            col3, col4 = st.columns(2)

            col3.metric("False Negatives", f"{cm[1][0]:,}")
            col4.metric("True Positives", f"{cm[1][1]:,}")


# --------------------------------------------------
# 7. Customer prediction
# --------------------------------------------------

elif page == "Customer Prediction":

    st.title("👤 Customer Default Prediction")

    st.warning(
        "This initial interface accepts the dataset's "
        "23 encoded features directly. These are not yet "
        "converted into customer-friendly form fields."
    )

    st.markdown(
        """
        Enter the values of the 23 features in the same
        order and encoding used by the original dataset.

        X1 represents credit limit, X2 represents sex,
        X3 represents education, X4 represents marital
        status, X5 represents age, and X6–X23 contain
        repayment history, bill amounts, and payment amounts.
        """
    )

    with st.form("prediction_form"):

        values = {}

        for start in range(1, 24, 4):
            cols = st.columns(4)

            for offset, col in enumerate(cols):
                feature_number = start + offset

                if feature_number > 23:
                    break

                feature = f"X{feature_number}"

                with col:
                    values[feature] = st.number_input(
                        feature,
                        value=0,
                        step=1,
                    )

        selected_model = st.selectbox(
            "Choose a model",
            list(models.keys()),
        )

        submitted = st.form_submit_button(
            "Predict Default Status"
        )

    if submitted:

        input_df = pd.DataFrame(
            [[values[col] for col in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )

        model = models[selected_model]

        try:
            prediction = int(
                model.predict(input_df)[0]
            )

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error(
                    "Predicted outcome: Default (1)"
                )
            else:
                st.success(
                    "Predicted outcome: No Default (0)"
                )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(
                    input_df
                )[0]

                classes = list(model.classes_)
                default_index = classes.index(1)

                st.metric(
                    "Estimated Default Probability",
                    f"{probabilities[default_index]:.2%}",
                )

            else:
                st.info(
                    "Probability estimates are unavailable "
                    "for this model configuration."
                )

            st.caption(
                "This is an experimental model prediction, "
                "not a financial decision or guarantee."
            )

        except Exception as exc:
            st.error(
                f"Prediction failed: {exc}"
            )