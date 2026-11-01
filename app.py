```python
# Validate the uploaded credit card default dataset

expected_features = [f"X{i}" for i in range(1, 24)]

# Check for duplicate column names
if df.columns.duplicated().any():
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    st.error(f"The uploaded CSV contains duplicate columns: {duplicate_columns}")
    st.stop()

# Check for missing required features
missing_columns = [
    col for col in expected_features
    if col not in df.columns
]

if missing_columns:
    st.error(
        "The uploaded CSV is missing required feature columns: "
        + ", ".join(missing_columns)
    )
    st.info(
        "Expected columns: X1 through X23. "
        "Please upload the test_data.csv file from the credit card default dataset."
    )
    st.stop()

# Select features in the exact order expected by the trained models
X = df[expected_features].copy()

# Validate feature values
if X.isnull().any().any():
    st.error("The uploaded CSV contains missing values in the input features.")
    st.stop()

if not X.apply(lambda col: pd.to_numeric(col, errors="coerce").notna().all()).all():
    st.error("All 23 input features must contain numeric values.")
    st.stop()

# Convert features to numeric values
X = X.astype("int64")

st.success(
    f"Dataset validated successfully! "
    f"{len(X):,} records and {len(expected_features)} features found."
)
```
