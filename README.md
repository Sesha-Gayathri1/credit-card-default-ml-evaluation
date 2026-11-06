## Problem Statement

This project addresses the problem of predicting whether a credit card user is likely to default on their next payment. Credit card default prediction is a binary classification problem in which the goal is to 
classify each customer as either likely to def ault (1) or not likely to default (0), based on their financial and repayment behavior. 

## Dataset Description

The Default of Credit Card Clients dataset was obtained from the UCI Machine Learning Repository. It contains 30,000 observations relating to credit card clients in Taiwan and is designed for predicting default payment behaviour.

The dataset contains 23 explanatory features and one binary target variable. The features include credit limits, demographic attributes, repayment status over six months, monthly bill statement amounts, and previous payment amounts. The target variable, `Y`, represents whether a customer defaults on the next month's payment, where 0 indicates no default and 1 indicates default.

The dataset contains 23,364 observations in class 0 and 6,636 observations in class 1. Therefore, the target distribution is imbalanced, with approximately 77.88% of observations belonging to class 0 and 22.12% belonging to class 1.

This project compares five classification algorithms: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine. Their performance will be evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The customer identifier is excluded from model training. The data will be divided into training and testing subsets, and all models will be evaluated on the same held-out test set. Preprocessing will be fitted on the training data to prevent information leakage.

**Dataset source:** UCI Machine Learning Repository  
**Dataset DOI:** https://doi.org/10.24432/C55S3H

| Check | Result | Interpretation |
|---|---:|---|
| Dataset size | 30,000 × 24 | 30,000 observations, 23 features, and one target |
| Data types | 24 integer columns | All columns are currently represented as integers |
| Missing values | 0 | No missing values detected |
| Class 0 | 23,364 (77.88%) | Majority class |
| Class 1 | 6,636 (22.12%) | Minority class |
| Duplicate rows | 35 | Requires investigation |


| Check | Training set | Test set |
|---|---:|---:|
| Total observations | 24,000 | 6,000 |
| Input features | 23 | 23 |
| Class 0 — No default | 18,691 | 4,673 |
| Class 1 — Default | 5,309 | 1,327 |
| Class 0 percentage | 77.88% | 77.88% |
| Class 1 percentage | 22.12% | 22.12% |


| Model | Feature scaling recommended? | Reason |
|---|---|---|
| Logistic Regression | Yes | Helps numerical optimization when features have very different scales |
| Decision Tree | No | Tree splits generally do not depend on feature scale |
| KNN | Yes | Distances are sensitive to differences in feature scales |
| Gaussian Naive Bayes | Not essential | Scaling is not generally required for its Gaussian likelihood calculations |
| SVM | Yes | Feature scale can strongly affect margins and kernel calculations |


### Experimental Results and Model Comparison

Five classification algorithms—Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine—were evaluated using the Default of Credit Card Clients dataset. Their performance was compared using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The Support Vector Machine achieved the highest accuracy of 81.60%, an AUC of 0.7210, an F1 Score of 0.4486, and an MCC of 0.3809. These results make it the strongest overall candidate among the five evaluated models based on the observed metrics.

Logistic Regression achieved the highest precision of 0.6868, indicating that its positive predictions were comparatively reliable. However, its recall was only 0.2396, meaning that it identified approximately 23.96% of actual defaulters.

Gaussian Naive Bayes achieved the highest recall of 0.8176. However, its accuracy was only 41.60%, and its precision was 0.2496. This indicates a substantial trade-off between detecting defaulters and avoiding false-positive predictions.

The K-Nearest Neighbors classifier achieved an F1 Score of 0.4322, while the Decision Tree obtained an accuracy of 71.45% and an MCC of 0.2042.

The comparison demonstrates that accuracy alone is insufficient for assessing classifiers on a dataset with an imbalanced target distribution. AUC, precision, recall, F1 Score, and MCC provide complementary information about discrimination, positive predictions, class detection, and overall classification quality.

Based on the current experimental results, Support Vector Machine is selected as the preferred baseline model. Further improvements could explore hyperparameter tuning, class weighting, and decision-threshold adjustment, particularly to improve the identification of customers who default.

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree | 0.7145 | 0.6075 | 0.3694 | 0.4115 | 0.3893 | 0.2042 |
| KNN | 0.7928 | 0.7015 | 0.5487 | 0.3564 | 0.4322 | 0.3233 |
| Naive Bayes | 0.4160 | 0.6516 | 0.2496 | 0.8176 | 0.3824 | 0.1111 |
| Support Vector Machine | 0.8160 | 0.7210 | 0.6652 | 0.3384 | 0.4486 | 0.3809 |

### Observation about Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression produced very high precision, which means it is relatively reliable when predicting a default. However, its recall is low, so it misses many actual defaulters and is not ideal for imbalanced default detection. |
| Decision Tree | The Decision Tree is easy to interpret and provides reasonably balanced results, but its accuracy and AUC are lower than the SVM and Logistic Regression models. It is a simple baseline model but not the strongest choice for this dataset. |
| KNN | KNN shows good overall accuracy and a moderate F1 score, but it still underperforms SVM in terms of AUC and MCC. Its performance is also sensitive to feature scaling, which can affect model consistency. |
| Naive Bayes | Naive Bayes achieved the highest recall, making it very effective at detecting actual default cases. However, it has poor precision and the lowest accuracy, which indicates a large number of false positives. |
| Support Vector Machine | The Support Vector Machine is the strongest overall model on this dataset. It achieves the highest Accuracy, AUC, F1 Score, and MCC, making it the best balanced model for predicting credit card default behaviour. |

### Overall, Winner for your dataset?

The best-performing model for this dataset is the Support Vector Machine (SVM). It achieved the highest accuracy (0.8160), the highest AUC (0.7210), the highest F1 score (0.4486), and the highest MCC (0.3809), making it the most reliable model for this credit card default classification task.

### Step 6: Deploy on Streamlit Community Cloud

To deploy the application on Streamlit Community Cloud, follow these steps:

1. Save the project in a GitHub repository.
2. Ensure the repository contains the main app file (`app.py`) and the dependency file (`requirements.txt`).
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with your GitHub account.
4. Click the **New app** button and select the repository, branch, and main file path (`app.py`).
5. Streamlit will automatically install the required libraries from `requirements.txt` and deploy the app.
6. After deployment, copy the generated public URL and share it with your audience.

This project is ready to be deployed and used as an interactive credit card default prediction dashboard.
