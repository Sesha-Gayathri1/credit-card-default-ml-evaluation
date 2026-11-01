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