# Predict-Loan-Eligibility

Technical Report: Loan Eligibility Prediction for Dream Housing Finance company

1. Introduction-
This report outlines the methodology, implementation, and findings of our Loan Eligibility Prediction project. The objective is to develop a machine learning model that accurately predicts loan eligibility based on customer data, including demographic and financial attributes.

2. Dataset Overview-
The dataset consists of customer information such as:
•	Gender
•	Marital Status
•	Education
•	Number of Dependents
•	Applicant and Co-applicant Income
•	Loan Amount
•	Credit History
These features serve as predictors for the loan eligibility outcome (Approved or Not Approved).

3. Data Preprocessing-
3.1 Handling Missing Values-
Missing values were identified and imputed appropriately to maintain data integrity. Techniques such as mean/mode imputation were used based on the nature of the missing values.
3.2 Feature Encoding-
Categorical variables were encoded using label encoding and one-hot encoding where necessary to make them suitable for machine learning models.
3.3 Feature Scaling-
Continuous numerical features were scaled using StandardScaler to ensure uniform feature distributions for models sensitive to feature magnitudes.
3.4 Splitting Data-
The dataset was split into training (X_train, y_train) and validation (X_val, y_val) sets to evaluate model performance objectively.

4. Model Selection and Hyperparameter Tuning-
4.1 Initial Model Training-
Several classification models were tested, including:
•	Decision Tree
•	K-Nearest Neighbors (KNN)
•	Logistic Regression
•	Support Vector Machine (SVM)
•	Random Forest
•	XGBoost
Each model was trained, and performance metrics such as accuracy were evaluated. However, accuracy alone was insufficient to assess model quality.
4.2 GridSearchCV for Hyperparameter Optimization-
To optimize performance, GridSearchCV was applied to fine-tune hyperparameters for each model. After tuning, the best models were selected based on AUC (Area Under the Curve) rather than accuracy, as AUC provides a better evaluation metric for imbalanced classification problems.

5. Model Evaluation-
5.1 Performance Metrics-
To get a deeper understanding of model performance, the following metrics were considered:
•	Precision: To measure how many predicted approvals were actually correct.
•	Recall: To check how many actual approvals were correctly identified.
•	F1-score: A balance between precision and recall.
•	AUC (Area Under the Curve): To measure the model’s ability to distinguish between classes.
5.2 Optimizing AUC Score-
The AUC scores were analyzed for each model, and optimization was performed to improve them. The Random Forest model, after hyperparameter tuning, achieved the highest AUC score of 0.798, making it the best-performing model.

6. Feature Importance Analysis-
Random Forest’s feature importance was analyzed to determine which features contributed most to predictions. Credit history, loan amount, and applicant income were identified as the most influential features.

7. Cross-Validation Evaluation-
To ensure model robustness, k-fold cross-validation was applied, verifying that the Random Forest model performed consistently across different data splits.

8. Ensemble Model Approach-
An ensemble model combining Random Forest and XGBoost was tested to improve performance. However, it did not yield better AUC scores, indicating that an ensemble approach was not beneficial in this case.

9. Conclusion and Final Decision-
After thorough evaluation, the Random Forest model was selected as the final model due to its highest AUC score (0.798) and stable performance across cross-validation. Further improvements could be achieved by refining feature selection and exploring alternative resampling techniques.
This study highlights the importance of evaluating multiple models, optimizing hyperparameters, and using appropriate performance metrics for loan eligibility prediction.

