import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib

# Load cleaned dataset
df = pd.read_excel(r"data\Telco_customer_churn_clean.xlsx")

# Separate target variable
target = 'Churn Label'
X = df.drop(columns=[target])
y = df[target].apply(lambda x: 1 if x == 'Yes' else 0)

# Encode categorical variables
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----- Logistic Regression -----
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]))
print(classification_report(y_test, y_pred_log))

# ----- Random Forest -----
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))
print(classification_report(y_test, y_pred_rf))

# Save the better model (choose RF if better ROC-AUC)
best_model = rf_model if roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]) >= roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]) else log_model
joblib.dump(best_model, "churn_model.pkl")
print("\nBest model saved as churn_model.pkl")

# Save label encoders for later use
joblib.dump(label_encoders, "label_encoders.pkl")
print("Label encoders saved as label_encoders.pkl")
