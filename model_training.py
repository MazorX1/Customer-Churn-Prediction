import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from imblearn.over_sampling import SMOTE

# Load cleaned dataset
df = pd.read_excel("data/Telco_customer_churn_clean.xlsx")

# Convert target to numeric
df['__target_num__'] = df['Churn Label'].apply(lambda x: 1 if x == 'Yes' else 0)

# Features and target
drop_cols = ['Churn Label', '__target_num__']
if 'CustomerID' in df.columns:
    drop_cols.append('CustomerID')

X = df.drop(columns=drop_cols)
y = df['__target_num__']

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 1: Balance dataset with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Step 2: Train model with class weight emphasis
model = RandomForestClassifier(class_weight={0: 1, 1: 2}, random_state=42)
model.fit(X_train, y_train)

# Step 3: Get prediction probabilities for class 1
y_probs = model.predict_proba(X_test)[:, 1]

# Step 4: Find optimal threshold (maximizing F1 score)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]

print(f"Best Threshold: {best_threshold:.2f}")
print(f"Precision: {precisions[best_idx]:.2f}, Recall: {recalls[best_idx]:.2f}, F1: {f1_scores[best_idx]:.2f}")

# Step 5: Predict with best threshold
y_pred_opt = (y_probs >= best_threshold).astype(int)

# Step 6: Confusion matrix & report
print("\nConfusion Matrix (Optimized Threshold):")
print(confusion_matrix(y_test, y_pred_opt))

print("\nClassification Report (Optimized Threshold):")
print(classification_report(y_test, y_pred_opt))
