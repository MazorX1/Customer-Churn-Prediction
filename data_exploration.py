import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File paths
raw_file_path = os.path.join("data", "Telco_customer_churn.xlsx")
clean_file_path = os.path.join("data", "Telco_customer_churn_clean.xlsx")

# -------------------------------
# Load the dataset
# -------------------------------
df = pd.read_excel(raw_file_path)

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nColumns in Dataset:")
print(df.columns)

# -------------------------------
# Data Cleaning
# -------------------------------
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
print("\nMissing values in 'Total Charges' after conversion:")
print(df['Total Charges'].isnull().sum())

df = df.dropna(subset=['Total Charges']).reset_index(drop=True)
print("\nData types after fixing 'Total Charges':")
print(df.dtypes)

# -------------------------------
# Remove data leakage columns BEFORE EDA
# -------------------------------
leakage_cols = [
    "CustomerID",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason"
]
df = df.drop(columns=[col for col in leakage_cols if col in df.columns])

# Save target separately
target = df["Churn Label"]

# Save cleaned data
df.to_excel(clean_file_path, index=False)
print(f"\nCleaned data saved to: {clean_file_path}")

# -------------------------------
# Exploratory Data Analysis (EDA)
# -------------------------------
print("\nChurn Label value counts:")
print(df['Churn Label'].value_counts())

# Plot churn distribution
sns.countplot(x='Churn Label', data=df)
plt.title('Churn Distribution')
plt.show()

# Identify numeric features (exclude ID column if exists)
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Plot distributions of numeric features
df[numeric_features].hist(figsize=(12, 10))
plt.suptitle('Numeric Feature Distributions')
plt.show()

# Correlation with churn
df['Churn_num'] = df['Churn Label'].apply(lambda x: 1 if x == 'Yes' else 0)
correlations = df.corr(numeric_only=True)['Churn_num'].sort_values(ascending=False)

print("\nCorrelations with Churn:")
print(correlations)

# Drop temporary numeric churn column
df.drop('Churn_num', axis=1, inplace=True)
