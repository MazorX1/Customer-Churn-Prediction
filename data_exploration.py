import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the 'data' folder
df = pd.read_excel(r"data\Telco_customer_churn.xlsx")

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display dataset info
print("\nDataset Info:")
print(df.info())

# Display basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display column names
print("\nColumns in Dataset:")
print(df.columns)

# Convert 'Total Charges' to numeric (fixing blank spaces)
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# Check how many NaNs appeared after conversion
print("\nMissing values in 'Total Charges' after conversion:")
print(df['Total Charges'].isnull().sum())

# Drop rows with NaN in 'Total Charges'
df = df.dropna(subset=['Total Charges'])

# Reset index after drop
df.reset_index(drop=True, inplace=True)

print("\nData types after fixing 'Total Charges':")
print(df.dtypes)

# Save the cleaned DataFrame to a new Excel file
cleaned_file_path = r"data\Telco_customer_churn_clean.xlsx"
df.to_excel(cleaned_file_path, index=False)
print(f"\nCleaned data saved to: {cleaned_file_path}")

# --- Exploratory Data Analysis (EDA) ---

# Check target variable distribution using 'Churn Label'
print("\nChurn Label value counts:")
print(df['Churn Label'].value_counts())

# Visualize churn distribution
sns.countplot(x='Churn Label', data=df)
plt.title('Churn Distribution')
plt.show()

# Identify numeric features (exclude 'CustomerID' if present)
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'CustomerID' in numeric_features:
    numeric_features.remove('CustomerID')

# Plot distributions of numeric features
df[numeric_features].hist(figsize=(12, 10))
plt.suptitle('Numeric Feature Distributions')
plt.show()

# Correlation with churn (convert 'Churn Label' to numeric)
df['Churn_num'] = df['Churn Label'].apply(lambda x: 1 if x == 'Yes' else 0)
correlations = df.corr()['Churn_num'].sort_values(ascending=False)
print("\nCorrelations with Churn:")
print(correlations)

# Drop temporary churn numeric column
df.drop('Churn_num', axis=1, inplace=True)
