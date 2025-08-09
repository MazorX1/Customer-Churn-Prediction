import pandas as pd

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