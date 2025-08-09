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

# Convert 'Total Charges' to numeric (fixing blank spaces)
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# Check how many NaNs appeared after conversion
print("\nMissing values in 'Total Charges' after conversion:")
print(df['Total Charges'].isnull().sum())

# Option: Drop rows with NaN in 'Total Charges'
df = df.dropna(subset=['Total Charges'])

# Reset index after drop
df.reset_index(drop=True, inplace=True)

print("\nData types after fixing 'Total Charges':")
print(df.dtypes)