import pandas as pd

# Load the dataset
df = pd.read_excel("Telco_customer_churn.xlsx")

# Display basic info
print("First 5 rows:")
print(df.head(), "\n")

print("Dataset info:")
print(df.info(), "\n")

print("Missing values per column:")
print(df.isnull().sum(), "\n")

print("Basic statistics for numeric columns:")
print(df.describe(), "\n")
