import pandas as pd

# Load cleaned dataset
df = pd.read_excel("data/Telco_customer_churn_clean.xlsx")
df.columns = df.columns.str.strip()  # remove extra spaces

# Detect target column
if 'Churn Value' in df.columns:
    target_col = 'Churn Value'
    target_numeric = True
elif 'Churn Label' in df.columns:
    target_col = 'Churn Label'
    # Create a numeric version for correlation
    df['__target_num__'] = df[target_col].map({'No': 0, 'Yes': 1})
    target_numeric = False
    target_col_num = '__target_num__'
else:
    raise ValueError("No target column found ('Churn Label' or 'Churn Value').")

# Numeric correlation
print("\n=== Correlation with target (numeric features) ===")
if target_numeric:
    numeric_corr = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
else:
    numeric_corr = df.corr(numeric_only=True)[target_col_num].sort_values(ascending=False)
print(numeric_corr)

# Check potential leakage in categorical columns
print("\n=== Potential Leakage in Categorical Columns ===")
for col in df.select_dtypes(include=['object', 'category']).columns:
    if col != target_col:
        if target_numeric:
            match_rate = (df[col] == df[target_col].map({0: 'No', 1: 'Yes'})).mean()
        else:
            match_rate = (df[col] == df[target_col]).mean()
        if match_rate > 0.9:  # More than 90% match is suspicious
            print(f"⚠ {col} matches target {match_rate*100:.2f}% of the time")

# Check for perfect predictors
print("\n=== Columns that perfectly predict target ===")
if target_numeric:
    tgt = target_col
else:
    tgt = target_col_num

for col in df.columns:
    if col not in [target_col, '__target_num__']:
        if df.groupby(col)[tgt].nunique().max() == 1:
            print(f"⚠ {col} perfectly predicts churn")
