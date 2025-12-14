import pandas as pd

df = pd.read_excel('DataParkir_Fix.xlsx')
df = df.iloc[1:].reset_index(drop=True)
df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]

# Check Number of columns
number_cols = [c for c in df.columns if c.startswith('Number of')]
print("Number of columns and their data:")
for col in number_cols:
    print(f"{col}: dtype={df[col].dtype}, min={df[col].min()}, max={df[col].max()}, mean={df[col].mean()}")

# Check Hours columns
hours_cols = [c for c in df.columns if c.startswith(('Peak Hours', 'Moderate Hours', 'Off-Peak Hours'))]
print("\nHours columns (first 3):")
for col in hours_cols[:3]:
    print(f"{col}: dtype={df[col].dtype}, min={df[col].min()}, max={df[col].max()}, mean={df[col].mean()}")
