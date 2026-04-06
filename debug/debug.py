import pandas as pd

df = pd.read_csv("data/test_split.csv")
print(f"Total rows: {len(df)}")
print(f"Frontal/Lateral unique values: {df['Frontal/Lateral'].unique()}")
print(f"Frontal count: {len(df[df['Frontal/Lateral'] == 'Frontal'])}")
print(f"NaN count: {df['Frontal/Lateral'].isna().sum()}")