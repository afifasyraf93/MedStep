import pandas as pd
import os

df = pd.read_csv("data/test_split.csv")
print(f"Step 1 - CSV loaded: {len(df)} rows")

# Check Frontal/Lateral
df_filtered = df[df["Frontal/Lateral"] == "Frontal"].copy()
print(f"Step 2 - After frontal filter: {len(df_filtered)} rows")

# Check label columns exist
col_map = {
    "pneumonia":        "Pneumonia",
    "cardiomegaly":     "Cardiomegaly",
    "pleural_effusion": "Pleural Effusion",
    "pneumothorax":     "Pneumothorax",
    "atelectasis":      "Atelectasis",
    "lung_mass":        "Lung Lesion"
}
print(f"\nStep 3 - Column check:")
for internal, csv_col in col_map.items():
    exists = csv_col in df.columns
    print(f"  {csv_col:<20} exists={exists}")

# Check label values before cleaning
print(f"\nStep 4 - Label values before cleaning:")
for internal, csv_col in col_map.items():
    if csv_col in df.columns:
        print(f"  {csv_col}: {df[csv_col].value_counts().to_dict()}")

# Simulate cleaning
print(f"\nStep 5 - After cleaning labels:")
for internal, csv_col in col_map.items():
    if csv_col in df.columns:
        df[csv_col] = df[csv_col].fillna(0).replace(-1, 0).astype(int)
print(f"  Rows remaining: {len(df)}")

# Check image paths
print(f"\nStep 6 - Image path check (first 5):")
for i, row in df.head().iterrows():
    full_path = os.path.join("data", row["Path"])
    exists = os.path.exists(full_path)
    print(f"  {full_path} → exists={exists}")

# Check how many images actually exist
print(f"\nStep 7 - Checking all image paths on disk...")
found = 0
missing = 0
missing_examples = []
for _, row in df.iterrows():
    full_path = os.path.join("data", row["Path"])
    if os.path.exists(full_path):
        found += 1
    else:
        missing += 1
        if len(missing_examples) < 5:
            missing_examples.append(full_path)

print(f"  Found:   {found}")
print(f"  Missing: {missing}")
if missing_examples:
    print(f"  Missing examples:")
    for p in missing_examples:
        print(f"    {p}")