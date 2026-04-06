# save as debug_pa_pool.py
import pandas as pd

TRAIN_CSV = "data/CheXpert-v1.0-small/train.csv"
VALID_CSV = "data/CheXpert-v1.0-small/valid.csv"

df = pd.concat([
    pd.read_csv(TRAIN_CSV),
    pd.read_csv(VALID_CSV)
], ignore_index=True)

# Filter frontal PA only
df = df[df["Frontal/Lateral"] == "Frontal"]
df = df[df["AP/PA"] == "PA"]

col_map = {
    "Pneumonia":        "pneumonia",
    "Cardiomegaly":     "cardiomegaly",
    "Pleural Effusion": "pleural_effusion",
    "Pneumothorax":     "pneumothorax",
    "Atelectasis":      "atelectasis",
    "Lung Lesion":      "lung_mass"
}

print(f"Total PA frontal images: {len(df)}")
print(f"\n{'Pathology':<20} {'Pos Rows':>10} {'Patients':>10} {'Enough?':>10}")
print("-" * 54)

for csv_col, internal in col_map.items():
    if csv_col in df.columns:
        pos = df[df[csv_col] == 1]
        n_rows = len(pos)
        n_patients = pos.copy()
        n_patients["pid"] = n_patients["Path"].apply(
            lambda x: x.split("/")[2] if len(x.split("/")) > 2 else x
        )
        n_pat = n_patients["pid"].nunique()
        enough = "✓" if n_pat >= 1000 else f"⚠ only {n_pat}"
        print(f"{internal:<20} {n_rows:>10} {n_pat:>10} {enough:>10}")