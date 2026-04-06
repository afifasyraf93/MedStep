import pandas as pd

df = pd.read_csv("data/train_split.csv")

col_map = {
    "Pneumonia": "pneumonia",
    "Cardiomegaly": "cardiomegaly",
    "Pleural Effusion": "pleural_effusion",
    "Pneumothorax": "pneumothorax",
    "Atelectasis": "atelectasis",
    "Lung Lesion": "lung_mass"
}

print(f"Total train images: {len(df)}\n")
print(f"{'Pathology':<20} {'Pos':>8} {'Neg':>8} {'Uncertain':>10} {'Pos%':>8}")
print("-" * 58)

for csv_col, internal in col_map.items():
    if csv_col in df.columns:
        pos = (df[csv_col] == 1).sum()
        neg = (df[csv_col] == 0).sum()
        unc = (df[csv_col] == -1).sum()
        total = len(df)
        print(f"{internal:<20} {pos:>8} {neg:>8} {unc:>10} "
              f"{pos/total*100:>7.1f}%")