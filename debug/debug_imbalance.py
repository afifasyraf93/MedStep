# save as debug_imbalance.py
import pandas as pd
import numpy as np

train = pd.read_csv("data/train_split.csv")
test  = pd.read_csv("data/test_split.csv")

col_map = {
    "Pneumonia":       "pneumonia",
    "Cardiomegaly":    "cardiomegaly",
    "Pleural Effusion":"pleural_effusion",
    "Pneumothorax":    "pneumothorax",
    "Atelectasis":     "atelectasis",
    "Lung Lesion":     "lung_mass"
}

print(f"{'Pathology':<20} {'Train Pos':>10} {'Train Neg':>10} "
      f"{'Ratio':>8} {'Test Pos':>10}")
print("-" * 62)

for csv_col, internal in col_map.items():
    if csv_col in train.columns:
        tr_pos = (train[csv_col] == 1).sum()
        tr_neg = (train[csv_col] == 0).sum()
        te_pos = (test[csv_col]  == 1).sum()
        ratio  = tr_neg / tr_pos if tr_pos > 0 else 0
        print(f"{internal:<20} {tr_pos:>10} {tr_neg:>10} "
              f"{ratio:>8.1f}x {te_pos:>10}")

# Also check multi-label co-occurrence
print(f"\nMulti-label distribution:")
cols = list(col_map.keys())
label_sums = train[cols].sum(axis=1)
for n in range(7):
    count = (label_sums == n).sum()
    pct   = count / len(train) * 100
    print(f"  {n} labels: {count:>6} images ({pct:.1f}%)")