# save as debug_pa.py
import pandas as pd

df = pd.read_csv("data/dataset_sampled.csv")
pa = df[df["AP/PA"] == "PA"]
ap = df[df["AP/PA"] == "AP"]

col_map = {
    "Pneumonia":        "pneumonia",
    "Cardiomegaly":     "cardiomegaly",
    "Pleural Effusion": "pleural_effusion",
    "Pneumothorax":     "pneumothorax",
    "Atelectasis":      "atelectasis",
    "Lung Lesion":      "lung_mass"
}

print(f"Total: {len(df)} | PA: {len(pa)} | AP: {len(ap)}")
print(f"\n{'Pathology':<20} {'PA Pos':>8} {'AP Pos':>8} {'PA%':>8}")
print("-" * 48)
for csv_col, internal in col_map.items():
    if csv_col in df.columns:
        pa_pos = (pa[csv_col] == 1).sum()
        ap_pos = (ap[csv_col] == 1).sum()
        total_pos = pa_pos + ap_pos
        pa_pct = pa_pos / total_pos * 100 if total_pos > 0 else 0
        print(f"{internal:<20} {pa_pos:>8} {ap_pos:>8} {pa_pct:>7.1f}%")