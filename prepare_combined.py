import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CHEXPERT_CSV  = "data/dataset_sampled.csv"
NIH_CSV       = "data/nih_sampled.csv"
OUTPUT_CSV    = "data/combined_sampled.csv"
TRAIN_OUT_CSV = "data/combined_train_split.csv"
TEST_OUT_CSV  = "data/combined_test_split.csv"
RANDOM_SEED   = 42

PATHOLOGIES = [
    "pneumonia", "cardiomegaly", "pleural_effusion",
    "pneumothorax", "atelectasis", "lung_mass"
]
# ──────────────────────────────────────────────────────────────────────────────


def load_and_merge():
    print("Loading CheXpert sampled CSV...")
    chex = pd.read_csv(CHEXPERT_CSV)
    print(f"CheXpert rows: {len(chex)}")

    # Rename CheXpert columns to internal names
    CHEXPERT_COL_MAP = {
        "Pneumonia":        "pneumonia",
        "Cardiomegaly":     "cardiomegaly",
        "Pleural Effusion": "pleural_effusion",
        "Pneumothorax":     "pneumothorax",
        "Atelectasis":      "atelectasis",
        "Lung Lesion":      "lung_mass"
    }

    # Clean uncertain labels (-1 → 0) and rename
    for csv_col, internal_col in CHEXPERT_COL_MAP.items():
        if csv_col in chex.columns:
            chex[csv_col] = chex[csv_col].fillna(0).replace(-1, 0).astype(int)
    chex = chex.rename(columns=CHEXPERT_COL_MAP)

    print("Loading NIH sampled CSV...")
    nih = pd.read_csv(NIH_CSV)
    print(f"NIH rows: {len(nih)}")

    # Keep only the columns we need
    cols = ["Path", "patient_id"] + PATHOLOGIES
    chex = chex[cols].copy()
    nih  = nih[cols].copy()

    # Prefix patient IDs to avoid collision
    chex["patient_id"] = "chex_" + chex["patient_id"].astype(str)
    nih["patient_id"]  = "nih_"  + nih["patient_id"].astype(str)

    # Combine
    combined = pd.concat([chex, nih], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Path"]).copy()

    print(f"\nCombined total: {len(combined)} images")
    print(f"Unique patients: {combined['patient_id'].nunique()}")
    return combined


def split_and_save(df):
    unique_patients = df["patient_id"].unique()

    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=RANDOM_SEED
    )

    train_df = df[df["patient_id"].isin(train_patients)].copy()
    test_df  = df[df["patient_id"].isin(test_patients)].copy()

    df.to_csv(OUTPUT_CSV, index=False)
    train_df.to_csv(TRAIN_OUT_CSV, index=False)
    test_df.to_csv(TEST_OUT_CSV, index=False)

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Saved: {TRAIN_OUT_CSV}, {TEST_OUT_CSV}")
    return train_df, test_df


def print_summary(train_df, test_df):
    total = len(train_df) + len(test_df)
    print("\n── Class Distribution ───────────────────────────────")
    print(f"{'Pathology':<20} {'Train':>8} {'Test':>8} {'% of imgs':>10}")
    print("-" * 50)
    for p in PATHOLOGIES:
        t = int(train_df[p].sum())
        v = int(test_df[p].sum())
        pct = (t + v) / total * 100
        print(f"{p:<20} {t:>8} {v:>8} {pct:>9.1f}%")
    print(f"\nTOTAL: {total} images")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    combined = load_and_merge()
    train_df, test_df = split_and_save(combined)
    print_summary(train_df, test_df)
    print("\nCombined dataset preparation complete.")