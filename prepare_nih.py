import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ─── CONFIG ───────────────────────────────────────────────────────────────────
NIH_CSV       = r"D:\Projek\MedStep\data\NIH\Data_Entry_2017.csv"
NIH_IMG_DIRS  = [
    rf"D:\Projek\MedStep\data\NIH\images_{str(i).zfill(3)}\images"
    for i in range(1, 13)
]
OUTPUT_CSV    = "data/nih_sampled.csv"
TRAIN_OUT_CSV = "data/nih_train_split.csv"
TEST_OUT_CSV  = "data/nih_test_split.csv"

TARGET_PER_CLASS = 8000
RANDOM_SEED      = 42

# NIH label strings → your internal names
NIH_LABEL_MAP = {
    "Pneumonia":    "pneumonia",
    "Cardiomegaly": "cardiomegaly",
    "Effusion":     "pleural_effusion",
    "Pneumothorax": "pneumothorax",
    "Atelectasis":  "atelectasis",
    "Mass":         "lung_mass",
    "Nodule":       "lung_mass",
}

PATHOLOGIES = [
    "pneumonia", "cardiomegaly", "pleural_effusion",
    "pneumothorax", "atelectasis", "lung_mass"
]
# ──────────────────────────────────────────────────────────────────────────────


def build_image_path_map():
    """Build a dict of {filename: full_path} scanning all 12 image folders."""
    print("Scanning image folders...")
    path_map = {}
    for img_dir in NIH_IMG_DIRS:
        for fname in os.listdir(img_dir):
            if fname.endswith(".png"):
                path_map[fname] = os.path.join(img_dir, fname)
    print(f"Found {len(path_map)} images")
    return path_map


def load_and_clean(path_map):
    """Load CSV, filter PA/AP frontal, create binary columns for each pathology."""
    print("Loading NIH CSV...")
    df = pd.read_csv(NIH_CSV)
    print(f"Total rows: {len(df)}")

    # Keep PA and AP frontal views only
    df = df[df["View Position"].isin(["PA", "AP"])].copy()
    print(f"After frontal filter: {len(df)} images")

    # Add full image path
    df["full_path"] = df["Image Index"].map(path_map)
    df = df[df["full_path"].notna()].copy()

    # Add patient_id column
    df["patient_id"] = df["Patient ID"].astype(str)

    # TODO: Create binary columns for each pathology
    # For each internal name in PATHOLOGIES, add a column of 0s first
    # Then for each NIH label in NIH_LABEL_MAP:
    #   set the corresponding internal column to 1
    #   where the Finding Labels string contains that NIH label
    # Remember: Mass and Nodule both map to lung_mass (use OR logic)
    # Hint: df["Finding Labels"].str.contains(nih_label, case=False)
    for internal in PATHOLOGIES:
        df[internal] = 0

    for nih_label, internal in NIH_LABEL_MAP.items():
        mask = df["Finding Labels"].str.contains(nih_label, case=False, na=False)
        df.loc[mask, internal] = 1

    return df


def balanced_sample(df):
    """Same patient-level sampling logic as prepare_dataset.py."""
    np.random.seed(RANDOM_SEED)
    print(f"\nTarget per pathology: {TARGET_PER_CLASS} images each\n")

    class_buckets = {}
    for pathology in PATHOLOGIES:
        positive = df[df[pathology] == 1].copy()
        unique_patients = positive["patient_id"].unique()
        print(f"  {pathology}: {len(positive)} positive rows | "
              f"{len(unique_patients)} unique patients")

        one_per_patient = (
            positive.groupby("patient_id")
            .first()
            .reset_index()
        )

        available = len(one_per_patient)
        n_to_sample = min(TARGET_PER_CLASS, available)

        if available < TARGET_PER_CLASS:
            print(f"    ⚠ Only {available} available, taking all")

        selected = one_per_patient.sample(n=n_to_sample, random_state=RANDOM_SEED)
        class_buckets[pathology] = selected
        print(f"    → Sampled {len(selected)} images")

    all_selected = pd.concat(class_buckets.values(), ignore_index=True)

    before_dedup = len(all_selected)
    all_selected = all_selected.drop_duplicates(subset=["Image Index"]).copy()
    after_dedup = len(all_selected)
    print(f"\nBefore deduplication: {before_dedup}")
    print(f"After deduplication:  {after_dedup}")

    return all_selected


def split_and_save(df):
    """Patient-level 80/20 split."""
    unique_patients = df["patient_id"].unique()
    print(f"Unique patients: {len(unique_patients)}")

    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=RANDOM_SEED
    )

    train_df = df[df["patient_id"].isin(train_patients)].copy()
    test_df  = df[df["patient_id"].isin(test_patients)].copy()

    # TODO: Add a "Path" column = full_path
    # This makes NIH compatible with CheXpertDataset which reads row["Path"]
    train_df["Path"]    = train_df["full_path"]
    test_df["Path"]     = test_df["full_path"]
    df["Path"]          = df["full_path"]

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
    path_map = build_image_path_map()
    df = load_and_clean(path_map)
    df = balanced_sample(df)
    train_df, test_df = split_and_save(df)
    print_summary(train_df, test_df)
    print("\nNIH dataset preparation complete.")