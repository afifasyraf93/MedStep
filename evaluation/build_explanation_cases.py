"""
build_explanation_cases.py
===========================
Builds data/explanation_eval_cases.csv — the case selection file used by
explanation_evaluation.py.

It samples CASES_PER_PATHOLOGY positive cases for each of the 6 pathologies
from the test split, so the explanation test runs on real held-out images.

Two ways to use it:

  A) FROM SCRATCH (sample from the test split):
       python build_explanation_cases.py

  B) REUSE GRAD-CAM CASES (recommended — keeps the two human evals aligned):
       python build_explanation_cases.py --from-gradcam path/to/gradcam_cases.csv
     The Grad-CAM file just needs an image-path column and a pathology column;
     adjust GRADCAM_PATH_COL / GRADCAM_PATHOLOGY_COL below if they differ.

Output columns: image_path, pathology   (exactly what the evaluator expects)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd

from modules.detection import PATHOLOGIES
from modules.dataset import COL_MAP


# ─── CONFIG ─────────────────────────────────────────────────────────────────
TEST_CSV            = "data/test_split.csv"
BASE_PATH           = "data"            # prepended to the relative Path column
OUTPUT_CSV          = "data/explanation_eval_cases.csv"

# Chapter 3, Section 3.6.2.4: sample of 30 generated reports manually reviewed.
SAMPLE_SIZE          = 30
CASES_PER_PATHOLOGY  = SAMPLE_SIZE // 6   # 6 pathologies -> 5 per pathology
RANDOM_SEED          = 42                 # reproducible sampling

# only frontal views — match the Grad-CAM test scope.
# CheXpert uses an "AP/PA" column; if your split lacks it this is skipped.
FRONTAL_COL         = "AP/PA"
FRONTAL_VALUES      = {"AP", "PA"}

# column names if reusing a Grad-CAM selection file (option B)
GRADCAM_PATH_COL      = "image_path"
GRADCAM_PATHOLOGY_COL = "pathology"
# ──────────────────────────────────────────────────────────────────────────────


def build_from_test_split():
    """Sample positive cases per pathology straight from the test split."""
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test split not found: {TEST_CSV}")

    df = pd.read_csv(TEST_CSV)

    # normalise label columns the same way the dataset does:
    # NaN -> 0, uncertain (-1) -> 0, then rename to internal names.
    for csv_col, internal_col in COL_MAP.items():
        if csv_col in df.columns:
            df[csv_col] = df[csv_col].fillna(0).replace(-1, 0).astype(int)
    df = df.rename(columns=COL_MAP)

    # keep frontal views only, if the column is present
    if FRONTAL_COL in df.columns:
        df = df[df[FRONTAL_COL].isin(FRONTAL_VALUES)]
    else:
        print(f"  (note: '{FRONTAL_COL}' column not found — skipping frontal filter)")

    rows = []
    for pathology in PATHOLOGIES:
        positives = df[df[pathology] == 1]

        if len(positives) < CASES_PER_PATHOLOGY:
            print(f"  !! only {len(positives)} positive cases for "
                  f"{pathology} (wanted {CASES_PER_PATHOLOGY})")

        n = min(CASES_PER_PATHOLOGY, len(positives))
        sample = positives.sample(n=n, random_state=RANDOM_SEED)

        for _, r in sample.iterrows():
            rows.append({
                "image_path": os.path.join(BASE_PATH, r["Path"]),
                "pathology":  pathology,
            })

    return pd.DataFrame(rows)


def build_from_gradcam(gradcam_csv):
    """Reuse the exact images/labels from the Grad-CAM evaluation."""
    if not os.path.exists(gradcam_csv):
        raise FileNotFoundError(f"Grad-CAM case file not found: {gradcam_csv}")

    g = pd.read_csv(gradcam_csv)

    for col in (GRADCAM_PATH_COL, GRADCAM_PATHOLOGY_COL):
        if col not in g.columns:
            raise ValueError(
                f"Expected column '{col}' in {gradcam_csv}. "
                f"Found: {list(g.columns)}. "
                f"Adjust GRADCAM_PATH_COL / GRADCAM_PATHOLOGY_COL."
            )

    out = g[[GRADCAM_PATH_COL, GRADCAM_PATHOLOGY_COL]].rename(columns={
        GRADCAM_PATH_COL:      "image_path",
        GRADCAM_PATHOLOGY_COL: "pathology",
    })

    # warn if any pathology label is outside the known set (typo guard)
    unknown = set(out["pathology"]) - set(PATHOLOGIES)
    if unknown:
        print(f"  !! pathology labels not in PATHOLOGIES: {unknown}")

    return out


def main():
    if "--from-gradcam" in sys.argv:
        idx = sys.argv.index("--from-gradcam")
        gradcam_csv = sys.argv[idx + 1]
        df = build_from_gradcam(gradcam_csv)
        source = f"Grad-CAM cases ({gradcam_csv})"
    else:
        df = build_from_test_split()
        source = f"test split ({TEST_CSV})"

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSource: {source}")
    print(f"Wrote {len(df)} cases to {OUTPUT_CSV}")
    print("\nCases per pathology:")
    print(df["pathology"].value_counts().reindex(PATHOLOGIES, fill_value=0).to_string())
    print("\nNext: python explanation_evaluation.py generate")


if __name__ == "__main__":
    main()