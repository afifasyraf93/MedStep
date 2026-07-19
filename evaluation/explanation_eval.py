"""
explanation_evaluation.py
==========================
Manual review of the explanation module, matching Chapter 3 (Section 3.6.2.4)
exactly:

    Sample size:       30 generated reports
    Criteria (TWO):
        1. Factual accuracy        - findings described in the report must
                                      match the detection results.
        2. Terminology correctness - medical terms in the report are used
                                      correctly and appropriately.
    Success criteria (TWO separate thresholds, scored independently):
        - > 85% of reports are factually accurate
        - > 90% of reports follow proper medical terminology

This is a MANUAL review (Chapter 3 says "manually reviewed"), so the script
does not invent extra automated checks. It only does the mechanical parts:
generating the 30 reports, building a sheet for a human to mark each
criterion Pass/Fail, and then computing the two percentages against the two
thresholds above.

Workflow (matches the Grad-CAM script's two-phase pattern):

    Phase 1:  python explanation_evaluation.py generate
              -> generates 30 reports, writes a CSV scoring sheet with two
                 blank columns: factual_accuracy, terminology_correctness

    (you fill in 1/0 for each column by hand, reviewing each report)

    Phase 2:  python explanation_evaluation.py aggregate
              -> reads the filled sheet, computes the two percentages,
                 checks them against the two thresholds, saves a summary.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import pandas as pd

from modules.detection import PATHOLOGIES
from modules.explanation import generate_report


# ─── CONFIG (must match Chapter 3, Section 3.6.2.4) ────────────────────────
SAMPLE_SIZE = 30   # total reports manually reviewed
CASES_PER_PATHOLOGY = SAMPLE_SIZE // len(PATHOLOGIES)   # 5 per pathology

OUTPUT_DIR  = "experiments/results"
SHEET_PATH  = os.path.join(OUTPUT_DIR, "explanation_scoring_sheet.csv")

# case selection file: columns image_path, pathology (one row per case)
CASE_SELECTION_CSV = "data/explanation_eval_cases.csv"

# the two criteria, exactly as worded in Chapter 3
CRITERIA = {
    "factual_accuracy":
        "The findings described in the report must match the detection results.",
    "terminology_correctness":
        "Medical terms in the report are used correctly and appropriately.",
}

# the two success thresholds, exactly as stated in Chapter 3 (">85%", ">90%")
THRESHOLDS = {
    "factual_accuracy":        0.85,
    "terminology_correctness": 0.90,
}
# ──────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — GENERATE REPORTS + BUILD SCORING SHEET
# ════════════════════════════════════════════════════════════════════════════
def load_cases():
    """Load the case selection CSV (image_path, pathology per row)."""
    if not os.path.exists(CASE_SELECTION_CSV):
        raise FileNotFoundError(
            f"Case selection file not found: {CASE_SELECTION_CSV}\n"
            f"Run build_explanation_cases.py first, or create it manually with "
            f"columns: image_path, pathology  ({CASES_PER_PATHOLOGY} rows per "
            f"pathology, {SAMPLE_SIZE} total)."
        )
    df = pd.read_csv(CASE_SELECTION_CSV)

    required = {"image_path", "pathology"}
    if not required.issubset(df.columns):
        raise ValueError(f"{CASE_SELECTION_CSV} must have columns: {required}")

    if len(df) != SAMPLE_SIZE:
        print(f"  note: case file has {len(df)} rows, Chapter 3 specifies "
              f"{SAMPLE_SIZE}. Continuing with what is available.")

    return df.to_dict(orient="records")


def get_detections_for_case(case):
    """Build the detection dict that gets passed to generate_report.

    Chapter 3's criterion #1 is: 'findings described in the report must
    match the detection results.' This means the detections used here ARE
    the ground truth the human rater checks the report against — so the
    rater needs to see them too. The scoring sheet includes a
    detections_text column for exactly this reason.

    This uses the case's labelled pathology as a high-confidence detection.
    If you would rather feed real model outputs, replace this with a
    detection.py forward pass on case["image_path"] and keep the rest
    unchanged — the rater just compares findings against whatever
    detections_text shows.
    """
    detections = {p: 0.05 for p in PATHOLOGIES}
    detections[case["pathology"]] = 0.92
    return detections


def generate():
    """Phase 1: generate reports for every case and write the scoring sheet."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cases = load_cases()

    rows = []
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] generating report for "
              f"{case['pathology']:16s} {case['image_path']}")

        detections = get_detections_for_case(case)

        try:
            report = generate_report(case["image_path"], detections)
        except Exception as e:
            print(f"    !! failed: {e}")
            report = {"findings": f"ERROR: {e}",
                      "impression": "",
                      "detections_text": ""}

        row = {
            "case_id":         i + 1,
            "pathology":       case["pathology"],
            "image_path":      case["image_path"],
            "detections_text": report["detections_text"],
            "findings":        report["findings"],
            "impression":      report["impression"],
        }
        # two blank columns for the human rater, exactly matching Chapter 3
        for criterion in CRITERIA:
            row[criterion] = ""
        row["rater_notes"] = ""

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(SHEET_PATH, index=False)

    print(f"\nScoring sheet written to: {SHEET_PATH}")
    print(f"Cases: {len(df)}")
    print("\nNext step:")
    print("  1. Open the CSV in Excel / Google Sheets.")
    print("  2. For each report, compare 'findings' against 'detections_text'.")
    print("  3. Mark factual_accuracy as 1 (pass) or 0 (fail).")
    print("  4. Mark terminology_correctness as 1 (pass) or 0 (fail).")
    print("  5. Save, then run:  python explanation_evaluation.py aggregate")
    print("\nCriteria:")
    for name, desc in CRITERIA.items():
        print(f"  - {name}: {desc}")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — AGGREGATE THE FILLED SCORING SHEET
# ════════════════════════════════════════════════════════════════════════════
def aggregate():
    """Phase 2: read the filled sheet, compute the two percentages from
    Chapter 3, and check each against its own threshold."""
    if not os.path.exists(SHEET_PATH):
        raise FileNotFoundError(
            f"Filled scoring sheet not found: {SHEET_PATH}\n"
            f"Run 'generate' first, then fill in the scores."
        )
    df = pd.read_csv(SHEET_PATH)

    criteria = list(CRITERIA.keys())
    for c in criteria:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    missing = df[criteria].isna().any(axis=1)
    if missing.any():
        bad = df.loc[missing, "case_id"].tolist()
        raise ValueError(
            f"Some cases are not fully scored (blank/non-numeric): case_id {bad}\n"
            f"Every criterion must be 1 or 0 for every case."
        )

    n = len(df)

    print("\n=== EXPLANATION MODULE TEST RESULTS ===")
    print(f"Reports manually reviewed: {n}\n")

    results = {}
    for criterion in criteria:
        pass_count = int(df[criterion].sum())
        pass_rate = pass_count / n
        threshold = THRESHOLDS[criterion]
        passed = pass_rate > threshold

        label = criterion.replace("_", " ").title()
        print(f"{label}:")
        print(f"  {pass_count}/{n} reports passed ({pass_rate*100:.1f}%)")
        print(f"  Target:  > {threshold*100:.0f}%")
        print(f"  Result:  {'PASS ✅' if passed else 'FAIL ❌'}\n")

        results[criterion] = {
            "pass_count": pass_count,
            "total":      n,
            "pass_rate_%": round(pass_rate * 100, 1),
            "target_%":   threshold * 100,
            "result":     "PASS" if passed else "FAIL",
        }

    # per-pathology breakdown is not part of Chapter 3's success criteria,
    # but it is useful supporting detail for the Chapter 4 discussion
    print("Per-pathology breakdown (supporting detail, not a success criterion):")
    per_path = df.groupby("pathology")[criteria].mean().mul(100).round(1)
    print(per_path.to_string())

    summary = {
        "reports_reviewed": n,
        "criteria":         results,
        "per_pathology_%":  per_path.to_dict(orient="index"),
    }
    summary_path = os.path.join(OUTPUT_DIR, "explanation_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "generate"

    if cmd == "generate":
        generate()
    elif cmd == "aggregate":
        aggregate()
    else:
        print("Usage:")
        print("  python explanation_evaluation.py generate")
        print("  python explanation_evaluation.py aggregate")