import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

from modules.detection import build_model, PATHOLOGIES
from modules.dataset import CheXpertDataset


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_best_threshold_per_pathology(all_labels, all_probs):
    """
    For each pathology, search thresholds 0.05-0.95 and pick the one
    that maximises F1 score individually.
    Returns dict: {pathology: best_threshold}
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    best = {}
    for i, path in enumerate(PATHOLOGIES):
        best_f1  = 0.0
        best_thr = 0.5
        for thr in thresholds:
            preds = (all_probs[:, i] >= thr).astype(int)
            f1    = f1_score(all_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_thr = thr
        best[path] = round(float(best_thr), 2)
    return best


def compute_auc(all_labels, all_probs):
    """Compute AUC-ROC per pathology. Returns dict: {pathology: auc}"""
    aucs = {}
    for i, path in enumerate(PATHOLOGIES):
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            auc = float("nan")  # only one class present in test set
        aucs[path] = float(auc)
    return aucs


def print_confusion_matrix(all_labels, all_probs, thresholds):
    """Print TP, FP, FN, TN per pathology using per-pathology thresholds."""
    print("\n── Confusion Matrix (per pathology) ────────────────────────────")
    print(f"{'Pathology':<20} {'Thr':>5} {'TP':>7} {'FP':>7} "
          f"{'FN':>7} {'TN':>7}  {'Sens':>7} {'Spec':>7}")
    print("-" * 72)

    for i, path in enumerate(PATHOLOGIES):
        thr   = thresholds[path]
        preds = (all_probs[:, i] >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            all_labels[:, i], preds, labels=[0, 1]
        ).ravel()

        # Sensitivity = Recall = TP / (TP + FN)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Specificity = TN / (TN + FP)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"{path:<20} {thr:>5.2f} {tp:>7} {fp:>7} "
              f"{fn:>7} {tn:>7}  {sens:>7.3f} {spec:>7.3f}")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────
    model_files = glob.glob("models/densenet121_best.pth")
    if not model_files:
        model_files = glob.glob("models/densenet121_*.pth")
    if not model_files:
        model_files = glob.glob("models/resnet50_best.pth")
    if not model_files:
        model_files = glob.glob("models/resnet50_*.pth")
    if not model_files:
        print("ERROR: No model files found in models/")
        return

    latest_model = max(model_files, key=os.path.getctime)
    print(f"Evaluating model: {latest_model}")

    model = build_model().to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()

    # ── Load test dataset ─────────────────────────────────────
    test_dataset = CheXpertDataset(
        csv_path="data/test_split.csv",
        base_path="data",
        is_train=False
    )
    print(f"Test dataset size: {len(test_dataset)} images")

    test_loader = DataLoader(
        test_dataset, batch_size=64,
        shuffle=False, num_workers=0
    )

    # ── Run inference ─────────────────────────────────────────
    print("\nRunning inference...")
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs  = np.vstack(all_probs)   # (N, 6)
    all_labels = np.vstack(all_labels)  # (N, 6)

    print(f"Probs range: min={all_probs.min():.4f} "
          f"max={all_probs.max():.4f}")

    # ── True Pos vs Pred Pos overview ─────────────────────────
    print(f"\n── Prediction Overview (threshold=0.5) ─────────────────────")
    print(f"  {'Pathology':<20} {'True Pos':>10} {'Pred Pos':>10} "
          f"{'Ratio':>8}")
    print(f"  {'-'*52}")
    preds_05 = (all_probs >= 0.5).astype(int)
    for i, path in enumerate(PATHOLOGIES):
        true_pos = int(all_labels[:, i].sum())
        pred_pos = int(preds_05[:, i].sum())
        ratio    = pred_pos / true_pos if true_pos > 0 else 0
        flag     = "✓" if 0.7 <= ratio <= 1.3 else "⚠"
        print(f"  {path:<20} {true_pos:>10} {pred_pos:>10} "
              f"{ratio:>7.2f}x {flag}")

    # ── AUC-ROC ───────────────────────────────────────────────
    print(f"\n── AUC-ROC Per Pathology ────────────────────────────────────")
    print(f"{'Pathology':<20} {'AUC':>8}  {'Status':>10}")
    print("-" * 42)
    aucs     = compute_auc(all_labels, all_probs)
    auc_pass = True
    for path in PATHOLOGIES:
        auc    = aucs[path]
        status = "PASS ✓" if auc >= 0.80 else "FAIL ✗"
        if auc < 0.80:
            auc_pass = False
        print(f"{path:<20} {auc:>8.3f}  {status:>10}")
    print("-" * 42)
    macro_auc = np.nanmean(list(aucs.values()))
    print(f"{'Macro AUC':<20} {macro_auc:>8.3f}")
    print(f"AUC Overall: {'ALL PASSED ✓' if auc_pass else 'SOME FAILED ✗'}")

    # ── Per-pathology threshold search ────────────────────────
    print(f"\n── Per-Pathology Best Threshold Search ──────────────────────")
    print(f"{'Pathology':<20} {'Best Thr':>9} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8} {'Status':>10}")
    print("-" * 70)

    best_thresholds = find_best_threshold_per_pathology(
        all_labels, all_probs
    )

    all_pass  = True
    f1_scores = {}

    for i, path in enumerate(PATHOLOGIES):
        thr   = best_thresholds[path]
        preds = (all_probs[:, i] >= thr).astype(int)

        p  = precision_score(all_labels[:, i], preds, zero_division=0)
        r  = recall_score(   all_labels[:, i], preds, zero_division=0)
        f1 = f1_score(       all_labels[:, i], preds, zero_division=0)

        f1_scores[path] = float(f1)
        status = "PASS ✓" if f1 >= 0.70 else "FAIL ✗"
        if f1 < 0.70:
            all_pass = False

        print(f"{path:<20} {thr:>9.2f} {p:>10.3f} "
              f"{r:>8.3f} {f1:>8.3f} {status:>10}")

    print("-" * 70)
    macro_f1 = np.mean(list(f1_scores.values()))
    print(f"{'Macro F1':<20} {'':>9} {'':>10} {'':>8} {macro_f1:>8.3f}")
    print(f"\nF1 Overall: {'ALL PASSED ✓' if all_pass else 'SOME FAILED ✗'}")

    # ── Global threshold comparison (for reference) ───────────
    print(f"\n── Global Threshold Comparison (reference) ─────────────────")
    header = f"{'Threshold':<12}" + "".join(
        f"{p[:8]:>10}" for p in PATHOLOGIES
    ) + f"{'MacroF1':>10}"
    print(header)
    print("-" * (12 + 10 * 6 + 10))

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        preds = (all_probs >= threshold).astype(int)
        f1s   = []
        row   = f"{threshold:<12.1f}"
        for i in range(6):
            f1 = f1_score(all_labels[:, i], preds[:, i], zero_division=0)
            f1s.append(f1)
            row += f"{f1:>10.3f}"
        macro = np.mean(f1s)
        row  += f"{macro:>10.3f}"
        print(row)

    # ── Confusion matrix ──────────────────────────────────────
    print_confusion_matrix(all_labels, all_probs, best_thresholds)

    # ── Save config ───────────────────────────────────────────
    config = {
        "model_path":        latest_model,
        "per_pathology_thresholds": best_thresholds,
        "macro_f1":          float(macro_f1),
        "macro_auc":         float(macro_auc),
        "per_pathology_f1":  f1_scores,
        "per_pathology_auc": aucs
    }
    os.makedirs("models", exist_ok=True)
    with open("models/eval_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n── Summary ──────────────────────────────────────────────────")
    print(f"  Macro F1:  {macro_f1:.3f} "
          f"({'✓' if macro_f1 >= 0.70 else '✗'}  target ≥ 0.70)")
    print(f"  Macro AUC: {macro_auc:.3f} "
          f"({'✓' if macro_auc >= 0.80 else '✗'}  target ≥ 0.80)")
    print(f"\nEvaluation config saved to models/eval_config.json")


if __name__ == "__main__":
    evaluate()