import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, f1_score, 
                             precision_score, recall_score, accuracy_score)
from modules.dataset import CheXpertDataset, COL_MAP
from modules.detection import build_model, PATHOLOGIES, TRANSFORM

def evaluate(model_name, test_csv=None, base_path=None, model_path=None):
    TEST_CSV   = test_csv   or "data/test_split.csv"
    BASE_PATH  = base_path  or "data"
    MODEL_PATH = model_path or f"models/{model_name}_best.pth"
    BATCH_SIZE  = 32
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    model   = build_model(model_name)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model   = model.to(DEVICE)
    model.eval()

    # --- Dataloader ---
    dataset = CheXpertDataset(TEST_CSV, BASE_PATH, transform=TRANSFORM)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, 
                         shuffle=False, num_workers=0)

    # --- Collect predictions ---
    all_labels  = []   # true labels
    all_probs   = []   # sigmoid probabilities

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            # TODO: apply sigmoid to outputs to get probabilities
            probability = torch.sigmoid(outputs).cpu().numpy()
            # TODO: append probs (cpu, numpy) to all_probs
            all_probs.append(probability)
            # TODO: append labels (numpy) to all_labels
            all_labels.append(labels.numpy())

    # Stack into arrays of shape [N, 6]
    all_labels = np.vstack(all_labels)
    all_probs  = np.vstack(all_probs)

    # --- Compute metrics ---
    # Threshold probabilities at 0.5 for F1/Precision/Recall/Accuracy
    all_preds = (all_probs >= 0.5).astype(int)

    results = []
    for i, pathology in enumerate(PATHOLOGIES):
        # TODO: compute auc, f1, precision, recall, accuracy
        # # for column i of all_labels and all_probs/all_preds
        metrics = {
            "pathology": pathology,
            "auc":       roc_auc_score(all_labels[:, i], all_probs[:, i]),
            "f1":        f1_score(all_labels[:, i], all_preds[:, i], zero_division=0),
            "precision": precision_score(all_labels[:, i], all_preds[:, i], zero_division=0),
            "recall":    recall_score(all_labels[:, i], all_preds[:, i], zero_division=0),
            "accuracy":  accuracy_score(all_labels[:, i], all_preds[:, i])
        }
        # append a dict to results
        results.append(metrics)

    # --- Print table ---
    df = pd.DataFrame(results)
    df = df.set_index("pathology")
    print(f"\n=== {model_name.upper()} Results ===")
    print(df.round(3).to_string())
    print(f"\nMacro AUC: {df['auc'].mean():.3f}")
    print(f"Macro F1:  {df['f1'].mean():.3f}")

    return df


def compare():
    df_resnet   = evaluate("resnet50")
    df_densenet = evaluate("densenet121")

    print("\n=== AUC COMPARISON ===")
    comparison = pd.DataFrame({
        "pathology":   PATHOLOGIES,
        "resnet50":    df_resnet["auc"].values,
        "densenet121": df_densenet["auc"].values,
    }).set_index("pathology")
    print(comparison.round(3).to_string())
    print(f"\nResNet50   Macro AUC: {df_resnet['auc'].mean():.3f}")
    print(f"DenseNet121 Macro AUC: {df_densenet['auc'].mean():.3f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare()
    else:
        model_name = sys.argv[1] if len(sys.argv) > 1 else "densenet121"
        evaluate(model_name)