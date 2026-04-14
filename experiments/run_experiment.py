import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.dataset import CheXpertDataset, COL_MAP
from modules.detection import build_model, PATHOLOGIES, TRANSFORM, TRAIN_TRANSFORM, MODEL_NAMES
from modules.evaluation import evaluate

DATASET_CONFIGS = {
    "chexpert": {
        "train_csv": "data/train_split.csv",
        "test_csv":  "data/test_split.csv",
        "base_path": "data"
    },
    "nih": {
        "train_csv": "data/nih_train_split.csv",
        "test_csv":  "data/nih_test_split.csv",
        "base_path": "data"
    },
    "combined": {
        "train_csv": "data/combined_train_split.csv",
        "test_csv":  "data/combined_test_split.csv",
        "base_path": "data"
    }
}

def compute_pos_weights(csv_path):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(csv_path)
    # Handle CheXpert original column names
    from modules.dataset import COL_MAP
    for csv_col, internal in COL_MAP.items():
        if csv_col in df.columns:
            df[csv_col] = df[csv_col].fillna(0).replace(-1, 0)
    df = df.rename(columns=COL_MAP)
    weights = []
    for p in PATHOLOGIES:
        if p in df.columns:
            pos = (df[p] == 1).sum()
            neg = (df[p] == 0).sum()
            weights.append(float(np.sqrt(neg / pos)) if pos > 0 else 1.0)
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(model_name, dataset_name, config):
    EPOCHS     = 30
    BATCH_SIZE = 64
    LR         = 3e-5
    PATIENCE   = 8
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH  = f"models/{dataset_name}_{model_name}_best.pth"

    train_dataset = CheXpertDataset(
        config["train_csv"], config["base_path"], transform=TRAIN_TRANSFORM)
    val_dataset = CheXpertDataset(
        config["test_csv"], config["base_path"], transform=TRANSFORM)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=2)

    model = build_model(model_name).to(DEVICE)
    pos_weights = compute_pos_weights(config["train_csv"]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    # TODO: declare train_curves and val_curves as empty lists
    train_curves = []
    val_curves = []

    for epoch in range(EPOCHS):
        # -- Train --
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # TODO: append train_loss to train_curves and val_loss to val_curves
        train_curves.append(round(train_loss, 4))
        val_curves.append(round(val_loss, 4))
        scheduler.step(val_loss)

        print(f"  [{model_name}] Epoch {epoch+1}/{EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"    ✓ Saved best model")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # TODO: return dict with keys:
    # "best_val_loss", "train_curves", "val_curves", "model_path"
    return {
        "best_val_loss": round(best_val_loss, 4),
        "train_curves": train_curves,
        "val_curves": val_curves,
        "model_path": SAVE_PATH
    }


def run_experiment(dataset_name):
    if dataset_name not in DATASET_CONFIGS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Choose from: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    config = DATASET_CONFIGS[dataset_name]
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments/results", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {dataset_name.upper()}")
    print(f"{'='*60}")

    experiment_results = {}

    for model_name in MODEL_NAMES:
        print(f"\n--- Training {model_name} on {dataset_name} ---")

        # Train and get curves
        train_info = train_model(model_name, dataset_name, config)

        # Evaluate
        print(f"  Evaluating {model_name}...")
        # TODO: call evaluate() with correct args and store as metrics_df
        # Hint: evaluate() is imported from modules.evaluate
        # It needs model_name and test config — check its signature
        # You need to temporarily override its TEST_CSV and BASE_PATH
        # Simplest approach: pass model path and test csv directly
        # For now just call: metrics_df = evaluate(model_name)
        metrics_df = evaluate(
            model_name,
            test_csv   = config["test_csv"],
            base_path  = config["base_path"],
            model_path = f"models/{dataset_name}_{model_name}_best.pth"
        )

        # Store results
        experiment_results[model_name] = {
            "train_curves": train_info["train_curves"],
            "val_curves":   train_info["val_curves"],
            "best_val_loss": train_info["best_val_loss"],
            "metrics": metrics_df.reset_index().to_dict(orient="records")
        }

        # Save incrementally after each model
        out_path = f"experiments/results/{dataset_name}_{model_name}.json"
        with open(out_path, "w") as f:
            json.dump(experiment_results[model_name], f, indent=2)
        print(f"  Saved → {out_path}")

    print(f"\n✓ Experiment '{dataset_name}' complete.")


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "chexpert"
    run_experiment(dataset_name)