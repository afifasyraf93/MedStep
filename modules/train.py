import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from pathlib import Path
from dataset import CheXpertDataset, COL_MAP
from detection import build_model, PATHOLOGIES, TRANSFORM, TRAIN_TRANSFORM

def compute_pos_weights(csv_path):
    df = pd.read_csv(csv_path)

    for csv_col in COL_MAP:
        if csv_col in df.columns:
            df[csv_col] = df[csv_col].fillna(0).replace(-1, 0)
    df = df.rename(columns=COL_MAP)

    weights = []
    for p in PATHOLOGIES:
        pos = (df[p] == 1).sum()
        neg = (df[p] == 0).sum()
        pos_weights = np.sqrt(neg / pos)
        weights.append(pos_weights)
    
    return torch.tensor(weights, dtype=torch.float32)

def train(model_name="densenet121"):
    # --- Config ---
    TRAIN_CSV = "data/train_split.csv"
    VAL_CSV   = "data/test_split.csv"
    BASE_PATH = "data"
    EPOCHS = 50
    BATCH_SIZE = 32
    PATIENCE = 15
    epoch_no_improve = 0
    LR = 3e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH = f"models/{model_name}_best.pth"

    # --- Datasets ---
    train_dataset = CheXpertDataset(TRAIN_CSV, BASE_PATH, transform=TRAIN_TRANSFORM)
    val_dataset   = CheXpertDataset(VAL_CSV,   BASE_PATH, transform=TRANSFORM)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model ---
    model = build_model(model_name).to(DEVICE)

    # --- Loss with pos_weight ---
    pos_weights = compute_pos_weights(TRAIN_CSV).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    # --- LR Scheduler ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,      # halve the LR when plateau detected
        patience=3,      # wait 3 epochs before reducing
        min_lr=1e-7
    )

    # --- Training loop ---
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # -- Train phase --
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

        # -- Validation phase --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} |"
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)

        # -- Save best model --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model: {SAVE_PATH}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epoch_no_improve = 0
            torch.save(model.state_dict, SAVE_PATH)
            print(f"Saved best model: {SAVE_PATH}")
        else:
            epoch_no_improve += 1
            if epoch_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "densenet121"
    train(model_name) 