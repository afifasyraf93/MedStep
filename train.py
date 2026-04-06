import os
import math
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from modules.detection import build_model
from modules.dataset import CheXpertDataset

COL_MAP = {
    "Pneumonia":        "pneumonia",
    "Cardiomegaly":     "cardiomegaly",
    "Pleural Effusion": "pleural_effusion",
    "Pneumothorax":     "pneumothorax",
    "Atelectasis":      "atelectasis",
    "Lung Lesion":      "lung_mass"
}


def compute_pos_weights(csv_path):
    df = pd.read_csv(csv_path)
    weights = []
    print("\nClass weights (sqrt-dampened):")
    print(f"{'Pathology':<20} {'Pos':>8} {'Neg':>8} {'Weight':>8}")
    print("-" * 48)
    for csv_col, internal in COL_MAP.items():
        if csv_col in df.columns:
            pos = (df[csv_col] == 1).sum()
            neg = (df[csv_col] == 0).sum()
            w   = math.sqrt(neg / pos) if pos > 0 else 1.0
            weights.append(w)
            print(f"{internal:<20} {pos:>8} {neg:>8} {w:>8.2f}x")
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            total_loss += criterion(model(imgs), labels).item()
    return total_loss / len(loader)


def get_scheduler(optimizer, warmup_epochs=3, total_epochs=60):
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
    )
    return warmup, cosine


def train(resume=False):
    os.makedirs("models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset = CheXpertDataset(
        csv_path="data/train_split.csv",
        base_path="data", is_train=True
    )
    val_dataset = CheXpertDataset(
        csv_path="data/test_split.csv",
        base_path="data", is_train=False
    )
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=32,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32,
        shuffle=False, num_workers=0
    )

    model = build_model().to(device)

    if resume and os.path.exists("models/densenet121_best.pth"):
        model.load_state_dict(
            torch.load("models/densenet121_best.pth", map_location=device)
        )
        print("Resumed from checkpoint")

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} "
          f"({trainable/total*100:.1f}%)")

    pos_weights = compute_pos_weights("data/train_split.csv").to(device)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = torch.optim.Adam([
        {"params": model.features.denseblock1.parameters(), "lr": 1e-5},
        {"params": model.features.denseblock2.parameters(), "lr": 1e-5},
        {"params": model.features.denseblock3.parameters(), "lr": 5e-5},
        {"params": model.features.denseblock4.parameters(), "lr": 1e-4},
        {"params": model.features.norm5.parameters(),       "lr": 1e-4},
        {"params": model.classifier.parameters(),           "lr": 3e-4}
    ])

    TOTAL_EPOCHS  = 60
    WARMUP_EPOCHS = 3
    warmup_sched, cosine_sched = get_scheduler(
        optimizer, WARMUP_EPOCHS, TOTAL_EPOCHS
    )

    best_val_loss = float("inf")
    patience      = 15
    patience_ctr  = 0
    best_epoch    = 0

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        avg_train = total_loss / len(train_loader)
        avg_val   = evaluate_epoch(model, val_loader, criterion, device)

        # Step scheduler
        if epoch < WARMUP_EPOCHS:
            warmup_sched.step()
        else:
            cosine_sched.step()

        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"Patience: {patience_ctr}/{patience}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch    = epoch + 1
            patience_ctr  = 0
            torch.save(model.state_dict(),
                       "models/densenet121_best.pth")
            print(f"  → Best saved at epoch {best_epoch} "
                  f"(val={best_val_loss:.4f})")
        else:
            patience_ctr += 1

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       f"models/densenet121_epoch{epoch+1}.pth")

        if patience_ctr >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}. "
                  f"Best was epoch {best_epoch}")
            break

    print(f"Done. Best model from epoch {best_epoch}.")


if __name__ == "__main__":
    train(resume=False)