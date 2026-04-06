import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from modules.detection import PATHOLOGIES, TRAIN_TRANSFORM, TRANSFORM

COL_MAP = {
    "pneumonia":        "Pneumonia",
    "cardiomegaly":     "Cardiomegaly",
    "pleural_effusion": "Pleural Effusion",
    "pneumothorax":     "Pneumothorax",
    "atelectasis":      "Atelectasis",
    "lung_mass":        "Lung Lesion"
}

class CheXpertDataset(Dataset):
    def __init__(self, csv_path, base_path="data", is_train=True):
        self.df        = pd.read_csv(csv_path).reset_index(drop=True)
        self.base_path = base_path
        self.is_train  = is_train
        self.transform = TRAIN_TRANSFORM if is_train else TRANSFORM

        # Clean labels — CSV is already frontal-filtered, no extra filter needed
        for col in COL_MAP.values():
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col]
                    .fillna(0)
                    .replace(-1, 0)
                    .astype(int)
                )

        print(f"Dataset loaded: {len(self.df)} images "
            f"({'train' if is_train else 'test'})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.base_path, row["Path"])
        img      = Image.open(img_path).convert("RGB")
        img      = self.transform(img)

        labels = torch.zeros(len(PATHOLOGIES))
        for i, (internal, csv_col) in enumerate(COL_MAP.items()):
            if csv_col in row:
                labels[i] = float(row[csv_col])

        return img, labels