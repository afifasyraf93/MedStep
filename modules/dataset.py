import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

COL_MAP = {
    "Pneumonia":        "pneumonia",
    "Cardiomegaly":     "cardiomegaly",
    "Pleural Effusion": "pleural_effusion",
    "Pneumothorax":     "pneumothorax",
    "Atelectasis":      "atelectasis",
    "Lung Lesion":      "lung_mass"    
}
class CheXpertDataset(Dataset):
    def __init__(self, csv_path, base_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.base_path = base_path
        for csv_col, internal_col in COL_MAP.items():
            if csv_col in self.df.columns:
                self.df[csv_col] = (
                    self.df[csv_col]
                    .fillna(0)
                    .replace(-1, 0)
                    .astype(int)
                )
        self.df = self.df.rename(columns=COL_MAP)
        self.transform = transform
        pass
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = os.path.join(self.base_path, row["Path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        labels = torch.tensor(
            row[list(COL_MAP.values())].values.astype(np.float32),
            dtype=torch.float32
        )
        
        return image, labels