import torch
import torch.nn as nn
import numpy as np
import faiss
import json
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from modules.dataset import CheXpertDataset, COL_MAP
from modules.detection import build_model, PATHOLOGIES, TRANSFORM

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/combined_densenet121_best.pth"
MODEL_NAME   = "densenet121"
TRAIN_CSV    = "data/combined_train_split.csv"
BASE_PATH    = "data"
INDEX_PATH   = "faiss_index/index.faiss"
META_PATH    = "faiss_index/metadata.json"
BATCH_SIZE   = 64
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM    = 1024  # DenseNet121 global avg pool output
# ──────────────────────────────────────────────────────────────────────────────


def build_embedding_model(model_name, model_path):
    """Load model and remove the classifier to get embedding extractor."""
    model = build_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # For DenseNet121 — wrap to extract embeddings before classifier
    class DenseNetEmbedder(nn.Module):
        def __init__(self, densenet):
            super().__init__()
            self.features = densenet.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            # TODO: pass x through self.features
            # apply ReLU
            # apply self.pool
            # flatten to [batch, 1024]
            # return
            x = self.features(x)
            x = torch.nn.functional.relu(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return x

    embedder = DenseNetEmbedder(model)
    embedder.eval()
    return embedder


def extract_embeddings(embedder, dataloader):
    """Extract embeddings for all images in the dataloader."""
    all_embeddings = []
    all_paths      = []
    all_labels     = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, 
                                                    desc="Extracting embeddings")):
            images = images.to(DEVICE)
            # TODO: get embeddings from embedder
            # move to cpu, convert to numpy
            # append to all_embeddings
            # append labels.numpy() to all_labels
            embeddings = embedder(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            pass

    all_embeddings = np.vstack(all_embeddings).astype(np.float32)
    all_labels     = np.vstack(all_labels)
    return all_embeddings, all_labels


def build_index(embeddings):
    """Build FAISS IndexFlatL2 from embeddings."""
    # TODO: create faiss.IndexFlatL2 with EMBED_DIM
    # add all embeddings to index
    # return index
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(embeddings)
    return index


def save_metadata(dataloader, all_labels):
    """Save image paths and labels as metadata JSON."""
    dataset = dataloader.dataset
    metadata = []

    # TODO: build a dict with:
    # "index": position in FAISS (just use enumerate)
    # "path": row["Path"]
    # "labels": dict of {pathology: int} for each pathology
    for idx, (i, row) in enumerate(dataset.df.iterrows()):    
        metadata.append({
            "index":  idx,
            "path":   row["Path"],
            "labels": {p: int(row[p]) for p in PATHOLOGIES}
        })

    with open(META_PATH, "w") as f:
        json.dump(metadata, f)
    print(f"Saved metadata → {META_PATH} ({len(metadata)} entries)")


if __name__ == "__main__":
    os.makedirs("faiss_index", exist_ok=True)

    print("Loading dataset...")
    dataset    = CheXpertDataset(TRAIN_CSV, BASE_PATH, transform=TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=0)
    print(f"Dataset size: {len(dataset)} images")

    print("Building embedding model...")
    embedder = build_embedding_model(MODEL_NAME, MODEL_PATH)

    print("Extracting embeddings...")
    embeddings, all_labels = extract_embeddings(embedder, dataloader)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Building FAISS index...")
    index = build_index(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved index → {INDEX_PATH}")

    save_metadata(dataloader, all_labels)
    print("Done.")