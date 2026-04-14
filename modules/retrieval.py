import faiss
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from modules.detection import build_model, PATHOLOGIES, TRANSFORM

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INDEX_PATH  = "faiss_index/index.faiss"
META_PATH   = "faiss_index/metadata.json"
MODEL_PATH  = "models/combined_densenet121_best.pth"
MODEL_NAME  = "densenet121"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K       = 3
# ──────────────────────────────────────────────────────────────────────────────


class DenseNetEmbedder(nn.Module):
    """Same embedder as build_faiss_index.py — extracts 1024-dim vectors."""
    def __init__(self, densenet):
        super().__init__()
        self.features = densenet.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


def load_retrieval_system():
    """Load FAISS index, metadata and embedder. Call once at startup."""
    # TODO: load faiss index using faiss.read_index
    # TODO: load metadata from META_PATH as JSON
    # TODO: build embedder same way as build_faiss_index.py
    #       load model weights, wrap in DenseNetEmbedder, set eval
    # return index, metadata, embedder
    index    = faiss.read_index(INDEX_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)

    model = build_model(MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    embedder = DenseNetEmbedder(model).to(DEVICE)
    embedder.eval()

    return index, metadata, embedder


def retrieve(image_tensor, index, metadata, embedder, top_k=TOP_K):
    """Retrieve top-k similar cases for a query image.

    Args:
        image_tensor: preprocessed tensor [1, 3, 224, 224]
        index: loaded FAISS index
        metadata: loaded metadata list
        embedder: DenseNetEmbedder model
        top_k: number of results to return

    Returns:
        list of dicts with keys:
        {
            "rank": 1,
            "path": "path/to/image.jpg",
            "labels": {"pneumonia": 0, "cardiomegaly": 1, ...},
            "distance": 12.34
        }
    """
    # TODO: extract embedding from image_tensor
    # move to cpu, convert to numpy float32, shape [1, 1024]
    # search index for top_k nearest neighbours
    # distances, indices = index.search(...)
    # build results list from metadata using indices
    with torch.no_grad():
        embedding = embedder(image_tensor.to(DEVICE))
        embedding = embedding.cpu().numpy().astype(np.float32)

    distances, indices = index.search(embedding, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        entry = metadata[idx]
        results.append({
            "rank":     rank + 1,
            "path":     entry["path"],
            "labels":   entry["labels"],
            "distance": round(float(dist), 4)
        })
    return results