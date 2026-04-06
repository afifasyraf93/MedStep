import os
import json
import torch
import numpy as np
import faiss
import pandas as pd
from PIL import Image
from modules.detection import TRANSFORM, PATHOLOGIES


def extract_embedding(model, image_path, device="cuda"):
    """
    Extract 1024-dim embedding from DenseNet121 global average pooling layer.
    This is the feature vector before the classifier head.
    """
    img    = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Extract features before classifier
        features = model.features(tensor)
        # Global average pooling → (1, 1024)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        embedding = torch.flatten(out, 1).cpu().numpy()[0]  # (1024,)

    return embedding.astype(np.float32)


def build_faiss_index(model, csv_path, base_path="data",
                      index_dir="faiss_index", device="cuda"):
    """
    Build FAISS index from all training images.
    Saves index + metadata to disk.

    Args:
        model:      loaded DenseNet121 model
        csv_path:   path to train_split.csv
        base_path:  base data directory
        index_dir:  where to save index files
        device:     cuda or cpu
    """
    os.makedirs(index_dir, exist_ok=True)

    df = pd.read_csv(csv_path).reset_index(drop=True)
    print(f"Building FAISS index from {len(df)} images...")

    model.eval()
    embeddings = []
    metadata   = []
    failed     = 0

    for i, row in df.iterrows():
        img_path = os.path.join(base_path, row["Path"])

        if not os.path.exists(img_path):
            failed += 1
            continue

        try:
            emb = extract_embedding(model, img_path, device)
            embeddings.append(emb)

            # Store metadata for retrieval
            meta = {
                "index":     len(embeddings) - 1,
                "path":      row["Path"],
                "full_path": img_path,
                "labels":    {}
            }

            # Store pathology labels
            col_map = {
                "pneumonia":        "Pneumonia",
                "cardiomegaly":     "Cardiomegaly",
                "pleural_effusion": "Pleural Effusion",
                "pneumothorax":     "Pneumothorax",
                "atelectasis":      "Atelectasis",
                "lung_mass":        "Lung Lesion"
            }
            for internal, csv_col in col_map.items():
                if csv_col in row:
                    meta["labels"][internal] = int(row[csv_col])

            metadata.append(meta)

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  WARNING: Failed on {img_path}: {e}")

        # Progress every 500 images
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(df)} images "
                  f"({len(embeddings)} embedded, {failed} failed)")

    print(f"\nEmbedding complete: {len(embeddings)} images, "
          f"{failed} failed")

    # Build FAISS index
    dim   = 1024
    index = faiss.IndexFlatL2(dim)  # L2 distance

    # Normalize embeddings for better similarity search
    emb_array = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(emb_array)
    index.add(emb_array)

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # Save index and metadata
    index_path    = os.path.join(index_dir, "index.faiss")
    metadata_path = os.path.join(index_dir, "metadata.json")

    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    print(f"Saved index    → {index_path}")
    print(f"Saved metadata → {metadata_path}")

    return index, metadata


def load_faiss_index(index_dir="faiss_index"):
    """Load FAISS index and metadata from disk."""
    index_path    = os.path.join(index_dir, "index.faiss")
    metadata_path = os.path.join(index_dir, "metadata.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            f"Run build_faiss_index.py first."
        )

    index = faiss.read_index(index_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"FAISS index loaded: {index.ntotal} vectors")
    return index, metadata


def retrieve_similar(model, image_path, index, metadata,
                     top_k=3, device="cuda"):
    """
    Retrieve top-k similar cases for a query image.

    Args:
        model:      loaded model
        image_path: path to query image
        index:      loaded FAISS index
        metadata:   loaded metadata list
        top_k:      number of similar cases to return
        device:     cuda or cpu

    Returns:
        list of dicts with keys:
            path, labels, similarity_score, rank
    """
    # Extract query embedding
    query_emb = extract_embedding(model, image_path, device)
    query_emb = query_emb.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_emb)

    # Search index
    distances, indices = index.search(query_emb, top_k + 1)
    # +1 because the query itself might be in the index

    results = []
    for rank, (dist, idx) in enumerate(
        zip(distances[0], indices[0])
    ):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue

        meta = metadata[idx]

        # Convert L2 distance to similarity score (0-1)
        # Lower L2 = more similar, so invert
        similarity = float(1 / (1 + dist))

        results.append({
            "rank":       rank + 1,
            "path":       meta["path"],
            "full_path":  meta["full_path"],
            "labels":     meta["labels"],
            "similarity": similarity,
            "distance":   float(dist)
        })

        if len(results) >= top_k:
            break

    return results