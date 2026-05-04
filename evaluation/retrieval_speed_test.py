import torch
import time
import numpy as np
import pandas as pd
import sys
sys.path.append(".")

from modules.retrieval import retrieve
from modules.detection import build_model, TRANSFORM
from PIL import Image
import faiss
import json
import os

# Config
MODEL_PATH   = "models/combined_densenet121_best.pth"
FAISS_INDEX  = "faiss_index/index.faiss"
METADATA     = "faiss_index/metadata.json"
CSV_PATH     = "data/CheXpert-v1.0-small/train.csv"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_QUERIES  = 100

def load_model():
    model = build_model("densenet121", num_classes=6)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)

def load_faiss():
    index = faiss.read_index(FAISS_INDEX)
    with open(METADATA, "r") as f:
        metadata = json.load(f)
    return index, metadata

def get_embedder(model):
    """Return model without final classifier as embedder."""
    import torch.nn as nn
    # TODO: for DenseNet121, embedder = features + adaptive pool
    # hint: wrap in a small nn.Module or use a hook
    # simpler: just use the full model and extract features
    class Embedder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.features = model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return x
    return Embedder(model).to(DEVICE)

def run_speed_test():
    print("Loading model and index...")
    model = load_model()
    index, metadata = load_faiss()
    embedder = get_embedder(model)

    # Load 100 random frontal images from CSV
    df = pd.read_csv(CSV_PATH)
    df = df[df["Frontal/Lateral"] == "Frontal"]
    sample = df.sample(n=NUM_QUERIES, random_state=42)
    image_paths = sample["Path"].tolist()

    print(f"Running {NUM_QUERIES} retrieval queries...")
    times = []
    failed = 0

    for i, img_path in enumerate(image_paths):
        full_path = os.path.join("data", img_path)
        if not os.path.exists(full_path):
            failed += 1
            continue

        try:
            img = Image.open(full_path).convert("RGB")
            image_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

            # TODO: time the retrieve() call
            # hint: use time.perf_counter() before and after
            start = time.perf_counter()
            results = retrieve(image_tensor, index, metadata, embedder)
            end = time.perf_counter()

            elapsed = end - start
            times.append(elapsed)

            if (i + 1) % 10 == 0:
                print(f"  Query {i+1}/{NUM_QUERIES} — {elapsed:.3f}s")

        except Exception as e:
            failed += 1
            print(f"  Failed query {i+1}: {e}")

    # Results
    times = np.array(times)
    print(f"\n{'='*40}")
    print(f"Retrieval Speed Test Results")
    print(f"{'='*40}")
    print(f"Total queries:     {NUM_QUERIES}")
    print(f"Successful:        {len(times)}")
    print(f"Failed:            {failed}")
    print(f"{'─'*40}")
    print(f"Average time:      {times.mean():.4f}s")
    print(f"Median time:       {np.median(times):.4f}s")
    print(f"Min time:          {times.min():.4f}s")
    print(f"Max time:          {times.max():.4f}s")
    print(f"Std deviation:     {times.std():.4f}s")
    print(f"Queries < 1s:      {(times < 1.0).sum()}/{len(times)} ({(times < 1.0).mean()*100:.1f}%)")
    print(f"{'─'*40}")
    print(f"Target:            < 1.0s average")
    print(f"Result:            {'✅ PASS' if times.mean() < 1.0 else '❌ FAIL'}")

    # Save results
    results_df = pd.DataFrame({
        "query": range(1, len(times)+1),
        "time_seconds": times
    })
    results_df.to_csv("evaluation/retrieval_speed_results.csv", index=False)
    print(f"\nDetailed results saved to evaluation/retrieval_speed_results.csv")

if __name__ == "__main__":
    run_speed_test()