import torch
import time
import pandas as pd
from modules.detection import load_model
from modules.retrieval import load_faiss_index, retrieve_similar

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = load_model("models/densenet121_best.pth", device=device)

# Load pre-built index
index, metadata = load_faiss_index("faiss_index")

# Pick a test image
test_df  = pd.read_csv("data/test_split.csv")
test_img = "data/" + test_df.iloc[0]["Path"]
print(f"Query image: {test_img}\n")

# Time the retrieval
start   = time.time()
results = retrieve_similar(
    model, test_img, index, metadata,
    top_k=3, device=device
)
elapsed = time.time() - start

print(f"Retrieval time: {elapsed:.3f}s\n")
print(f"Top-3 similar cases:")
print("-" * 60)

for r in results:
    # Show which pathologies the similar case has
    positive_labels = [
        k for k, v in r["labels"].items() if v == 1
    ]
    print(f"Rank {r['rank']}: similarity={r['similarity']:.3f}")
    print(f"  Path:     {r['path']}")
    print(f"  Labels:   {', '.join(positive_labels) if positive_labels else 'none'}")
    print()

print(f"Target: retrieval < 1.0 sec → "
      f"{'PASS ✓' if elapsed < 1.0 else 'FAIL ✗'}")