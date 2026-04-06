import torch
from modules.detection import load_model
from modules.retrieval import build_faiss_index

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = load_model("models/densenet121_best.pth", device=device)
print("Model loaded")

index, metadata = build_faiss_index(
    model        = model,
    csv_path     = "data/train_split.csv",
    base_path    = "data",
    index_dir    = "faiss_index",
    device       = device
)

print(f"\nIndex built successfully")
print(f"Total vectors: {index.ntotal}")
print(f"Total metadata entries: {len(metadata)}")