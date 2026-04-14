# test_retrieval.py
import torch
import numpy as np
from PIL import Image
from modules.detection import TRANSFORM
from modules.retrieval import load_retrieval_system, retrieve

TEST_IMAGE = r"D:\Projek\MedStep\data\CheXpert-v1.0-small\train\patient21923\study2\view1_frontal.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading retrieval system...")
index, metadata, embedder = load_retrieval_system()
print(f"Index size: {index.ntotal} vectors")

img = Image.open(TEST_IMAGE).convert("RGB")
image_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

print("Retrieving similar cases...")
results = retrieve(image_tensor, index, metadata, embedder)

for r in results:
    print(f"\nRank {r['rank']} — Distance: {r['distance']}")
    print(f"  Path: {r['path']}")
    print(f"  Labels: {r['labels']}")