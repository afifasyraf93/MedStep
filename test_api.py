# test_api.py
import requests
import json

url = "http://localhost:8000/analyze"
image_path = r"D:\Projek\MedStep\data\CheXpert-v1.0-small\train\patient21923\study2\view1_frontal.jpg"

with open(image_path, "rb") as f:
    response = requests.post(url, files={"file": f})

result = response.json()

print("=== DETECTIONS ===")
for pathology, prob in result["detections"].items():
    flag = "✓" if prob >= 0.5 else " "
    print(f"  {flag} {pathology}: {prob}")

print("\n=== HEATMAPS ===")
for pathology, b64 in result["heatmaps"].items():
    print(f"  {pathology}: {len(b64)} chars")

print("\n=== SIMILAR CASES ===")
for case in result["similar_cases"]:
    print(f"  Rank {case['rank']} — {case['path']} (dist: {case['distance']})")

print("\n=== REPORT ===")
print("Findings:", result["report"]["findings"])
print("Impression:", result["report"]["impression"])