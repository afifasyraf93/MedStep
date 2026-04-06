import torch
import pandas as pd
from modules.detection import load_model, predict
from modules.localization import generate_all_heatmaps
from modules.explanation import generate_explanation

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = load_model("models/densenet121_best.pth", device=device)

# Pick a test image with some detections
test_df = pd.read_csv("data/test_split.csv")

# Find image with pleural effusion (most reliable pathology)
effusion_cases = test_df[test_df["Pleural Effusion"] == 1]
test_img = "data/" + effusion_cases.iloc[0]["Path"]
print(f"Test image: {test_img}\n")

# Step 1: Detection
print("Step 1: Detection...")
results = predict(model, test_img, device=device, threshold=0.3)
for path, info in results.items():
    if info["detected"]:
        print(f"  DETECTED: {path} ({info['probability']:.1%})")

# Step 2: Grad-CAM (get best heatmap)
print("\nStep 2: Grad-CAM...")
heatmaps = generate_all_heatmaps(
    model, test_img, results, device=device, threshold=0.3
)

# Use highest scoring heatmap
best_heatmap = None
if heatmaps:
    best = max(heatmaps.items(), key=lambda x: x[1]["score"])
    best_heatmap = best[1]["heatmap"]
    print(f"  Best heatmap: {best[0]} (score={best[1]['score']:.3f})")

# Step 3: Explanation
print("\nStep 3: Generating explanation via Groq...")
explanation = generate_explanation(
    image_path        = test_img,
    detection_results = results,
    heatmap_pil       = best_heatmap,
    threshold         = 0.3
)

if explanation["success"]:
    print("\n" + "="*60)
    print("DETECTION SUMMARY:")
    print(explanation["detection_summary"])
    print("\nFINDINGS:")
    print(explanation["findings"])
    print("\nIMPRESSION:")
    print(explanation["impression"])
    print("="*60)
else:
    print(f"ERROR: {explanation['error']}")