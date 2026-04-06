import torch
import pandas as pd
from modules.detection import load_model, predict
from modules.localization import generate_all_heatmaps, save_heatmap

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = load_model("models/densenet121_best.pth", device=device)

test_df = pd.read_csv("data/test_split.csv")

col_map = {
    "pneumonia":        "Pneumonia",
    "cardiomegaly":     "Cardiomegaly",
    "pleural_effusion": "Pleural Effusion",
    "pneumothorax":     "Pneumothorax",
    "atelectasis":      "Atelectasis",
    "lung_mass":        "Lung Lesion"
}

print("Testing one confirmed positive case per pathology...\n")

for internal, csv_col in col_map.items():
    # Find first test image confirmed positive for this pathology
    positive_cases = test_df[test_df[csv_col] == 1]
    if len(positive_cases) == 0:
        print(f"{internal}: no positive cases in test set")
        continue

    test_img = "data/" + positive_cases.iloc[0]["Path"]
    print(f"── {internal} ──────────────────────────")
    print(f"   Image: {test_img}")

    results  = predict(model, test_img, device=device, threshold=0.1)
    heatmaps = generate_all_heatmaps(
        model, test_img, results, device=device, threshold=0.1
    )

    # Save heatmap for this pathology if generated
    if internal in heatmaps:
        out = f"outputs/heatmap_{internal}.png"
        save_heatmap(heatmaps[internal]["heatmap"], out)
        print(f"   Saved: {out}")
        print(f"   Score: {heatmaps[internal]['score']:.3f} "
              f"[{heatmaps[internal]['quality']}]")
    else:
        print(f"   No heatmap generated "
              f"(prob={results[internal]['probability']:.3f})")
    print()

print("Done. Check outputs/ folder for all heatmaps.")