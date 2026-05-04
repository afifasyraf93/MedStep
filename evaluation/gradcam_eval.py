import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(".")

from modules.detection import build_model
from modules.localization import generate_heatmap

# Config
MODEL_PATH  = "models/combined_densenet121_best.pth"
CSV_PATH    = "data/CheXpert-v1.0-small/train.csv"
OUTPUT_DIR  = "evaluation/gradcam_results"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CASES_PER_PATHOLOGY = 5  # 5 cases × 6 pathologies = 30 total

PATHOLOGIES = [
    "pneumonia", "cardiomegaly", "pleural_effusion",
    "pneumothorax", "atelectasis", "lung_mass"
]

CHEXPERT_MAP = {
    "pneumonia":        "Pneumonia",
    "cardiomegaly":     "Cardiomegaly",
    "pleural_effusion": "Pleural Effusion",
    "pneumothorax":     "Pneumothorax",
    "atelectasis":      "Atelectasis",
    "lung_mass":        "Lung Lesion"
}

from torchvision import transforms
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model():
    model = build_model("densenet121", num_classes=6)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)

def get_test_cases(df, pathology, n=5):
    col = CHEXPERT_MAP[pathology]
    positive = df[
        (df[col] == 1) &
        (df["Frontal/Lateral"] == "Frontal")
    ].dropna(subset=[col])
    sample = positive.sample(n=min(n, len(positive)), random_state=42)
    return sample["Path"].tolist()

def run_gradcam_eval():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model()
    df = pd.read_csv(CSV_PATH)

    results = []

    for pathology in PATHOLOGIES:
        cases = get_test_cases(df, pathology, CASES_PER_PATHOLOGY)
        if not cases:
            print(f"No cases found for {pathology}")
            continue

        for i, img_path in enumerate(cases):
            full_path = os.path.join("data", img_path)
            if not os.path.exists(full_path):
                continue

            # Load image
            img = Image.open(full_path).convert("RGB")
            img_resized = img.resize((224, 224))
            original_array = np.array(img_resized).astype(np.float32) / 255.0
            image_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

            # Generate heatmap
            # TODO: call generate_heatmap from localization.py
            # hint: check what parameters generate_heatmap expects
            pathology_index = PATHOLOGIES.index(pathology)
            heatmap, _ = generate_heatmap(
                model,
                image_tensor,
                pathology_index,
                original_array,
                model_name="densenet121"
            )

            if heatmap is None:
                continue

            # Save side by side
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(original_array)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(heatmap)
            axes[1].set_title(f"Grad-CAM: {pathology}")
            axes[1].axis("off")

            save_path = f"{OUTPUT_DIR}/{pathology}_case{i+1}.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=100)
            plt.close()

            results.append({
                "pathology": pathology,
                "case": i + 1,
                "image_path": img_path,
                "output_path": save_path,
                "correct": None  # fill manually after inspection
            })
            print(f"Saved: {save_path}")

    # Save results CSV for manual scoring
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/gradcam_results.csv", index=False)
    print(f"\nDone. {len(results)} cases saved to {OUTPUT_DIR}/")
    print("Open the images and fill in 'correct' column (1=correct, 0=wrong)")

if __name__ == "__main__":
    run_gradcam_eval()