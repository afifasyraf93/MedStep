import torch
import numpy as np
from PIL import Image
from modules.detection import build_model, TRANSFORM, PATHOLOGIES
from modules.localization import generate_heatmap

# Load model
MODEL_PATH = "models/combined_densenet121_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model("densenet121")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Load a test image — use any image from your dataset
TEST_IMAGE = r"D:\Projek\MedStep\data\CheXpert-v1.0-small\train\patient21923\study2\view1_frontal.jpg"

img = Image.open(TEST_IMAGE).convert("RGB")
img_resized = img.resize((224, 224))
original_array = np.array(img_resized).astype(np.float32) / 255.0

image_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

# Test heatmap for pathology index 0 (pneumonia)
print("Generating heatmap...")
heatmap_array, base64_str = generate_heatmap(
    model, image_tensor, 
    pathology_index=2,  # pleural_effusion — most common, likely to activate
    original_image_array=original_array,
    model_name="densenet121"
)

print(f"Heatmap shape: {heatmap_array.shape}")
print(f"Base64 length: {len(base64_str)}")
print("✓ Grad-CAM working")

# Save output to verify visually
from PIL import Image
import base64, io
img_bytes = base64.b64decode(base64_str)
result_img = Image.open(io.BytesIO(img_bytes))
result_img.save("test_heatmap_output.png")
print("Saved → test_heatmap_output.png")