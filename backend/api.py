import os
import sys
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.detection import build_model, PATHOLOGIES, TRANSFORM
from modules.localization import generate_all_heatmaps
from modules.retrieval import load_retrieval_system, retrieve
from modules.explanation import generate_report

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/combined_densenet121_best.pth"
MODEL_NAME  = "densenet121"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="MedStep API", version="1.1.1")

# Load all models at startup — once only
print("Loading detection model...")
detection_model = build_model(MODEL_NAME)
detection_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE))
detection_model = detection_model.to(DEVICE)
detection_model.eval()

print("Loading retrieval system...")
faiss_index, metadata, embedder = load_retrieval_system()

print("All models loaded. API ready.")


@app.get("/")
def root():
    return {"message": "MedStep API", "version": "1.1.1", "status": "running"}


@app.get("/health")
def health():
    return {
        "status":       "healthy",
        "model":        MODEL_NAME,
        "model_loaded": detection_model is not None,
        "faiss_vectors": faiss_index.ntotal,
        "device":       str(DEVICE)
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Main pipeline endpoint.
    Accepts a chest X-ray image and returns:
    - detections: per-pathology probabilities
    - heatmaps: base64 encoded heatmaps per detected pathology
    - similar_cases: top-3 FAISS results
    - report: findings + impression from Groq
    """
    # --- Load and preprocess image ---
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Resize for display and processing
    img_resized = img.resize((224, 224))
    original_array = np.array(img_resized).astype(np.float32) / 255.0

    # Preprocess for model
    image_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    # TODO: Step 1 — Run detection
    # pass image_tensor through detection_model
    # apply sigmoid to get probabilities
    # build detections dict {pathology: float probability}
    # Hint: zip(PATHOLOGIES, probs[0].tolist())
    with torch.no_grad():
        outputs = detection_model(image_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()

    detections = {
        p: round(float(prob), 4)
        for p, prob in zip(PATHOLOGIES, probs[0].tolist())
    }

    # TODO: Step 2 — Generate heatmaps
    # call generate_all_heatmaps with detection_model, image_tensor,
    # detections dict, original_array, MODEL_NAME
    heatmaps = generate_all_heatmaps(
        detection_model, image_tensor,
        detections, original_array, MODEL_NAME
    )

    # TODO: Step 3 — Retrieve similar cases
    # call retrieve with image_tensor, faiss_index, metadata, embedder
    similar_cases = retrieve(image_tensor, faiss_index, metadata, embedder)

    # TODO: Step 4 — Save image temporarily for Groq
    # save img to a temp file in uploads/ folder
    # call generate_report with temp path and detections
    # hint: use img.save(temp_path)
    os.makedirs("uploads", exist_ok=True)
    temp_path = f"uploads/temp_{file.filename}"
    img.save(temp_path)
    report = generate_report(temp_path, detections)

    # TODO: Step 5 — Return JSON response
    # return all results as dict
    return JSONResponse(content={
        "detections":    detections,
        "heatmaps":      heatmaps,
        "similar_cases": similar_cases,
        "report":        report
    })

@app.get("/image")
def serve_image(path: str):
    """Serve training images for similar cases display."""
    # TODO: construct full path from BASE_PATH + path
    # check if file exists
    # return FileResponse
    full_path = os.path.join("data", path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(full_path)


if __name__ == "__main__":
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=False)