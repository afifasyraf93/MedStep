import os
import sys
import json
import uuid
import shutil
import torch
from pathlib import Path
from fastapi import (
    FastAPI, UploadFile, File,
    HTTPException, Depends, Header
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from database.db import init_db
from auth.auth import (
    register_user, login_user,
    verify_session, logout_user, get_user_by_id
)
from database.history import save_history, get_user_history, get_history_by_id
from modules.detection import load_model, predict, PATHOLOGIES
from modules.localization import (
    generate_all_heatmaps, heatmap_to_base64
)
from modules.retrieval import load_faiss_index, retrieve_similar
from modules.explanation import generate_explanation
from fastapi.responses import FileResponse

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "MedStep API",
    description = "AI-assisted chest X-ray interpretation for medical students",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)

# ── Global state ──────────────────────────────────────────────────────────────

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL      = None
FAISS_IDX  = None
FAISS_META = None
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global MODEL, FAISS_IDX, FAISS_META

    print(f"Starting MedStep API on device: {DEVICE}")

    # Init database
    init_db()

    # Load model
    MODEL = load_model("models/densenet121_best.pth", device=DEVICE)
    print("Model loaded")

    # Load FAISS index
    FAISS_IDX, FAISS_META = load_faiss_index("faiss_index")
    print("FAISS index loaded")

    print("MedStep API ready")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email:    str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


# ── Auth dependency ───────────────────────────────────────────────────────────

def get_current_user(authorization: str = Header(None)):
    """Dependency — extract and verify session token from header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="No token provided")

    # Strip "Bearer " prefix if present
    token = authorization.replace("Bearer ", "").strip()

    valid, user_id = verify_session(token)
    if not valid:
        raise HTTPException(
            status_code=401, detail="Invalid or expired session"
        )
    return user_id


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":  "running",
        "device":  DEVICE,
        "model":   "densenet121",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": MODEL is not None,
        "faiss_loaded": FAISS_IDX is not None,
        "device":       DEVICE
    }


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.post("/auth/register")
def register(req: RegisterRequest):
    ok, msg = register_user(req.username, req.email, req.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@app.post("/auth/login")
def login(req: LoginRequest):
    ok, result = login_user(req.username, req.password)
    if not ok:
        raise HTTPException(status_code=401, detail=result)
    return {"success": True, "token": result}


@app.post("/auth/logout")
def logout(user_id: int = Depends(get_current_user),
           authorization: str = Header(None)):
    token = authorization.replace("Bearer ", "").strip()
    logout_user(token)
    return {"success": True, "message": "Logged out"}


@app.get("/auth/me")
def me(user_id: int = Depends(get_current_user)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ── Analysis route ────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    file:    UploadFile = File(...),
    user_id: int        = Depends(get_current_user)
):
    """
    Main analysis endpoint.
    Accepts chest X-ray image, returns full analysis.
    """
    # ── Validate file ─────────────────────────────────────────
    allowed = {".jpg", ".jpeg", ".png"}
    ext     = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not allowed. Use JPG or PNG."
        )

    if file.size and file.size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum 10MB."
        )

    # ── Save uploaded file ────────────────────────────────────
    file_id   = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # ── Step 1: Detection ─────────────────────────────────
        print(f"Analyzing {file_path}...")
        detection_results = predict(
            MODEL, file_path, device=DEVICE, threshold=0.3
        )

        detected_pathologies = [
            k for k, v in detection_results.items()
            if v["detected"]
        ]
        print(f"  Detected: {detected_pathologies}")

        # ── Step 2: Grad-CAM ──────────────────────────────────
        heatmaps = generate_all_heatmaps(
            MODEL, file_path, detection_results,
            device=DEVICE, threshold=0.3
        )

        # Convert heatmaps to base64 for JSON response
        heatmaps_b64 = {}
        best_heatmap_pil = None
        best_score       = 0.0

        for pathology, data in heatmaps.items():
            heatmaps_b64[pathology] = {
                "image":   heatmap_to_base64(data["heatmap"]),
                "score":   data["score"],
                "quality": data["quality"]
            }
            if data["score"] > best_score:
                best_score       = data["score"]
                best_heatmap_pil = data["heatmap"]

        # ── Step 3: FAISS retrieval ───────────────────────────
        similar_cases = retrieve_similar(
            MODEL, file_path, FAISS_IDX, FAISS_META,
            top_k=3, device=DEVICE
        )

        # Clean up similar cases for JSON
        similar_clean = []
        for case in similar_cases:
            positive_labels = [
                k for k, v in case["labels"].items() if v == 1
            ]
            similar_clean.append({
                "rank":       case["rank"],
                "labels":     positive_labels,
                "path":       case["path"],
                "similarity": round(case["similarity"], 3)
            })

        # ── Step 4: Explanation ───────────────────────────────
        explanation = generate_explanation(
            image_path        = file_path,
            detection_results = detection_results,
            heatmap_pil       = best_heatmap_pil,
            threshold         = 0.3
        )

        # ── Step 5: Save to history ───────────────────────────
        history_id = save_history(
            user_id           = user_id,
            image_path        = file_path,
            detection_results = detection_results,
            findings          = explanation.get("findings", ""),
            impression        = explanation.get("impression", ""),
            full_report       = explanation.get("full_report", "")
        )

        # ── Build response ────────────────────────────────────
        return JSONResponse({
            "success":    True,
            "history_id": history_id,
            "detection":  detection_results,
            "detected":   detected_pathologies,
            "heatmaps":   heatmaps_b64,
            "similar_cases": similar_clean,
            "explanation": {
                "findings":    explanation.get("findings", ""),
                "impression":  explanation.get("impression", ""),
                "full_report": explanation.get("full_report", "")
            }
        })

    except Exception as e:
        # Clean up uploaded file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


# ── History routes ────────────────────────────────────────────────────────────

@app.get("/history")
def history(user_id: int = Depends(get_current_user)):
    """Get analysis history for current user."""
    entries = get_user_history(user_id, limit=20)
    return {"success": True, "history": entries}


@app.get("/history/{history_id}")
def history_detail(
    history_id: int,
    user_id:    int = Depends(get_current_user)
):
    """Get single history entry."""
    entry = get_history_by_id(history_id, user_id)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail="History entry not found"
        )
    return {"success": True, "entry": entry}

@app.post("/history/{history_id}/delete")
def delete_history(
    history_id: int,
    user_id:    int = Depends(get_current_user)
):
    """Delete a history entry. Users can only delete their own."""
    from database.db import get_session, History as HistoryModel

    db = get_session()
    try:
        entry = db.query(HistoryModel).filter_by(
            history_id = history_id,
            user_id    = user_id
        ).first()

        if not entry:
            raise HTTPException(
                status_code = 404,
                detail      = "History entry not found"
            )

        # Clean up uploaded image file if it exists
        if entry.image_path and os.path.exists(entry.image_path):
            try:
                os.remove(entry.image_path)
            except Exception:
                pass  # Don't fail if file already gone

        db.delete(entry)
        db.commit()
        return {"success": True, "message": "Entry deleted"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/image")
def serve_image(path: str, user_id: int = Depends(get_current_user)):
    """Serve a training image by path for similar case display."""
    # Security: only allow paths within the data directory
    full_path = os.path.join("data", path)
    full_path = os.path.normpath(full_path)

    # Prevent path traversal attacks
    if not full_path.startswith(os.path.normpath("data")):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(full_path, media_type="image/jpeg")

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = False
    )