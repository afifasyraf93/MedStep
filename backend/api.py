import os
import sys
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi import Request
from PIL import Image
import io
import base64
import json
from datetime import datetime, timezone, timedelta
SGT = timezone(timedelta(hours=8))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.detection import build_model, PATHOLOGIES, TRANSFORM
from modules.localization import generate_all_heatmaps
from modules.retrieval import load_retrieval_system, retrieve
from modules.explanation import generate_report
from fastapi import Depends, Header
from fastapi.responses import FileResponse
from fastapi.responses import Response
from database.db import get_db, init_db
from sqlalchemy.orm import Session
from sqlalchemy import or_
import auth.auth as auth
from database.history import save_history, get_history, delete_history
from database.db import get_db, init_db, CXRCase, User, History, Heatmap
from groq import Groq
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading retrieval system...")
faiss_index, metadata, embedder = load_retrieval_system()

print("All models loaded. API ready.")

init_db()

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
async def analyze(request: Request, db: Session = Depends(get_db),
                  file: UploadFile = File(...),
                  mode: str = "ai"):
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

    try:
        token = request.headers.get("token")
        if token:
            user_id = auth.verify_session(db, token)

             # Save heatmaps to DB instead of disk
            history_record = save_history(
                db,
                user_id      = user_id,
                report       = str(report),
                detections   = json.dumps(detections),
                heatmaps_dir = None,
                case_name    = "Second Opinion" if mode == "second_opinion" else None
            )

            if heatmaps:
                for pathology, b64_str in heatmaps.items():
                    if b64_str:
                        img_bytes = base64.b64decode(b64_str)
                        db.add(Heatmap(
                            history_id=history_record.history_id,
                            pathology=pathology,
                            image_data=img_bytes
                        ))
                db.commit()

            # THEN save history ONCE
            case_name = "Second Opinion" if mode == "second_opinion" else None
            save_history(
                db,
                user_id      = user_id,
                report       = str(report),
                detections   = json.dumps(detections),
                heatmaps_dir = heatmaps_dir,
                case_name    = case_name
            )

            # Update progress
            user = db.query(User).filter(User.user_id == user_id).first()
            user.analyses_count = (user.analyses_count or 0) + 1
            today = datetime.now(SGT).replace(tzinfo=None).date()
            if user.last_active:
                last = user.last_active.date()
                if last == today:
                    pass
                elif (today - last).days == 1:
                    user.streak_days = (user.streak_days or 0) + 1
                else:
                    user.streak_days = 1
            else:
                user.streak_days = 1
            user.last_active = datetime.now(SGT).replace(tzinfo=None)
            db.commit()
    except Exception:
        pass

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


@app.post("/register")
def register(username: str, email: str, password: str, db: Session = Depends(get_db)):
    # TODO: call auth.register(db, username, email, password)
    # wrap in try/except ValueError — return {"error": str(e)} with status 400
    # on success return {"message": "registered successfully"}
    try:
        auth.register(db, username, email, password)
        return {"message": "registered successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/login")
def login(username: str, password: str, db: Session = Depends(get_db)):
    # TODO: call auth.login(db, username, password)
    # wrap in try/except ValueError — return {"error": str(e)} with status 400
    # on success return {"token": token}
    try:
        token = auth.login(db, username, password)
        return {"token": token}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/logout")
def logout(token: str = Header(...), db: Session = Depends(get_db)):
    # TODO: call auth.logout(db, token)
    # return {"message": "logged out"}
    try:
        auth.logout(db, token)
        return {"message": "logged out"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/history")
def history(token: str = Header(...), db: Session = Depends(get_db)):
    try:
        user_id = auth.verify_session(db, token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    results = get_history(db, user_id)
    return {"history": [
        {
            "history_id":  h.history_id,
            "timestamp":   str(h.timestamp),
            "report":      h.report,
            "case_name":   h.case_name,
            "patient_ref": h.patient_ref,
            "notes":       h.notes,
            "detections":  h.detections,
            "has_heatmaps": len(h.heatmaps) > 0
        }
        for h in results
        if not (h.report and h.report.startswith("SECOND_OPINION"))
    ]}


@app.delete("/history/{history_id}")
def delete_history_entry(history_id: int, token: str = Header(...), db: Session = Depends(get_db)):
    try:
        user_id = auth.verify_session(db, token)
        deleted = delete_history(db, history_id, user_id)

        if deleted:
            user = db.query(User).filter(User.user_id == user_id).first()
            if user and user.analyses_count > 0:
                user.analyses_count -= 1
                db.commit()

        return {"deleted": deleted}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/history/{history_id}/update")
def update_history(
    history_id: int,
    case_name: str = "",
    patient_ref: str = "",
    notes: str = "",
    token: str = Header(...),
    db: Session = Depends(get_db)
):
    try:
        user_id = auth.verify_session(db, token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    h = db.query(History).filter(
        History.history_id == history_id,
        History.user_id == user_id
    ).first()

    if not h:
        raise HTTPException(status_code=404, detail="Not found")

    if case_name: h.case_name = case_name
    if patient_ref: h.patient_ref = patient_ref
    if notes: h.notes = notes
    db.commit()
    return {"updated": True}    

@app.get("/heatmap")
def get_heatmap(history_id: int, pathology: str, db: Session = Depends(get_db)):
    record = db.query(Heatmap).filter(
        Heatmap.history_id == history_id,
        Heatmap.pathology == pathology
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Heatmap not found")
    return Response(content=record.image_data, media_type="image/png")

@app.get("/library")
def get_library(pathologies: str = "", db: Session = Depends(get_db)):
    # pathologies is a comma-separated string e.g. "pneumonia,cardiomegaly"
    # if empty, return all cases
    # if specified, filter rows where ANY of those columns == 1
    # return list of dicts
    # TODO: split pathologies string into list
    # hint: [p.strip() for p in pathologies.split(",") if p.strip()]
    pathology_list = [p.strip() for p in pathologies.split(",") if p.strip()]

    # TODO: if pathology_list is empty, query all cases
    # if not empty, filter with or_(*filters)
    if not pathology_list:
        cases = db.query(CXRCase).all()
    else:
        filters = [getattr(CXRCase, p) == 1 for p in pathology_list]
        cases = db.query(CXRCase).filter(or_(*filters)).all()

    # TODO: return list of dicts
    # each dict: image_id, patient_id, image_path + all 6 pathology columns
    return [
        {
            "image_id":   c.image_id,
            "patient_id": c.patient_id,
            "image_path": c.image_path,
            "labels": {
                # TODO: fill in all 6 pathology keys from the CXRCase object
                "pneumonia":        c.pneumonia,
                "cardiomegaly":     c.cardiomegaly,
                "pleural_effusion": c.pleural_effusion,
                "pneumothorax":     c.pneumothorax,
                "atelectasis":      c.atelectasis,
                "lung_mass":        c.lung_mass,
            }
        }
        for c in cases
    ]

load_dotenv("api.env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.get("/library/stats")
def get_library_stats(db: Session = Depends(get_db)):
    """Return count of cases per pathology."""
    from sqlalchemy import func
    stats = {}
    for pathology in ["pneumonia", "cardiomegaly", "pleural_effusion",
                      "pneumothorax", "atelectasis", "lung_mass"]:
        # TODO: count rows where pathology column == 1
        count = db.query(CXRCase).filter(
            getattr(CXRCase, pathology) == 1
        ).count()
        stats[pathology] = count

    # Count cases where ALL pathology columns == 0
    from sqlalchemy import and_
    no_finding_count = db.query(CXRCase).filter(
        and_(
            CXRCase.pneumonia == 0,
            CXRCase.cardiomegaly == 0,
            CXRCase.pleural_effusion == 0,
            CXRCase.pneumothorax == 0,
            CXRCase.atelectasis == 0,
            CXRCase.lung_mass == 0,
        )
    ).count()
    stats["no_finding"] = no_finding_count

    return stats

@app.post("/second-opinion/save")
def save_second_opinion(
    token: str = Header(...),
    db: Session = Depends(get_db),
    pathology: str = "",
    student_correct: bool = False
):
    try:
        user_id = auth.verify_session(db, token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save as a special history entry
    report = f"SECOND_OPINION|{pathology}|{student_correct}"
    save_history(db, user_id=user_id, report=report)
    return {"saved": True}

@app.get("/second-opinion/stats")
def get_second_opinion_stats(
    token: str = Header(...),
    db: Session = Depends(get_db)
):
    try:
        user_id = auth.verify_session(db, token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    history = get_history(db, user_id)
    
    # Filter second opinion entries
    so_entries = [h for h in history 
                  if h.report and h.report.startswith("SECOND_OPINION")]

    # Count correct/total per pathology
    stats = {p: {"correct": 0, "total": 0} for p in [
        "pneumonia", "cardiomegaly", "pleural_effusion",
        "pneumothorax", "atelectasis", "lung_mass"
    ]}

    for entry in so_entries:
        # TODO: parse report string "SECOND_OPINION|pathology|True/False"
        # hint: parts = entry.report.split("|")
        # pathology = parts[1], correct = parts[2] == "True"
        parts = entry.report.split("|")
        if len(parts) == 3:
            pathology = parts[1]
            correct = parts[2] == "True"
            if pathology in stats:
                stats[pathology]["total"] += 1
                if correct:
                    stats[pathology]["correct"] += 1

    # Calculate accuracy per pathology
    accuracy = {
        p: round(v["correct"] / v["total"] * 100, 1) if v["total"] > 0 else 0
        for p, v in stats.items()
    }
    return accuracy

@app.get("/encyclopedia/{pathology}")
def encyclopedia(pathology: str, token: str = Header(...),
                 db: Session = Depends(get_db)):
    try:
        auth.verify_session(db, token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prompt = f"""You are a radiology educator. Explain {pathology.replace('_', ' ')} 
    for a medical student learning chest X-ray interpretation. Include:
    1. Definition and pathophysiology
    2. Chest X-ray findings to look for
    3. Clinical significance
    4. Common mistakes students make when reading CXRs for this condition
    Keep it educational, concise and practical.
    """

    # TODO: call groq_client.chat.completions.create
    # text-only, same pattern as generate_impression in explanation.py
    # model = "meta-llama/llama-4-scout-17b-16e-instruct"
    # max_tokens = 1024
    # return {"content": response_text}
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        max_tokens=1024
    )
    return {"content": response.choices[0].message.content.strip()}

@app.get("/progress")
def get_progress(token: str = Header(...), db: Session = Depends(get_db)):
    try:
        user_id = auth.verify_session(db, token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    user = db.query(User).filter(User.user_id == user_id).first()
    
    from database.history import get_history
    history = get_history(db, user_id)
    actual_count = len([h for h in history 
                    if not (h.report and h.report.startswith("SECOND_OPINION"))])

    return {
        "username":       user.username,
        "analyses_count": actual_count,
        "streak_days":    user.streak_days,
        "last_active":    str(user.last_active) if user.last_active else None
    }

if __name__ == "__main__":
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=False)