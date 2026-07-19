from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
from fastapi.responses import Response
import os

app = FastAPI()

app.mount("/data", StaticFiles(directory="D:/Projek/MedStep/data"), name="data")
app.mount("/static", StaticFiles(directory="frontend_v2/static"), name="static")
templates = Jinja2Templates(directory="frontend_v2/templates")

API_BASE = "http://localhost:8000"

# ── Helper ──────────────────────────────────────────────
def get_token(request: Request) -> str | None:
    return request.cookies.get("token")

def get_username(request: Request) -> str | None:
    return request.cookies.get("username")

# ── Landing ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    if get_token(request):
        return RedirectResponse("/dashboard")
    return templates.TemplateResponse("landing.html", {"request": request})

# ── Login ────────────────────────────────────────────────
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if get_token(request):
        return RedirectResponse("/dashboard")
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, username: str = Form(...),
                password: str = Form(...)):
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{API_BASE}/login",
                                params={"username": username,
                                        "password": password})
    if res.status_code == 200:
        token = res.json()["token"]
        response = RedirectResponse("/dashboard", status_code=302)
        response.set_cookie("token", token)
        response.set_cookie("username", username)
        return response
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": res.json().get("detail", "Login failed")
    })

# ── Register ─────────────────────────────────────────────
@app.post("/register")
async def register(request: Request, username: str = Form(...),
                   email: str = Form(...), password: str = Form(...)):
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{API_BASE}/register",
                                params={"username": username,
                                        "email": email,
                                        "password": password})
    if res.status_code == 200:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "success": "Registered successfully! Please log in."
        })
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": res.json().get("detail", "Register failed")
    })

# ── Logout ───────────────────────────────────────────────
@app.get("/logout")
async def logout(request: Request):
    token = get_token(request)
    if token:
        async with httpx.AsyncClient() as client:
            await client.post(f"{API_BASE}/logout",
                              headers={"token": token})
    response = RedirectResponse("/")
    response.delete_cookie("token")
    response.delete_cookie("username")
    return response

# ── Dashboard ────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    token = get_token(request)
    if not token:
        return RedirectResponse("/login")
    async with httpx.AsyncClient() as client:
        progress = await client.get(f"{API_BASE}/progress",
                                    headers={"token": token})
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "username": get_username(request),
        "progress": progress.json() if progress.status_code == 200 else {}
    })

# ── Analysis ─────────────────────────────────────────────
@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    if not get_token(request):
        return RedirectResponse("/login")
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "username": get_username(request)
    })

# ── Library ──────────────────────────────────────────────
@app.get("/library", response_class=HTMLResponse)
async def library(request: Request, pathologies: str = ""):
    token = get_token(request)
    if not token:
        return RedirectResponse("/login")
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{API_BASE}/library",
                               params={"pathologies": pathologies})
    return templates.TemplateResponse("library.html", {
        "request": request,
        "username": get_username(request),
        "cases": res.json() if res.status_code == 200 else [],
        "selected": pathologies.split(",") if pathologies else []
    })

# ── History ──────────────────────────────────────────────
@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    token = get_token(request)
    if not token:
        return RedirectResponse("/login")
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{API_BASE}/history",
                               headers={"token": token})
    history_data = []
    if res.status_code == 200:
        history_data = [h for h in res.json()["history"]
                        if not (h["report"] and
                                h["report"].startswith("SECOND_OPINION"))]
    return templates.TemplateResponse("history.html", {
        "request": request,
        "username": get_username(request),
        "history": history_data
    })

# ── Encyclopedia ─────────────────────────────────────────
@app.get("/encyclopedia", response_class=HTMLResponse)
async def encyclopedia(request: Request):
    if not get_token(request):
        return RedirectResponse("/login")
    return templates.TemplateResponse("encyclopedia.html", {
        "request": request,
        "username": get_username(request)
    })

# ── API proxy for analysis (called via JS fetch) ─────────
@app.post("/api/analyze")
async def api_analyze(request: Request, file: UploadFile = File(...),
                      threshold: float = Form(0.5),
                      mode: str = Form("ai")):
    token = get_token(request)
    if not token:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    async with httpx.AsyncClient(timeout=120) as client:
        res = await client.post(
            f"{API_BASE}/analyze",
            files={"file": (file.filename, await file.read(), file.content_type)},
            headers={"token": token},
            params={"mode": mode}
        )
    return JSONResponse(res.json(), status_code=res.status_code)

@app.get("/api/library/stats")
async def proxy_library_stats():
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{API_BASE}/library/stats")
    return res.json()

@app.get("/api/second-opinion/stats")
async def proxy_so_stats(request: Request):
    token = get_token(request)
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{API_BASE}/second-opinion/stats",
                               headers={"token": token})
    return res.json()

@app.get("/api/history")
async def proxy_history(request: Request):
    token = get_token(request)
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{API_BASE}/history",
                               headers={"token": token})
    return res.json()

@app.post("/api/second-opinion/save")
async def proxy_so_save(request: Request, pathology: str = "",
                        student_correct: bool = False):
    token = get_token(request)
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{API_BASE}/second-opinion/save",
            headers={"token": token},
            params={"pathology": pathology,
                    "student_correct": student_correct}
        )
    return res.json()

@app.delete("/api/history/{history_id}")
async def proxy_delete_history(history_id: int, request: Request):
    token = get_token(request)
    async with httpx.AsyncClient() as client:
        res = await client.delete(
            f"{API_BASE}/history/{history_id}",
            headers={"token": token}
        )
    return res.json()

@app.get("/api/encyclopedia/{pathology}")
async def proxy_encyclopedia(pathology: str, request: Request):
    token = get_token(request)
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.get(
            f"{API_BASE}/encyclopedia/{pathology}",
            headers={"token": token}
        )
    return res.json()

@app.post("/api/history/{history_id}/update")
async def proxy_update_history(history_id: int, request: Request):
    token = get_token(request)
    body = await request.json()
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{API_BASE}/history/{history_id}/update",
            headers={"token": token},
            params={
                "case_name":   body.get("case_name", ""),
                "patient_ref": body.get("patient_ref", ""),
                "notes":       body.get("notes", "")
            }
        )
    return res.json()

@app.get("/api/library")
async def proxy_library(pathologies: str = ""):
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{API_BASE}/library",
                               params={"pathologies": pathologies})
    return res.json()

@app.post("/api/generate-pdf")
async def generate_pdf(request: Request):
    token = get_token(request)
    body = await request.json()

    import sys
    sys.path.append(".")
    from frontend_v2.utils.pdf_export import generate_pdf_report
    import base64

    # Decode image from base64 sent from frontend
    image_bytes = base64.b64decode(body.get("image_b64", ""))
    detections = body.get("detections", {})
    report = body.get("report", {})
    username = get_username(request) or "Student"
    threshold = body.get("threshold", 0.5)

    pdf_bytes = generate_pdf_report(
        image_bytes=image_bytes,
        detections=detections,
        report=report if isinstance(report, dict)
               else {"findings": str(report), "impression": ""},
        username=username,
        threshold=threshold
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=medstep_report.pdf"}
    )

@app.get("/api/image")
async def proxy_image(path: str):
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{API_BASE}/image", params={"path": path})
    from fastapi.responses import Response
    return Response(content=res.content, media_type=res.headers.get("content-type", "image/jpeg"))

@app.get("/api/heatmap")
async def proxy_heatmap(history_id: int, pathology: str):
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{API_BASE}/heatmap", params={"history_id": history_id, "pathology": pathology})
    return Response(content=res.content, media_type="image/png")

@app.get("/api/health")
async def proxy_health():
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{API_BASE}/health")
    return res.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8501, reload=True, 
                reload_dirs=["frontend_v2"])