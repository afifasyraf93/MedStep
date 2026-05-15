# 🫁 MedStep — AI-Assisted Chest X-Ray Interpretation

> Final Year Project | UiTM CSP650 | 2025–2026

MedStep is a web-based AI system that helps medical students learn chest X-ray interpretation. It detects 6 pathologies using DenseNet121, explains predictions with Grad-CAM heatmaps, and provides AI-generated radiology reports via Groq LLM.

---

## 📸 Screenshots

| Landing Page | Analysis | History |
|---|---|---|
| Split-screen with real CXR background | Detection + Grad-CAM heatmap | Case name, detections, heatmaps, AI report |

---

## ✨ Features

- **AI Detection** — DenseNet121 trained on 75,652 chest X-rays, detecting 6 pathologies with macro AUC 0.788
- **Grad-CAM Heatmaps** — Visual explanation of AI decisions with lung masking and intensity thresholding
- **Second Opinion Mode** — Student diagnoses first, then compares with AI and gets a score
- **FAISS Similarity Search** — Retrieves top-3 similar cases from 60,586 indexed vectors in ~36ms
- **AI Report Generation** — Findings and impression via Groq LLaMA 4 Scout
- **Case Organization** — Label analyses with case name, patient reference, and clinical notes
- **Pathology Encyclopedia** — AI-generated educational content per pathology
- **Case Library** — Browse 500 real CXR cases filterable by pathology
- **Dashboard** — 7 ECharts visualizations including AUC radar, model comparison, analysis timeline
- **PDF Export** — Download formatted analysis report
- **Streak & Progress Tracking** — Gamified learning progress per user

---

## 🧠 Model Performance

| Model | CheXpert | NIH | Combined |
|---|---|---|---|
| **DenseNet121** ✅ | 0.766 | 0.782 | **0.788** |
| ResNet50 | 0.768 | 0.780 | 0.788 |
| EfficientNetB0 | 0.758 | 0.774 | 0.779 |
| MobileNetV3 | 0.745 | 0.764 | 0.767 |

**Per-pathology AUC (DenseNet121, Combined dataset):**

| Pathology | AUC |
|---|---|
| Pneumonia | 0.728 |
| Cardiomegaly | 0.856 |
| Pleural Effusion | 0.856 |
| Pneumothorax | 0.793 |
| Atelectasis | 0.712 |
| Lung Mass | 0.797 |

---

## 🗂️ Project Structure

```
MedStep/
├── run.py                      # Single startup script
├── backend/
│   └── api.py                  # FastAPI REST API (port 8000)
├── frontend_v2/                # Active frontend (Bootstrap 5)
│   ├── main.py                 # FastAPI frontend server (port 8501)
│   ├── templates/              # Jinja2 HTML templates
│   │   ├── base.html
│   │   ├── landing.html
│   │   ├── login.html
│   │   ├── dashboard.html
│   │   ├── analysis.html
│   │   ├── library.html
│   │   ├── history.html
│   │   └── encyclopedia.html
│   └── static/
│       ├── css/custom.css
│       └── js/main.js
├── modules/
│   ├── detection.py            # DenseNet121 inference
│   ├── localization.py         # Grad-CAM + lung mask
│   ├── retrieval.py            # FAISS similarity search
│   ├── explanation.py          # Groq report generation
│   ├── evaluation.py           # AUC, F1, precision, recall
│   └── dataset.py              # Dataset loader
├── database/
│   ├── db.py                   # SQLAlchemy models
│   ├── history.py              # History CRUD
│   └── seed.py                 # Seed 500 CXR cases
├── auth/
│   └── auth.py                 # Register, login, session management
├── train.py                    # Model training loop
├── evaluation/
│   ├── gradcam_eval.py         # Grad-CAM localization evaluation
│   └── retrieval_speed_test.py # FAISS speed benchmark
├── experiments/
│   └── results/                # Per-model JSON experiment results
├── models/                     # Trained model weights (.pth)
├── faiss_index/                # FAISS index + metadata
├── data/                       # CheXpert + NIH datasets
└── archive/
    └── frontend_streamlit/     # Archived Streamlit frontend
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Detection | DenseNet121 (PyTorch) |
| Localization | pytorch-grad-cam |
| Retrieval | FAISS IndexFlatL2, 1024-dim embeddings |
| Report Generation | Groq API — LLaMA 4 Scout 17B |
| Backend API | FastAPI + Uvicorn (port 8000) |
| Frontend | FastAPI + Jinja2 + Bootstrap 5.3 |
| Charts | ECharts 5.4 |
| Database | SQLite + SQLAlchemy |
| Authentication | bcrypt + passlib |

---

## 🚀 Getting Started

### Prerequisites

```
Python 3.x
CUDA-compatible GPU (recommended)
CheXpert dataset (place in data/CheXpert-v1.0-small/)
NIH ChestX-ray dataset (place in data/NIH/)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/afifasyraf93/MedStep.git
cd MedStep

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create `api.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Database Setup

```bash
# Initialize database
python database/db.py

# Seed 500 sample cases
python -m database.seed
```

### Running the Application

```bash
python run.py
```

Opens at `http://localhost:8501`

API docs available at `http://localhost:8000/docs`

---

## 📊 Evaluation Results

| Metric | Result | Target | Status |
|---|---|---|---|
| Grad-CAM Localization | 85.7% (24/28 cases) | ≥ 80% | ✅ PASS |
| Retrieval Speed (avg) | 0.036s | < 1.0s | ✅ PASS |
| Retrieval Speed (max) | 0.303s | < 1.0s | ✅ PASS |
| SUS Usability Score | Pending | ≥ 65 | ⏳ |

---

## 📁 Dataset

| Dataset | Train | Test | Total |
|---|---|---|---|
| CheXpert | 48,005 | 11,995 | 60,000 |
| NIH | 12,508 | 3,144 | 15,652 |
| **Combined** | **60,586** | **15,066** | **75,652** |

Labels: `pneumonia`, `cardiomegaly`, `pleural_effusion`, `pneumothorax`, `atelectasis`, `lung_mass`

Uncertain labels (`-1`) treated as negative (`0`).

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/register` | Register new user |
| POST | `/login` | Login and get session token |
| POST | `/logout` | Logout |
| POST | `/analyze` | Run full CXR analysis pipeline |
| GET | `/history` | Get user analysis history |
| DELETE | `/history/{id}` | Delete history entry |
| POST | `/history/{id}/update` | Update case details |
| GET | `/progress` | Get user progress and streak |
| GET | `/library` | Browse CXR case library |
| GET | `/library/stats` | Pathology distribution stats |
| GET | `/encyclopedia/{pathology}` | AI educational content |
| GET | `/second-opinion/stats` | User diagnostic accuracy |
| GET | `/image` | Serve CXR image |
| GET | `/heatmap` | Serve saved Grad-CAM heatmap |
| GET | `/health` | API health check |

---

## 📝 License

This project is developed as a Final Year Project at UiTM (CSP650). For educational purposes only — not a substitute for professional medical diagnosis.

---

## 👤 Author

Student, Faculty of Computer and Mathematical Sciences  
Universiti Teknologi MARA (UiTM)  
CSP650 Final Year Project | 2025–2026
