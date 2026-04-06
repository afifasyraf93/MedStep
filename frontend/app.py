import streamlit as st
import requests
import base64
import json
from PIL import Image
from io import BytesIO
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="MedStep",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  THEME — Design tokens live here. Edit :root to retheme the whole app.
# ══════════════════════════════════════════════════════════════════════════════

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:           #080e1a;
    --bg-1:         #0d1526;
    --bg-2:         #121e35;
    --bg-3:         #19263f;

    --blue:         #3b82f6;
    --blue-dim:     rgba(59,130,246,0.15);
    --blue-border:  rgba(59,130,246,0.3);
    --teal:         #2dd4bf;
    --teal-dim:     rgba(45,212,191,0.12);

    --text:         #e2e8f0;
    --text-2:       #94a3b8;
    --text-3:       #4b5e78;

    --border:       rgba(255,255,255,0.07);
    --border-hi:    rgba(59,130,246,0.25);

    --red:          #f87171;
    --red-dim:      rgba(248,113,113,0.1);
    --green:        #4ade80;
    --green-dim:    rgba(74,222,128,0.1);
    --amber:        #fbbf24;
    --amber-dim:    rgba(251,191,36,0.1);

    --r:            8px;
    --r-lg:         12px;
    --font:         'DM Sans', sans-serif;
    --mono:         'DM Mono', monospace;
}

/* ─ Reset / base ─────────────────────────────────────────────────────────── */
html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; font-family: var(--font) !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
button[data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; }
* { font-family: var(--font) !important; box-sizing: border-box; }
h1,h2,h3,h4,h5 { color: var(--text) !important; font-family: var(--font) !important; }
p, li, span, label { color: var(--text-2) !important; }
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* ─ Streamlit overrides ──────────────────────────────────────────────────── */
.stTextInput input, .stTextInput textarea {
    background: var(--bg-2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: var(--r) !important;
    font-family: var(--font) !important;
}
.stTextInput input:focus { border-color: var(--blue) !important; box-shadow: 0 0 0 3px var(--blue-dim) !important; }
.stTextInput label, .stFileUploader label, .stSelectbox label {
    color: var(--text-2) !important; font-size: 12px !important; font-weight: 500 !important;
    text-transform: uppercase; letter-spacing: .05em;
}
.stSelectbox div[data-baseweb="select"] > div {
    background: var(--bg-2) !important; border-color: var(--border) !important; border-radius: var(--r) !important;
}
.stSelectbox div[data-baseweb="select"] * { color: var(--text) !important; }
[data-testid="stFileUploader"] {
    background: var(--bg-2) !important; border: 1px dashed var(--border-hi) !important;
    border-radius: var(--r-lg) !important;
}
[data-testid="stFileUploader"] * { color: var(--text-2) !important; }

/* Buttons */
.stButton > button {
    background: var(--bg-2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: var(--r) !important;
    font-family: var(--font) !important; font-size: 13px !important;
    padding: 6px 16px !important; transition: all .15s;
}
.stButton > button:hover { border-color: var(--blue) !important; color: var(--blue) !important; }
.stButton > button[kind="primary"] {
    background: var(--blue) !important; color: #fff !important;
    border-color: var(--blue) !important; font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover { opacity: .88 !important; }
.stDownloadButton button {
    background: var(--bg-2) !important; color: var(--teal) !important;
    border: 1px solid rgba(45,212,191,.35) !important; border-radius: var(--r) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-2) !important; border-radius: var(--r) !important;
    border: 1px solid var(--border) !important; gap: 2px; padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--text-2) !important;
    border-radius: 6px !important; font-size: 13px !important;
    font-weight: 500 !important; padding: 6px 14px !important; border: none !important;
}
.stTabs [aria-selected="true"] { background: var(--blue) !important; color: #fff !important; font-weight: 600 !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

/* Expanders */
details > summary {
    background: var(--bg-2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: var(--r) !important;
    padding: 10px 14px !important; font-size: 13px !important; font-weight: 500 !important;
    cursor: pointer;
}
details[open] > summary { border-bottom-left-radius: 0 !important; border-bottom-right-radius: 0 !important; }
details > summary p, details > summary span { color: var(--text) !important; }
.streamlit-expanderContent, details > div {
    background: var(--bg-2) !important; border: 1px solid var(--border) !important;
    border-top: none !important; border-radius: 0 0 var(--r) var(--r) !important;
    padding: 14px !important;
}

/* Alerts */
[data-testid="stAlert"] { border-radius: var(--r) !important; }

/* Dataframe */
.stDataFrame { background: var(--bg-1) !important; }
.stDataFrame * { color: var(--text) !important; font-family: var(--mono) !important; background: transparent !important; }
[data-testid="stDataFrame"] th {
    background: var(--bg-2) !important; color: var(--blue) !important;
    font-family: var(--font) !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: .06em;
}

/* Caption */
.stCaption p, [data-testid="stCaptionContainer"] p { color: var(--text-3) !important; font-size: 11px !important; }
.stMarkdown p { color: var(--text-2) !important; }

/* Spinner */
[data-testid="stSpinner"] p { color: var(--text-2) !important; }

/* ─ Custom components ─────────────────────────────────────────────────────── */

/* Top nav bar */
.topnav {
    display: flex; align-items: center;
    background: var(--bg-1); border-bottom: 1px solid var(--border);
    padding: 0 28px; height: 52px; gap: 6px; position: sticky; top: 0; z-index: 100;
}
.topnav-logo { font-size: 14px; font-weight: 700; color: var(--text) !important; margin-right: 20px; display: flex; align-items: center; gap: 7px; }
.topnav-logo span { color: var(--blue) !important; }
.topnav-spacer { flex: 1; }
.topnav-user { font-size: 12px; color: var(--text-2) !important; display: flex; align-items: center; gap: 6px; }
.topnav-dot { width: 6px; height: 6px; background: var(--green); border-radius: 50%; display: inline-block; }

/* Page wrapper with left padding */
.page-wrap { padding: 28px 32px 48px; }

/* Section eyebrow */
.eyebrow {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .12em; color: var(--blue) !important;
    margin-bottom: 6px; display: block;
}

/* Card */
.card {
    background: var(--bg-1); border: 1px solid var(--border);
    border-radius: var(--r-lg); padding: 20px 22px; margin-bottom: 14px;
}
.card-title {
    font-size: 13px; font-weight: 600; color: var(--text) !important;
    margin: 0 0 10px; display: flex; align-items: center; gap: 7px;
}
.card p, .card li { font-size: 13px; line-height: 1.65; color: var(--text-2) !important; margin: 0; }
.card ul { padding-left: 16px; margin: 0; }
.card li { margin-bottom: 4px; }
.card strong { color: var(--text) !important; }

/* Stat tile */
.stat-tile {
    background: var(--bg-1); border: 1px solid var(--border);
    border-radius: var(--r-lg); padding: 18px 16px; text-align: center;
}
.stat-val { font-family: var(--mono); font-size: 24px; font-weight: 500; color: var(--blue) !important; display: block; }
.stat-lbl { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--text-3) !important; margin-top: 3px; display: block; }

/* Badges */
.badge-pos { display: inline-flex; align-items: center; gap: 4px; background: var(--red-dim); color: var(--red) !important; border: 1px solid rgba(248,113,113,.25); padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; margin: 2px; }
.badge-neg { display: inline-flex; align-items: center; gap: 4px; background: var(--green-dim); color: var(--green) !important; border: 1px solid rgba(74,222,128,.25); padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; margin: 2px; }

/* Prob bar */
.pbar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.pbar-label { width: 136px; flex-shrink: 0; font-size: 12px; color: var(--text-2); }
.pbar-label.hot { color: var(--text) !important; font-weight: 600; }
.pbar-track { flex: 1; background: var(--bg-3); border-radius: 4px; height: 7px; overflow: hidden; }
.pbar-fill { height: 100%; border-radius: 4px; }
.pbar-pct { width: 40px; text-align: right; font-family: var(--mono); font-size: 11px; color: var(--text-3); flex-shrink: 0; }

/* Report box */
.rbox { background: var(--bg-2); border-left: 2px solid var(--blue); padding: 14px 18px; border-radius: 0 var(--r) var(--r) 0; margin: 8px 0; font-size: 13px; line-height: 1.75; color: var(--text-2) !important; }

/* Disclaimer */
.disc { background: var(--amber-dim); border: 1px solid rgba(251,191,36,.2); border-radius: var(--r); padding: 8px 14px; font-size: 11px; color: var(--amber) !important; margin-top: 12px; }

/* Sim bar */
.sim-track { background: var(--bg-3); border-radius: 4px; height: 5px; margin-top: 5px; }
.sim-fill { height: 100%; background: linear-gradient(90deg,var(--blue),var(--teal)); border-radius: 4px; }

/* Auth card */
.auth-wrap { max-width: 380px; margin: 60px auto 0; }
.auth-card { background: var(--bg-1); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 36px 36px 32px; }
.auth-head { text-align: center; margin-bottom: 28px; }
.auth-head .logo { font-size: 32px; }
.auth-head h1 { font-size: 20px; font-weight: 700; color: var(--text) !important; margin: 6px 0 3px; }
.auth-head p { font-size: 12px; color: var(--text-3) !important; margin: 0; }

/* Welcome hero */
.hero { padding: 40px 0 32px; }
.hero h1 { font-size: 28px; font-weight: 700; color: var(--text) !important; margin: 0 0 6px; }
.hero p { font-size: 14px; color: var(--text-2) !important; margin: 0; max-width: 520px; }

/* Analysis result section */
.result-header { background: var(--bg-1); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 18px 22px; margin-bottom: 20px; }
.result-header h3 { font-size: 15px; font-weight: 600; color: var(--text) !important; margin: 0 0 10px; }

/* Library image card */
.xray-card { background: var(--bg-1); border: 1px solid var(--border); border-radius: var(--r); overflow: hidden; cursor: pointer; transition: border-color .15s; }
.xray-card:hover { border-color: var(--blue-border); }
.xray-info { padding: 8px 10px; }
.xray-info .labels { font-size: 11px; color: var(--text-2) !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.xray-info .split { font-size: 10px; color: var(--text-3) !important; margin-top: 2px; }

/* Perf chart container */
.chart-card { background: var(--bg-1); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 20px 22px; margin-bottom: 16px; }
.chart-card h4 { font-size: 13px; font-weight: 600; color: var(--text) !important; margin: 0 0 16px; }

/* Nav tab strip for landing */
.landing-tabs { display: flex; gap: 2px; justify-content: center; margin: 0 auto 28px; max-width: 340px; background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--r); padding: 3px; }
.landing-tab { flex: 1; text-align: center; padding: 7px 0; font-size: 12px; font-weight: 500; color: var(--text-2) !important; cursor: pointer; border-radius: 6px; transition: all .15s; }
.landing-tab.active { background: var(--blue); color: #fff !important; font-weight: 600; }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "token":        None,
        "username":     None,
        "page":         "landing",
        "landing_tab":  "login",
        "last_result":  None,
        "show_results": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ══════════════════════════════════════════════════════════════════════════════
#  API HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _h():
    return {"Authorization": f"Bearer {st.session_state.token}"}

def api_post(ep, data=None, files=None):
    try:
        if files:
            r = requests.post(f"{API_URL}{ep}", headers=_h(), files=files, timeout=120)
        else:
            r = requests.post(f"{API_URL}{ep}", headers={**_h(), "Content-Type": "application/json"}, json=data, timeout=30)
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server."}
    except Exception as e:
        return {"error": str(e)}

def api_get(ep):
    try:
        r = requests.get(f"{API_URL}{ep}", headers=_h(), timeout=30)
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server."}
    except Exception as e:
        return {"error": str(e)}

def b64img(s):
    return Image.open(BytesIO(base64.b64decode(s)))

# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def nav(page):
    st.session_state.page = page
    st.rerun()

def eyebrow(t):
    st.markdown(f"<span class='eyebrow'>{t}</span>", unsafe_allow_html=True)

def card(icon, title, body_html):
    st.markdown(f"<div class='card'><div class='card-title'>{icon} {title}</div>{body_html}</div>", unsafe_allow_html=True)

def stat_tile(val, lbl):
    st.markdown(f"<div class='stat-tile'><span class='stat-val'>{val}</span><span class='stat-lbl'>{lbl}</span></div>", unsafe_allow_html=True)

def pbar(label, prob, detected):
    cls   = "hot" if detected else ""
    dot   = "●" if detected else "○"
    color = "var(--red)" if prob >= .5 else "var(--amber)" if prob >= .3 else "var(--green)"
    if not detected: color = "var(--green)"
    st.markdown(f"""
    <div class='pbar-row'>
        <div class='pbar-label {cls}'>{dot} {label}</div>
        <div class='pbar-track'><div class='pbar-fill' style='width:{prob*100:.1f}%;background:{color}'></div></div>
        <div class='pbar-pct'>{prob:.1%}</div>
    </div>""", unsafe_allow_html=True)

def rbox(text):
    st.markdown(f"<div class='rbox'>{text}</div>", unsafe_allow_html=True)

def disc(text):
    st.markdown(f"<div class='disc'>⚠ {text}</div>", unsafe_allow_html=True)

def badges(detection):
    html = ""
    for path, info in detection.items():
        lbl = path.replace("_", " ").title()
        if info["detected"]:
            html += f"<span class='badge-pos'>⚠ {lbl} {info['probability']:.0%}</span>"
        else:
            html += f"<span class='badge-neg'>✓ {lbl}</span>"
    st.markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TOP NAV BAR  (shown on all authenticated pages)
# ══════════════════════════════════════════════════════════════════════════════

NAV_PAGES = [
    ("welcome",     "Welcome"),
    ("analyze",     "Analysis"),
    ("history",     "History"),
    ("library",     "Library"),
    ("performance", "Performance"),
    ("about",       "About"),
]

def render_topnav():
    st.markdown(f"""
    <div class='topnav'>
        <div class='topnav-logo'>🫁 <span>Med</span>Step</div>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit buttons rendered in a row via columns
    cols = st.columns([1.2] + [1]*len(NAV_PAGES) + [1.5, 0.8])
    with cols[0]:
        st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

    for i, (pg, lbl) in enumerate(NAV_PAGES):
        with cols[i+1]:
            active = st.session_state.page == pg
            style  = "primary" if active else "secondary"
            if st.button(lbl, key=f"nav_{pg}", type=style, use_container_width=True):
                nav(pg)

    with cols[-2]:
        st.markdown(
            f"<div style='padding-top:6px;font-size:12px;color:var(--text-2);text-align:right'>"
            f"<span style='display:inline-block;width:6px;height:6px;background:var(--green);"
            f"border-radius:50%;margin-right:5px;vertical-align:middle'></span>"
            f"{st.session_state.username}</div>",
            unsafe_allow_html=True,
        )
    with cols[-1]:
        if st.button("Logout", key="nav_logout"):
            api_post("/auth/logout")
            st.session_state.token    = None
            st.session_state.username = None
            st.session_state.page     = "landing"
            st.session_state.landing_tab = "login"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LANDING  (login / register / about tabs)
# ══════════════════════════════════════════════════════════════════════════════

def page_landing():
    tab = st.session_state.landing_tab

    # Tab strip
    st.markdown("<div class='auth-wrap'>", unsafe_allow_html=True)
    t_login = "active" if tab == "login"    else ""
    t_reg   = "active" if tab == "register" else ""
    t_about = "active" if tab == "about"    else ""
    st.markdown(f"""
    <div class='landing-tabs'>
        <div class='landing-tab {t_login}'  onclick="">Login</div>
        <div class='landing-tab {t_reg}'    onclick="">Register</div>
        <div class='landing-tab {t_about}'  onclick="">About</div>
    </div>""", unsafe_allow_html=True)

    # Real buttons (hidden via columns trick — placed just below the visual strip)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Login",    key="lt_login",    use_container_width=True,
                     type="primary" if tab=="login" else "secondary"):
            st.session_state.landing_tab = "login";  st.rerun()
    with c2:
        if st.button("Register", key="lt_register", use_container_width=True,
                     type="primary" if tab=="register" else "secondary"):
            st.session_state.landing_tab = "register"; st.rerun()
    with c3:
        if st.button("About",    key="lt_about",    use_container_width=True,
                     type="primary" if tab=="about" else "secondary"):
            st.session_state.landing_tab = "about"; st.rerun()

    # ── Login ─────────────────────────────────────────────────────────────
    if tab == "login":
        st.markdown("<div class='auth-card'><div class='auth-head'><div class='logo'>🫁</div><h1>MedStep</h1><p>AI-Assisted Chest X-Ray Interpretation</p></div>", unsafe_allow_html=True)
        username = st.text_input("Username", key="li_u", placeholder="your username")
        password = st.text_input("Password", type="password", key="li_p", placeholder="your password")
        st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)
        if st.button("Sign in", type="primary", use_container_width=True, key="li_btn"):
            if not username or not password:
                st.error("Fill in all fields.")
            else:
                res = api_post("/auth/login", {"username": username, "password": password})
                if "error" in res:          st.error(res["error"])
                elif res.get("success"):
                    st.session_state.token    = res["token"]
                    st.session_state.username = username
                    st.session_state.page     = "welcome"
                    st.rerun()
                else:                       st.error(res.get("detail", "Login failed."))
        st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;font-size:12px'>No account? <a href='#' style='color:var(--blue)'>Register</a></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Register ──────────────────────────────────────────────────────────
    elif tab == "register":
        st.markdown("<div class='auth-card'><div class='auth-head'><div class='logo'>📝</div><h1>Create Account</h1><p>Join MedStep to start analysing X-rays</p></div>", unsafe_allow_html=True)
        username = st.text_input("Username (min 3 chars)", key="r_u")
        email    = st.text_input("Email",                  key="r_e")
        password = st.text_input("Password (min 6 chars)", type="password", key="r_p")
        confirm  = st.text_input("Confirm Password",       type="password", key="r_c")
        st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)
        if st.button("Create account", type="primary", use_container_width=True, key="r_btn"):
            if not all([username, email, password, confirm]):
                st.error("Fill in all fields.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                res = api_post("/auth/register", {"username": username, "email": email, "password": password})
                if "error" in res:          st.error(res["error"])
                elif res.get("success"):
                    st.success("Account created! Please log in.")
                    st.session_state.landing_tab = "login"; st.rerun()
                else:                       st.error(res.get("detail", "Registration failed."))
        st.markdown("</div>", unsafe_allow_html=True)

    # ── About (landing) ────────────────────────────────────────────────────
    else:
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
        st.markdown("<div class='auth-head'><div class='logo'>🫁</div><h1>MedStep</h1><p>AI-Assisted Chest X-Ray Interpretation for Medical Education</p></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:13px;color:var(--text-2);line-height:1.7'>
            <p>MedStep uses a DenseNet121 model trained on CheXpert to detect 6 chest pathologies,
            generate Grad-CAM heatmaps, retrieve similar cases, and produce natural-language radiology reports.</p>
            <p style='margin-top:10px;font-size:11px;color:var(--text-3)'>Educational use only. Not for clinical diagnosis.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: WELCOME
# ══════════════════════════════════════════════════════════════════════════════

def page_welcome():
    st.markdown("<div class='page-wrap'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='hero'>
        <h1>Welcome back, {st.session_state.username} 👋</h1>
        <p>MedStep uses deep learning to help medical students interpret chest X-rays —
        detecting pathologies, visualising attention regions, retrieving similar cases,
        and generating structured radiology reports.</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: stat_tile("0.767", "Macro AUC")
    with c2: stat_tile("0.517", "Macro F1")
    with c3: stat_tile("24k",   "Training Cases")
    with c4: stat_tile("6",     "Pathologies")
    with c5: stat_tile("<0.3s", "Retrieval")

    st.markdown("<div style='height:24px'/>", unsafe_allow_html=True)

    # Feature cards
    c1, c2, c3 = st.columns(3)
    with c1:
        card("🔬", "Detection",
             "<p>DenseNet121 trained on CheXpert detects Pleural Effusion, Cardiomegaly, Pneumothorax, Lung Mass, Pneumonia and Atelectasis with per-pathology thresholds.</p>")
        card("🔍", "Similar Case Retrieval",
             "<p>FAISS vector index over 24,020 training embeddings returns the top-3 most visually similar cases in under 0.3 seconds.</p>")
    with c2:
        card("🔥", "Grad-CAM Localisation",
             "<p>Class activation maps highlight the image regions driving each prediction, overlaid on the original X-ray with a lung-masked validity check.</p>")
        card("📄", "Radiology Report",
             "<p>Groq LLaMA Vision generates structured Findings and Impression sections in natural language, downloadable as a plain-text report.</p>")
    with c3:
        card("📋", "History",
             "<p>Every analysis is stored with its findings, impression and report. Browse, review or delete past entries at any time.</p>")
        card("📚", "X-Ray Library",
             "<p>Browse and filter the CheXpert dataset images by pathology label to compare your uploaded case against real examples.</p>")

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_analyze():
    st.markdown("<div class='page-wrap'>", unsafe_allow_html=True)
    eyebrow("Analysis")
    st.markdown("<h2 style='margin:0 0 4px;font-size:20px'>Chest X-Ray Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px;margin:0 0 24px'>Upload a frontal (AP/PA) chest X-ray to receive AI-assisted interpretation.</p>", unsafe_allow_html=True)

    upload_tab, results_tab = st.tabs(["📤  Upload", "📊  Results"])

    # ── Upload tab ─────────────────────────────────────────────────────────
    with upload_tab:
        col_up, col_prev = st.columns([1, 1], gap="large")

        with col_up:
            uploaded = st.file_uploader(
                "Chest X-ray image",
                type=["jpg", "jpeg", "png"],
                help="Frontal view recommended. JPG or PNG.",
            )
            if uploaded:
                if st.button("🔍  Run Analysis", type="primary", use_container_width=True):
                    with st.spinner("Analysing — this may take 20–30 s…"):
                        uploaded.seek(0)
                        res = api_post("/analyze", files={"file": (
                            uploaded.name, uploaded.getvalue(), uploaded.type
                        )})
                    if "error" in res:
                        st.error(f"Analysis failed: {res['error']}")
                    elif res.get("success"):
                        st.session_state.last_result  = res
                        st.session_state.show_results = True
                        st.success("Done — see the Results tab.")
                    else:
                        st.error(res.get("detail", "Analysis failed."))

        with col_prev:
            if uploaded:
                st.image(Image.open(uploaded), caption="Uploaded X-Ray", use_container_width=True)
            else:
                st.markdown("""
                <div class='card' style='text-align:center;padding:52px 24px'>
                    <div style='font-size:32px;margin-bottom:10px'>📤</div>
                    <p style='color:var(--text-3);font-size:13px'>
                        Select an image on the left<br>then click Run Analysis
                    </p>
                </div>""", unsafe_allow_html=True)

    # ── Results tab (full width) ───────────────────────────────────────────
    with results_tab:
        if not st.session_state.last_result:
            st.markdown("""
            <div class='card' style='text-align:center;padding:52px 24px'>
                <div style='font-size:32px;margin-bottom:10px'>📊</div>
                <p style='color:var(--text-3);font-size:13px'>
                    No results yet.<br>Upload an X-ray and run the analysis first.
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            render_results(st.session_state.last_result)

    st.markdown("</div>", unsafe_allow_html=True)


def render_results(result):
    detection     = result.get("detection",     {})
    heatmaps      = result.get("heatmaps",      {})
    similar_cases = result.get("similar_cases", [])
    explanation   = result.get("explanation",   {})

    # ── Summary row ───────────────────────────────────────────────────────
    st.markdown("<div class='result-header'>", unsafe_allow_html=True)
    eyebrow("Summary")
    st.markdown("<h3>Detection Summary</h3>", unsafe_allow_html=True)
    badges(detection)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Four sections: full-width, stacked ────────────────────────────────
    # Section 1: Detection probabilities
    with st.expander("📊  Detection — Pathology Probabilities", expanded=True):
        c1, c2 = st.columns(2)
        items  = list(detection.items())
        half   = len(items) // 2 + len(items) % 2
        with c1:
            for path, info in items[:half]:
                pbar(path.replace("_", " ").title(), info["probability"], info["detected"])
        with c2:
            for path, info in items[half:]:
                pbar(path.replace("_", " ").title(), info["probability"], info["detected"])

    # Section 2: Heatmap
    with st.expander("🔥  Heatmap — Grad-CAM Localisation", expanded=True):
        if not heatmaps:
            st.info("No pathologies above threshold — heatmaps not generated.")
        else:
            col_sel, col_img = st.columns([1, 2], gap="large")
            with col_sel:
                eyebrow("Select pathology")
                selected = st.selectbox(
                    "Pathology", list(heatmaps.keys()),
                    format_func=lambda x: x.replace("_", " ").title(),
                    label_visibility="collapsed",
                )
                if selected:
                    hmap = heatmaps[selected]
                    q    = hmap.get("quality", "unknown")
                    if q == "low":
                        st.warning("Low activation — localisation uncertain.")
                    else:
                        st.success(f"Score: {hmap['score']:.3f}")
                    disc("Red/Yellow = high attention · Blue = low attention")
            with col_img:
                if selected and selected in heatmaps:
                    st.image(b64img(heatmaps[selected]["image"]),
                             caption=f"Grad-CAM · {selected.replace('_',' ').title()}",
                             use_container_width=True)

    # Section 3: Similar cases
    with st.expander("🔍  Similar Cases — Top 3 from Training Set", expanded=True):
        if not similar_cases:
            st.info("No similar cases found.")
        else:
            cols = st.columns(len(similar_cases))
            for i, case in enumerate(similar_cases):
                labels    = case.get("labels",     [])
                sim       = case.get("similarity", 0)
                rank      = case.get("rank",       0)
                case_path = case.get("path",       "")
                label_str = ", ".join(l.replace("_", " ").title() for l in labels) if labels else "No findings"

                with cols[i]:
                    try:
                        ir = requests.get(f"{API_URL}/image", params={"path": case_path},
                                          headers=_h(), timeout=10)
                        if ir.status_code == 200:
                            st.image(Image.open(BytesIO(ir.content)), use_container_width=True)
                    except Exception:
                        st.markdown("<p style='color:var(--text-3);font-size:11px'>Image unavailable</p>", unsafe_allow_html=True)

                    st.markdown(f"<p style='font-size:12px;font-weight:600;color:var(--text);margin:4px 0 2px'>#{rank} · {sim:.1%} similar</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:11px;color:var(--text-2);margin:0 0 4px'>{label_str}</p>", unsafe_allow_html=True)
                    st.markdown(f"<div class='sim-track'><div class='sim-fill' style='width:{sim*100:.1f}%'></div></div>", unsafe_allow_html=True)

    # Section 4: Report
    with st.expander("📄  Report — AI-Generated Radiology Report", expanded=True):
        findings   = explanation.get("findings",   "")
        impression = explanation.get("impression", "")
        if not findings and not impression:
            st.info("No report generated.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                eyebrow("Findings")
                rbox(findings)
            with c2:
                eyebrow("Impression")
                rbox(impression)
            st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
            st.download_button("📥  Download Report", data=explanation.get("full_report", ""),
                               file_name="medstep_report.txt", mime="text/plain")
        disc("AI-generated for educational purposes only. Not for clinical use.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def page_history():
    st.markdown("<div class='page-wrap'>", unsafe_allow_html=True)
    eyebrow("History")
    st.markdown("<h2 style='margin:0 0 4px;font-size:20px'>Analysis History</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px;margin:0 0 24px'>All your previous chest X-ray analyses.</p>", unsafe_allow_html=True)

    result  = api_get("/history")
    if "error" in result:
        st.error(result["error"])
        st.markdown("</div>", unsafe_allow_html=True)
        return

    history = result.get("history", [])
    if not history:
        st.info("No history yet. Run an analysis to get started.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    eyebrow(f"{len(history)} Record(s)")

    for entry in history:
        detected  = entry.get("detected",  [])
        timestamp = entry.get("timestamp", "—")
        hid       = entry.get("history_id")
        label_str = ", ".join(d.replace("_", " ").title() for d in detected) if detected else "No findings"

        with st.expander(f"🗓  {timestamp}   ·   {label_str}", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                eyebrow("Findings")
                st.markdown(f"<p style='font-size:13px;line-height:1.65'>{entry.get('findings','N/A')}</p>", unsafe_allow_html=True)
            with c2:
                eyebrow("Impression")
                st.markdown(f"<p style='font-size:13px;line-height:1.65'>{entry.get('impression','N/A')}</p>", unsafe_allow_html=True)

            st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)
            b1, _, b2 = st.columns([2, 2, 1])
            with b1:
                if entry.get("report"):
                    st.download_button("📥 Download", data=entry["report"],
                                       file_name=f"report_{hid}.txt", mime="text/plain",
                                       key=f"dl_{hid}")
            with b2:
                if st.button("🗑 Delete", key=f"del_{hid}"):
                    dr = api_post(f"/history/{hid}/delete")
                    if dr.get("success"): st.success("Deleted."); st.rerun()
                    else: st.error(dr.get("detail", "Delete failed."))

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

def page_library():
    st.markdown("<div class='page-wrap'>", unsafe_allow_html=True)
    eyebrow("Library")
    st.markdown("<h2 style='margin:0 0 4px;font-size:20px'>X-Ray Image Library</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px;margin:0 0 20px'>Browse and filter the CheXpert dataset images by pathology label.</p>", unsafe_allow_html=True)

    PATHOLOGIES = ["All", "Pleural Effusion", "Cardiomegaly", "Pneumothorax",
                   "Lung Mass", "Pneumonia", "Atelectasis", "No Finding"]

    c_filter, c_count = st.columns([2, 1])
    with c_filter:
        selected_path = st.selectbox("Filter by pathology", PATHOLOGIES, key="lib_filter")
    with c_count:
        page_num = st.number_input("Page", min_value=1, value=1, step=1, key="lib_page")

    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

    # Fetch from API
    params = {"page": page_num, "limit": 12}
    if selected_path != "All":
        params["pathology"] = selected_path.lower().replace(" ", "_")

    try:
        r = requests.get(f"{API_URL}/library", params=params, headers=_h(), timeout=15)
        data = r.json()
    except Exception as e:
        st.error(f"Could not load library: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if "error" in data:
        st.error(data["error"])
        st.markdown("</div>", unsafe_allow_html=True)
        return

    images = data.get("images", [])
    total  = data.get("total",  0)

    if not images:
        st.info("No images found for this filter.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    eyebrow(f"Showing {len(images)} of {total} images")

    # Grid: 4 per row
    for row_start in range(0, len(images), 4):
        row_imgs = images[row_start:row_start+4]
        cols = st.columns(4, gap="small")
        for col, img_info in zip(cols, row_imgs):
            with col:
                path      = img_info.get("path", "")
                labels    = img_info.get("labels", [])
                split     = img_info.get("split",  "")
                label_str = ", ".join(l.replace("_", " ").title() for l in labels) if labels else "No Finding"

                try:
                    ir = requests.get(f"{API_URL}/image", params={"path": path},
                                      headers=_h(), timeout=8)
                    if ir.status_code == 200:
                        st.image(Image.open(BytesIO(ir.content)), use_container_width=True)
                    else:
                        st.markdown("<div style='height:120px;background:var(--bg-2);border-radius:6px;display:flex;align-items:center;justify-content:center'><span style='color:var(--text-3);font-size:11px'>No image</span></div>", unsafe_allow_html=True)
                except Exception:
                    st.markdown("<div style='height:120px;background:var(--bg-2);border-radius:6px'></div>", unsafe_allow_html=True)

                st.markdown(f"<p style='font-size:11px;color:var(--text-2);margin:3px 0 1px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{label_str}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:10px;color:var(--text-3);margin:0'>{split}</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def page_performance():
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    st.markdown("<div class='page-wrap'>", unsafe_allow_html=True)
    eyebrow("Performance")
    st.markdown("<h2 style='margin:0 0 4px;font-size:20px'>Module Performance</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px;margin:0 0 24px'>Evaluation metrics across all MedStep modules.</p>", unsafe_allow_html=True)

    # ── Detection stats ────────────────────────────────────────────────────
    PATHOLOGIES = ["Pleural Effusion", "Cardiomegaly", "Pneumothorax",
                   "Lung Mass",        "Pneumonia",    "Atelectasis"]
    AUC  = [0.844, 0.821, 0.788, 0.747, 0.715, 0.675]
    F1   = [0.747, 0.582, 0.517, 0.437, 0.348, 0.472]
    STATUS = ["Pass", "Pass", "Near", "Moderate", "Limited", "Limited"]

    # Top KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: stat_tile("0.767",  "Macro AUC")
    with k2: stat_tile("0.517",  "Macro F1")
    with k3: stat_tile("0.844",  "Best AUC (Effusion)")
    with k4: stat_tile("<0.3 s", "FAISS Retrieval")
    with k5: stat_tile("24,020", "Index Vectors")

    st.markdown("<div style='height:24px'/>", unsafe_allow_html=True)

    if HAS_PLOTLY:
        # ── Row 1: AUC bar + F1 bar ──────────────────────────────────────
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<h4>AUC-ROC per Pathology</h4>", unsafe_allow_html=True)
            colors = ["#4ade80" if a >= .80 else "#fbbf24" if a >= .75 else "#f87171" for a in AUC]
            fig = go.Figure(go.Bar(
                x=AUC, y=PATHOLOGIES, orientation="h",
                marker_color=colors,
                text=[f"{a:.3f}" for a in AUC], textposition="outside",
                textfont=dict(color="#94a3b8", size=11),
            ))
            fig.add_vline(x=0.80, line_dash="dash", line_color="rgba(255,255,255,0.2)",
                          annotation_text="Target 0.80", annotation_font_color="#4b5e78",
                          annotation_font_size=10)
            fig.update_layout(
                paper_bgcolor="transparent", plot_bgcolor="transparent",
                xaxis=dict(range=[0.5, 1.0], gridcolor="rgba(255,255,255,0.05)",
                           tickfont=dict(color="#4b5e78", size=10), tickformat=".2f"),
                yaxis=dict(tickfont=dict(color="#94a3b8", size=11)),
                margin=dict(l=0, r=40, t=0, b=0), height=240, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<h4>F1 Score per Pathology</h4>", unsafe_allow_html=True)
            colors_f1 = ["#4ade80" if f >= .60 else "#fbbf24" if f >= .45 else "#f87171" for f in F1]
            fig2 = go.Figure(go.Bar(
                x=F1, y=PATHOLOGIES, orientation="h",
                marker_color=colors_f1,
                text=[f"{f:.3f}" for f in F1], textposition="outside",
                textfont=dict(color="#94a3b8", size=11),
            ))
            fig2.update_layout(
                paper_bgcolor="transparent", plot_bgcolor="transparent",
                xaxis=dict(range=[0, 1.0], gridcolor="rgba(255,255,255,0.05)",
                           tickfont=dict(color="#4b5e78", size=10), tickformat=".2f"),
                yaxis=dict(tickfont=dict(color="#94a3b8", size=11)),
                margin=dict(l=0, r=40, t=0, b=0), height=240, showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 2: Radar + scatter ───────────────────────────────────────
        c3, c4 = st.columns(2)

        with c3:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<h4>AUC vs F1 — Radar</h4>", unsafe_allow_html=True)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatterpolar(r=AUC+[AUC[0]], theta=PATHOLOGIES+[PATHOLOGIES[0]],
                fill="toself", name="AUC",
                line_color="#3b82f6", fillcolor="rgba(59,130,246,0.15)"))
            fig3.add_trace(go.Scatterpolar(r=F1+[F1[0]], theta=PATHOLOGIES+[PATHOLOGIES[0]],
                fill="toself", name="F1",
                line_color="#2dd4bf", fillcolor="rgba(45,212,191,0.12)"))
            fig3.update_layout(
                paper_bgcolor="transparent", plot_bgcolor="transparent",
                polar=dict(
                    bgcolor="transparent",
                    radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color="#4b5e78",size=9), gridcolor="rgba(255,255,255,0.08)"),
                    angularaxis=dict(tickfont=dict(color="#94a3b8",size=10), gridcolor="rgba(255,255,255,0.08)"),
                ),
                legend=dict(font=dict(color="#94a3b8",size=11), bgcolor="transparent"),
                margin=dict(l=30,r=30,t=10,b=10), height=280,
            )
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c4:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<h4>AUC vs F1 — Scatter</h4>", unsafe_allow_html=True)
            color_map = {"Pass": "#4ade80", "Near": "#fbbf24",
                         "Moderate": "#fb923c", "Limited": "#f87171"}
            fig4 = go.Figure()
            for path, auc, f1, st_val in zip(PATHOLOGIES, AUC, F1, STATUS):
                fig4.add_trace(go.Scatter(
                    x=[auc], y=[f1], mode="markers+text",
                    marker=dict(size=12, color=color_map.get(st_val,"#94a3b8")),
                    text=[path.split()[0]], textposition="top center",
                    textfont=dict(color="#94a3b8", size=10),
                    name=path, showlegend=False,
                ))
            fig4.update_layout(
                paper_bgcolor="transparent", plot_bgcolor="transparent",
                xaxis=dict(title="AUC-ROC", range=[0.6,0.9], gridcolor="rgba(255,255,255,0.05)",
                           tickfont=dict(color="#4b5e78",size=10), titlefont=dict(color="#4b5e78",size=11)),
                yaxis=dict(title="F1 Score", range=[0.2,0.85], gridcolor="rgba(255,255,255,0.05)",
                           tickfont=dict(color="#4b5e78",size=10), titlefont=dict(color="#4b5e78",size=11)),
                margin=dict(l=40,r=10,t=10,b=40), height=280,
            )
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 3: Retrieval + module summary ────────────────────────────
        c5, c6 = st.columns(2)

        with c5:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<h4>FAISS Retrieval — Index Scale</h4>", unsafe_allow_html=True)
            fig5 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=24020,
                number={"font": {"color": "#3b82f6", "size": 28, "family": "DM Mono"}},
                gauge=dict(
                    axis=dict(range=[0, 30000], tickfont=dict(color="#4b5e78", size=9)),
                    bar=dict(color="#3b82f6"),
                    bgcolor="rgba(255,255,255,0.04)",
                    bordercolor="rgba(255,255,255,0.08)",
                    steps=[
                        dict(range=[0,10000],  color="rgba(59,130,246,0.06)"),
                        dict(range=[10000,20000], color="rgba(59,130,246,0.12)"),
                        dict(range=[20000,30000], color="rgba(59,130,246,0.18)"),
                    ],
                    threshold=dict(line=dict(color="#2dd4bf", width=2), value=24020),
                ),
                title={"text": "Vectors indexed", "font": {"color": "#4b5e78", "size": 11}},
            ))
            fig5.update_layout(
                paper_bgcolor="transparent", height=240, margin=dict(l=20,r=20,t=20,b=10)
            )
            st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c6:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<h4>Module Summary</h4>", unsafe_allow_html=True)
            summary_data = {
                "Module":  ["Detection", "Localisation", "Retrieval", "Explanation", "Auth", "Database"],
                "Status":  ["✓ Trained", "✓ Built",     "✓ Indexed", "✓ Live",     "✓ Secure", "✓ Active"],
                "Key Metric": ["AUC 0.767", "Grad-CAM", "<0.3 s", "LLaMA Vision", "bcrypt+JWT", "SQLite"],
            }
            st.dataframe(pd.DataFrame(summary_data),
                         use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Fallback: plain table if plotly not installed
        st.info("Install plotly (`pip install plotly`) to see charts.")
        df = pd.DataFrame({
            "Pathology": PATHOLOGIES, "AUC-ROC": AUC, "F1 Score": F1, "Status": STATUS
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

def page_about():
    st.markdown("<div class='page-wrap'>", unsafe_allow_html=True)
    eyebrow("About")
    st.markdown("<h2 style='margin:0 0 4px;font-size:20px'>About MedStep</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px;margin:0 0 24px'>AI-Assisted Chest X-Ray Interpretation for Medical Education.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        card("🎯", "Purpose",
             "<p>MedStep supports medical students in learning chest X-ray interpretation. It provides AI-assisted detection, visual Grad-CAM explanations, similar case retrieval and natural language radiology reports.</p>")
        card("🔬", "Detectable Pathologies",
             "<ul><li>🫁 Pneumonia</li><li>❤️ Cardiomegaly</li><li>💧 Pleural Effusion</li><li>🌬️ Pneumothorax</li><li>📉 Atelectasis</li><li>🔵 Lung Mass</li></ul>")
        card("📊", "Dataset",
             "<p>Trained on CheXpert-small (Irvin et al., 2019) — 24,020 mixed AP+PA frontal chest X-rays. Uncertain labels treated as negative for cleaner signal. Loss weighted by √(neg/pos).</p>")

    with c2:
        card("⚙️", "Technology Stack",
             "<ul><li><strong>Detection</strong> — DenseNet121, denseblock3+4 fine-tuned</li><li><strong>Localisation</strong> — Grad-CAM with lung mask validity check</li><li><strong>Retrieval</strong> — FAISS, 24k vectors, &lt;0.3 s</li><li><strong>Explanation</strong> — Groq LLaMA Vision API</li><li><strong>Backend</strong> — FastAPI + SQLite</li><li><strong>Frontend</strong> — Streamlit</li></ul>")
        card("🏗️", "Architecture Decisions",
             "<ul><li><strong>Model</strong> — DenseNet121 (CheXNet baseline)</li><li><strong>Uncertain labels</strong> — treated as negative</li><li><strong>Fine-tuning</strong> — 71.9% trainable params</li><li><strong>Evaluation</strong> — per-pathology threshold + AUC</li></ul>")
        card("⚠️", "Disclaimer",
             "<p>MedStep is an educational tool only. AI-generated reports must not be used for clinical diagnosis or patient management. Always consult a qualified radiologist.</p>")

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════

PROTECTED = {"welcome", "analyze", "history", "library", "performance", "about"}

def main():
    page = st.session_state.page

    # Guard: unauthenticated users can only see landing
    if not st.session_state.token and page in PROTECTED:
        st.session_state.page = "landing"
        page = "landing"

    if page == "landing":
        page_landing()
        return

    # Authenticated shell
    render_topnav()

    if   page == "welcome":     page_welcome()
    elif page == "analyze":     page_analyze()
    elif page == "history":     page_history()
    elif page == "library":     page_library()
    elif page == "performance": page_performance()
    elif page == "about":       page_about()


if __name__ == "__main__":
    main()