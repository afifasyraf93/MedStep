import streamlit as st
import requests
import base64
from PIL import Image
import io
from frontend.utils.pdf_export import generate_pdf_report

API_BASE = "http://localhost:8000"
PATHOLOGIES = ["pneumonia", "cardiomegaly", "pleural_effusion",
               "pneumothorax", "atelectasis", "lung_mass"]

def show():
    st.title("🔬 Analysis")

    # Mode toggle
    mode = st.radio(
        "Select Mode",
        ["🤖 AI Analysis", "🩺 Second Opinion"],
        horizontal=True
    )

    if mode == "🤖 AI Analysis":
        show_ai_analysis()
    else:
        show_second_opinion()


def show_ai_analysis():
    # Upload section
    st.markdown("""
        <div style="background:#FFFFFF; border-radius:12px; padding:24px; 
        box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-bottom:16px;">
        <h4 style="color:#2E6B8A; margin-top:0;">📤 Upload Chest X-Ray</h4>
        </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
    threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("<br>", unsafe_allow_html=True)

    if uploaded and st.button("🔍 Analyze", use_container_width=True):
        with st.spinner("Analyzing your X-ray..."):
            res = requests.post(
                f"{API_BASE}/analyze",
                files={"file": (uploaded.name, uploaded.getvalue(), "image/jpeg")},
                headers={"token": st.session_state.token}
            )

        if res.status_code == 200:
            data = res.json()

            # Section 1 — Image + Detections
            st.markdown("---")
            st.markdown("### 🖼️ Results")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Image**")
                img = Image.open(io.BytesIO(uploaded.getvalue()))
                st.image(img, use_container_width=True)

            with col2:
                st.markdown("**Detection Results**")
                for pathology, prob in data["detections"].items():
                    severity_label, severity_icon = get_severity(prob)
                    detected_marker = "✅" if prob >= threshold else "⬜"
                    st.write(f"{detected_marker} **{pathology.replace('_',' ').title()}** "
                             f"— {severity_icon} {severity_label} ({round(prob * 100, 1)}%)")
                    st.progress(prob)

            # Section 2 — Heatmap
            st.markdown("---")
            st.markdown("### 🔥 Heatmap Viewer")
            detected = [p for p, prob in data["detections"].items() 
                       if prob >= threshold]

            if detected:
                col_sel, col_conf = st.columns([2, 1])
                with col_sel:
                    selected = st.selectbox("Select pathology", detected,
                        format_func=lambda x: x.replace("_", " ").title())
                with col_conf:
                    st.metric("Confidence",
                              f"{round(data['detections'][selected]*100,1)}%")

                heatmap_b64 = data["heatmaps"].get(selected)
                if heatmap_b64:
                    col_orig, col_heat = st.columns(2)
                    with col_orig:
                        st.markdown("**Original**")
                        st.image(img, use_container_width=True)
                    with col_heat:
                        st.markdown(f"**Grad-CAM: {selected.replace('_',' ').title()}**")
                        heatmap_img = Image.open(
                            io.BytesIO(base64.b64decode(heatmap_b64)))
                        st.image(heatmap_img, use_container_width=True)
                    st.caption("🔴 Red = high activation  |  🔵 Blue = low activation")
            else:
                st.info("No pathologies detected above threshold. "
                       "Try lowering the threshold.")

            # Section 3 — Similar Cases
            st.markdown("---")
            st.markdown("### 🔍 Similar Cases")
            cols = st.columns(3)
            for i, case in enumerate(data["similar_cases"]):
                with cols[i]:
                    st.image(f"{API_BASE}/image?path={case['path']}",
                             use_container_width=True)
                    positive = [p for p, v in case["labels"].items() if v == 1]
                    st.caption(", ".join(positive) if positive else "No findings")

            # Section 4 — AI Report
            st.markdown("---")
            st.markdown("### 📋 AI Report")
            
            report = data["report"]
            if isinstance(report, dict):
                st.markdown("**Findings**")
                st.markdown(report.get("findings", ""))
                st.markdown("**Impression**")
                st.markdown(report.get("impression", ""))
            else:
                st.markdown(str(report))

            # PDF Download
            st.markdown("---")
            from utils.pdf_export import generate_pdf_report
            pdf_bytes = generate_pdf_report(
                image_bytes=uploaded.getvalue(),
                detections=data["detections"],
                report=data["report"] if isinstance(data["report"], dict) 
                       else {"findings": str(data["report"]), "impression": ""},
                username=st.session_state.username,
                threshold=threshold
            )
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name="medstep_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.error("Analysis failed. Please try again.")

def get_severity(prob: float) -> tuple:
    """Return (label, color) based on probability."""
    if prob >= 0.8:
        # TODO: return "High" with red color — ("High ⚠️", "🔴")
        return ("High ⚠️", "🔴")
    elif prob >= 0.5:
        # TODO: return "Moderate" with orange — ("Moderate ⚠️", "🟡")
        return ("Moderate ⚠️", "🟡")
    else:
        # TODO: return "Low" with green — ("Low", "🟢")
        return ("Low", "🟢")

def show_second_opinion():
    st.markdown("### 🩺 Second Opinion Mode")
    st.info("Make your diagnosis first — then let the AI verify it.")

    uploaded = st.file_uploader("Upload Chest X-Ray",
                                type=["jpg", "jpeg", "png"],
                                key="second_opinion_upload")

    if uploaded:
        st.image(Image.open(io.BytesIO(uploaded.getvalue())),
                 use_container_width=True, width=400)

        st.markdown("#### Your Diagnosis")
        st.caption("Select all pathologies you think are present:")

        # TODO: create a checkbox for each pathology
        # store results in a dict {pathology: bool}
        # hint: st.checkbox(pathology.replace("_", " ").title(), key=f"cb_{pathology}")
        student_diagnosis = {}
        cols = st.columns(2)
        for i, pathology in enumerate(PATHOLOGIES):
            with cols[i % 2]:
                # TODO: student_diagnosis[pathology] = st.checkbox(...)
                student_diagnosis[pathology] = st.checkbox(pathology.replace("_", " ").title(), key=f"cb_{pathology}")

        if st.button("🔍 Compare with AI", use_container_width=True):
            with st.spinner("Running AI analysis..."):
                # TODO: POST /analyze same as show_ai_analysis
                res = requests.post(
                    f"{API_BASE}/analyze",
                    files={"file": (uploaded.name, uploaded.getvalue(), "image/jpeg")},
                    headers={"token": st.session_state.token}
                )

            if res.status_code == 200:
                data = res.json()
                st.markdown("---")
                st.markdown("#### 📊 Comparison Results")

                # Comparison table
                col1, col2, col3 = st.columns(3)
                col1.markdown("**Pathology**")
                col2.markdown("**Your Diagnosis**")
                col3.markdown("**AI Prediction**")

                # TODO: loop through pathologies and show comparison
                # student said yes/no, AI probability + severity
                # highlight matches in green, mismatches in red
                for pathology in PATHOLOGIES:
                    c1, c2, c3 = st.columns(3)
                    student_said = student_diagnosis.get(pathology, False)
                    ai_prob = data["detections"].get(pathology, 0)
                    ai_said = ai_prob >= 0.5
                    
                    # TODO: determine match — student_said == ai_said
                    match = student_said == ai_said
                    icon = "✅" if match else "❌"

                    c1.write(f"{icon} {pathology.replace('_', ' ').title()}")
                    c2.write("Yes" if student_said else "No")
                    # TODO: c3 show AI probability + severity label
                    severity_label, severity_icon = get_severity(ai_prob)
                    c3.write(f"{severity_icon} {round(ai_prob * 100, 1)}% — {severity_label}")

                # Summary score
                # TODO: count matches and show score
                matches = sum(
                    1 for p in PATHOLOGIES
                    if student_diagnosis.get(p, False) == (data["detections"].get(p, 0) >= 0.5)
                )
                st.markdown("---")
                st.metric("Your Score", f"{matches}/{len(PATHOLOGIES)}")

                # Show AI report
                st.markdown("#### 📋 AI Report")
                st.markdown(str(data["report"]))

                # Save second opinion results
                for pathology in PATHOLOGIES:
                    student_said = student_diagnosis.get(pathology, False)
                    ai_said = data["detections"].get(pathology, 0) >= 0.5
                    match = student_said == ai_said
                    requests.post(
                        f"{API_BASE}/second-opinion/save",
                        headers={"token": st.session_state.token},
                        params={
                            "pathology": pathology,
                            "student_correct": match
                        }
                    )