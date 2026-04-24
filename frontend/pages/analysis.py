import streamlit as st
import requests
import base64
from PIL import Image
import io

API_BASE = "http://localhost:8000"
PATHOLOGIES = ["pneumonia", "cardiomegaly", "pleural_effusion",
               "pneumothorax", "atelectasis", "lung_mass"]

def show():
    st.title("🔬 Analysis")

    uploaded = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])
    threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)

    if uploaded and st.button("Analyze"):
        with st.spinner("Analyzing..."):
            # TODO: POST /analyze with file + token header
            # hint: files={"file": (uploaded.name, uploaded.getvalue(), "image/jpeg")}
            #       headers={"token": st.session_state.token}
            res = requests.post(
                f"{API_BASE}/analyze",
                files={"file": (uploaded.name, uploaded.getvalue(), "image/jpeg")},
                headers={"token": st.session_state.token}
            ) 

        if res.status_code == 200:
            data = res.json()

            # Layout — two columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                # TODO: display uploaded image with st.image
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(uploaded.getvalue()))
                st.image(img, use_container_width=True)

            with col2:
                st.subheader("Detections")
                # TODO: loop through data["detections"]
                # show pathology name, probability, progress bar
                # only highlight if prob >= threshold
                for pathology, prob in data["detections"].items():
                    severity_label, severity_icon = get_severity(prob)
                    detected_marker = "✅" if prob >= threshold else "⬜"
                    st.write(f"{detected_marker} **{pathology}** — {severity_icon} {severity_label} ({round(prob * 100, 1)}%)")
                    st.progress(prob)

            st.subheader("Heatmap Viewer")
            # TODO: selectbox of detected pathologies (prob >= threshold)
            # show selected heatmap from data["heatmaps"]
            # heatmaps are base64 strings — decode with base64.b64decode()
            # display with st.image()
            detected = [p for p, prob in data["detections"].items() if prob >= threshold]

            if detected:
                # TODO: st.selectbox to pick a pathology from detected list
                selected = st.selectbox("Select pathology", detected)

                # TODO: get the base64 heatmap string from data["heatmaps"]
                # data["heatmaps"] is a dict {pathology: base64_string}
                heatmap_b64 = data["heatmaps"].get(selected)

                if heatmap_b64:
                    # TODO: decode base64 string to bytes
                    # hint: base64.b64decode(heatmap_b64)
                    heatmap_bytes = base64.b64decode(heatmap_b64)
                    # TODO: convert bytes to image with Image.open(io.BytesIO(...))
                    heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                    # TODO: display with st.image
                    st.image(heatmap_img, caption=f"Grad-CAM: {selected}")
            else:
                st.info("No pathologies detected above threshold.")

            st.subheader("Similar Cases")
            # TODO: show data["similar_cases"] as 3 columns
            # each with image_id, pathology labels
            cols = st.columns(3)
            for i, case in enumerate(data["similar_cases"]):
                with cols[i]:
                    # TODO: st.write image_id
                    # TODO: show labels — only pathologies where value == 1
                    img_url = f"{API_BASE}/image?path={case['path']}"
                    st.image(img_url, use_container_width=True)
                    positive = [p for p, v in case["labels"].items() if v == 1]
                    st.write(", ".join(positive) if positive else "No findings")

            st.subheader("AI Report")
            # TODO: st.markdown(data["report"])
            st.markdown(data["report"])
        else:
            st.error("Analysis failed.")

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