import streamlit as st
import requests

API_BASE = "http://localhost:8000"
PATHOLOGIES = ["pneumonia", "cardiomegaly", "pleural_effusion",
               "pneumothorax", "atelectasis", "lung_mass"]

PATHOLOGY_DESCRIPTIONS = {
    "pneumonia":        "Infection causing inflammation of air sacs in lungs",
    "cardiomegaly":     "Enlargement of the heart visible on chest X-ray",
    "pleural_effusion": "Abnormal fluid collection between lung and chest wall",
    "pneumothorax":     "Collapsed lung due to air in the pleural space",
    "atelectasis":      "Partial or complete collapse of lung tissue",
    "lung_mass":        "Abnormal tissue growth or lesion in the lung",
}

def show():
    st.title("📖 Pathology Encyclopedia")
    st.write("Select a pathology to learn about its clinical significance, "
             "X-ray findings, and diagnostic features.")

    selected = st.selectbox("Choose a pathology", PATHOLOGIES,
                            format_func=lambda x: x.replace("_", " ").title())

    st.markdown(f"**{selected.replace('_', ' ').title()}** — "
                f"{PATHOLOGY_DESCRIPTIONS[selected]}")
    st.markdown("---")

    if st.button("📖 Generate Educational Content", use_container_width=True):
        with st.spinner("Generating content..."):
            # TODO: call GET /encyclopedia/{selected} endpoint
            # we need to add this endpoint to api.py first
            res = requests.get(
                f"{API_BASE}/encyclopedia/{selected}",
                headers={"token": st.session_state.token}
            )

        if res.status_code == 200:
            data = res.json()
            # TODO: display data["content"] with st.markdown
            st.markdown(data["content"])
        else:
            st.error("Failed to generate content.")