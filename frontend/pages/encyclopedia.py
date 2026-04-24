import streamlit as st
import requests

API_BASE = "http://localhost:8000"
PATHOLOGIES = ["pneumonia", "cardiomegaly", "pleural_effusion",
               "pneumothorax", "atelectasis", "lung_mass"]

PATHOLOGY_INFO = {
    "pneumonia": {
        "icon": "🫁",
        "description": "Infection causing inflammation of air sacs in lungs",
        "prevalence": "Common",
        "severity": "Moderate–High"
    },
    "cardiomegaly": {
        "icon": "❤️",
        "description": "Enlargement of the heart visible on chest X-ray",
        "prevalence": "Common",
        "severity": "Moderate–High"
    },
    "pleural_effusion": {
        "icon": "💧",
        "description": "Abnormal fluid collection between lung and chest wall",
        "prevalence": "Common",
        "severity": "Moderate–High"
    },
    "pneumothorax": {
        "icon": "💨",
        "description": "Collapsed lung due to air in the pleural space",
        "prevalence": "Less common",
        "severity": "High"
    },
    "atelectasis": {
        "icon": "🫧",
        "description": "Partial or complete collapse of lung tissue",
        "prevalence": "Common",
        "severity": "Moderate"
    },
    "lung_mass": {
        "icon": "🔴",
        "description": "Abnormal tissue growth or lesion in the lung",
        "prevalence": "Less common",
        "severity": "High"
    },
}

def show():
    st.title("📖 Pathology Encyclopedia")
    st.markdown("Select a pathology to learn about its clinical significance, "
                "X-ray findings, and diagnostic features.")
    st.markdown("---")

    # Pathology selector as cards
    selected = st.selectbox(
        "Choose a pathology",
        PATHOLOGIES,
        format_func=lambda x: f"{PATHOLOGY_INFO[x]['icon']} {x.replace('_', ' ').title()}"
    )

    info = PATHOLOGY_INFO[selected]

    # Info row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Condition", selected.replace("_", " ").title())
    with col2:
        st.metric("Prevalence", info["prevalence"])
    with col3:
        st.metric("Severity", info["severity"])

    st.markdown(f"*{info['description']}*")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("📖 Generate Educational Content",
                 use_container_width=True):
        with st.spinner("Generating content from AI..."):
            res = requests.get(
                f"{API_BASE}/encyclopedia/{selected}",
                headers={"token": st.session_state.token}
            )

        if res.status_code == 200:
            content = res.json()["content"]

            # Display in styled container
            st.markdown("---")
            st.markdown("### 📋 Educational Content")
            st.markdown(content)

            # Related cases from library
            st.markdown("---")
            st.markdown("### 🔍 Related Cases in Library")
            cases_res = requests.get(
                f"{API_BASE}/library",
                params={"pathologies": selected}
            )
            if cases_res.status_code == 200:
                cases = cases_res.json()[:3]  # show top 3
                if cases:
                    cols = st.columns(3)
                    for i, case in enumerate(cases):
                        with cols[i]:
                            img_url = f"{API_BASE}/image?path={case['image_path']}"
                            st.image(img_url, use_container_width=True)
                            st.caption(case["patient_id"])
                else:
                    st.info("No related cases found.")
        else:
            st.error("Failed to generate content.")