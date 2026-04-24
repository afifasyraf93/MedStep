import streamlit as st
import requests

API_BASE = "http://localhost:8000"
PATHOLOGIES = ["pneumonia", "cardiomegaly", "pleural_effusion",
               "pneumothorax", "atelectasis", "lung_mass"]

def show(selected=None):
    st.title("📚 Case Library")
    st.markdown("Browse real chest X-ray cases from the dataset. "
                "Filter by pathology to study specific conditions.")
    st.markdown("---")

    # Sidebar filter
    if selected is None:
        selected = []

    pathologies_str = ",".join(selected)
    res = requests.get(f"{API_BASE}/library",
                       params={"pathologies": pathologies_str})

    if res.status_code != 200:
        st.error("Failed to load cases.")
        return

    cases = res.json()

    # Summary
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(f"Showing **{len(cases)}** cases"
                    + (f" matching **{', '.join(s.replace('_',' ').title() for s in selected)}**"
                       if selected else ""))
    with col_b:
        st.markdown(f"*Total: {len(cases)} cases*")

    st.markdown("<br>", unsafe_allow_html=True)

    # Grid
    cols = st.columns(3)
    for i, case in enumerate(cases):
        with cols[i % 3]:
            # Image
            img_url = f"{API_BASE}/image?path={case['image_path']}"
            st.image(img_url, use_container_width=True)

            # Patient info
            st.markdown(f"**{case['patient_id']}**")

            # Pathology tags
            positive = [p.replace("_", " ").title()
                        for p, v in case["labels"].items() if v == 1]
            if positive:
                tags = " ".join([f"`{p}`" for p in positive])
                st.markdown(tags)
            else:
                st.caption("No findings")

            st.markdown("---")