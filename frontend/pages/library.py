import streamlit as st
import requests

API_BASE = "http://localhost:8000"
PATHOLOGIES = ["pneumonia", "cardiomegaly", "pleural_effusion",
               "pneumothorax", "atelectasis", "lung_mass"]

def show():
    st.title("📚 Case Library")

    # Sidebar filter
    selected = st.sidebar.multiselect(
        "Filter by Pathology",
        options=PATHOLOGIES,
        default=[]
    )

    # Build query string and call API
    # TODO: join selected list into comma-separated string
    # hint: ",".join(selected)
    pathologies_str = ",".join(selected)

    # TODO: GET /library with pathologies param
    # hint: requests.get(f"{API_BASE}/library", params={"pathologies": pathologies_str})
    res = requests.get(f"{API_BASE}/library", params={"pathologies": pathologies_str})

    if res.status_code == 200:
        cases = res.json()
        st.write(f"Showing **{len(cases)}** cases")

        # Display in 3-column grid
        cols = st.columns(3)
        for i, case in enumerate(cases):
            with cols[i % 3]:
                img_url = f"{API_BASE}/image?path={case['image_path']}"
                st.image(img_url, use_container_width=True)
                st.markdown(f"**{case['patient_id']}**")
                # TODO: show positive pathologies only
                # hint: same pattern as analysis.py similar cases
                positive = [p for p, v in case["labels"].items() if v == 1]
                st.write(", ".join(positive) if positive else "No findings")
                st.markdown("---")
    else:
        st.error("Failed to load library.")