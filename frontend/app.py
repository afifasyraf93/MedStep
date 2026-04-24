import streamlit as st
import requests

st.set_page_config(page_title="MedStep", page_icon="🫁", layout="wide")

st.markdown("""
    <style>
        [data-testid="stSidebarNav"] { display: none; }
    </style>
""", unsafe_allow_html=True)

API_BASE = "http://localhost:8000"

if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "show_login" not in st.session_state:
    st.session_state.show_login = False

def main():
    if not st.session_state.token:
        from pages.landing import show as landing
        from pages.login import show as login_page
        if st.session_state.show_login:
            login_page()
        else:
            landing()
        return

    # Sidebar
    with st.sidebar:
        # Logo + title
        st.markdown("""
            <div style="text-align:center; padding: 16px 0 8px 0;">
                <h2 style="color:#2E6B8A; margin:0;">🫁 MedStep</h2>
                <p style="color:#6B6B6B; font-size:0.8rem; margin:0;">
                AI-Assisted CXR Interpretation</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        # Navigation
        page = st.selectbox("Navigate", [
            "Analysis", "Case Library", "Dashboard",
            "History", "Encyclopedia"
        ])

        # Page-specific sidebar content
        sidebar_filter = None
        if page == "Case Library":
            st.sidebar.markdown("---")
            st.sidebar.subheader("🔍 Filter Cases")
            sidebar_filter = st.sidebar.multiselect(
                "Pathology",
                options=["pneumonia", "cardiomegaly", "pleural_effusion",
                        "pneumothorax", "atelectasis", "lung_mass"],
                default=[],
                format_func=lambda x: x.replace("_", " ").title()
            )

        st.markdown("---")

        # User info
        st.markdown(f"""
            <div style="padding: 8px 0;">
                <p style="margin:0; color:#6B6B6B; font-size:0.8rem;">Logged in as</p>
                <p style="margin:0; font-weight:bold; color:#2E6B8A;">
                👤 {st.session_state.username}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚪 Logout", use_container_width=True):
            requests.post(f"{API_BASE}/logout",
                         headers={"token": st.session_state.token})
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.show_login = False
            st.rerun()

        st.markdown("---")
        st.caption("💡 Toggle theme in ⋮ menu → Settings")
        st.caption("v1.0 — UiTM FYP CSP650")

    # Page routing
    if page == "Analysis":
        from pages.analysis import show
        show()
    elif page == "Case Library":
        from pages.library import show
        show(sidebar_filter)
    elif page == "Dashboard":
        from pages.dashboard import show
        show()
    elif page == "History":
        from pages.history import show
        show()
    elif page == "Encyclopedia":
        from pages.encyclopedia import show
        show()

if __name__ == "__main__":
    main()