import streamlit as st
import requests

st.set_page_config(
    page_title="MedStep",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE = "http://localhost:8000"

# Session state defaults
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None

st.sidebar.write(f"Token: {st.session_state.token}")

# Navigation
def main():
    if not st.session_state.token:
        from pages.landing import show as landing
        from pages.login import show as login_page

        if "show_login" not in st.session_state:
            st.session_state.show_login = False

        if st.session_state.show_login:
            login_page()
        else:
            landing()
        return

    page = st.sidebar.selectbox("Navigate", [
        "Analysis", "Case Library", "Dashboard", "History", "Encyclopedia"
    ])

    # Dark/Light mode toggle
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Toggle theme in ⋮ menu → Settings")

    st.sidebar.markdown("---")
    st.sidebar.write(f"Logged in as **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        # TODO: call POST /logout with token in headers
        requests.post(f"{API_BASE}/logout", headers={"token": st.session_state.token}) 
        # TODO: clear st.session_state.token and username
        st.session_state.token = None
        st.session_state.username = None
        # TODO: st.rerun()
        st.rerun()

    if page == "Analysis":
        from pages.analysis import show
        show()
    elif page == "Case Library":
        from pages.library import show
        show()
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