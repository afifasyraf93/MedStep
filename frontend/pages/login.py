import streamlit as st
import requests

API_BASE = "http://localhost:8000"

def show():
    st.title("🫁 MedStep")
    st.subheader("AI-Assisted Chest X-Ray Interpretation")

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Login"

    tab_names = ["Login", "Register"]
    default_index = tab_names.index(st.session_state.active_tab)
    tab1, tab2 = st.tabs(tab_names)

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            # TODO: POST /login with params username + password
            # if response ok, store token + username in session state
            # st.rerun() on success
            # st.error() on failure
            res = requests.post(f"{API_BASE}/login", params={"username": username, "password": password})
            if res.status_code == 200:
                st.session_state.token = res.json()["token"]
                st.session_state.username = username
                st.rerun()
            else:
                st.error(res.json().get("detail", "Login failed"))

    with tab2:
        username = st.text_input("Username", key="reg_user")
        email    = st.text_input("Email", key="reg_email")
        password = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Register"):
            # TODO: POST /register with params username + email + password
            # st.success() on success
            # st.error() on failure
            res = requests.post(f"{API_BASE}/register", params={"username": username, "password": password, "email": email})
            if res.status_code == 200:
                st.success("Registered successfully! Please log in.")
            else:
                st.error(res.json().get("detail", "Register failed"))