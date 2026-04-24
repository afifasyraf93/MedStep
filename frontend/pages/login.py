import streamlit as st
import requests

API_BASE = "http://localhost:8000"

def show():
    # Center the login card
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col_l, col_c, col_r = st.columns([1.5, 2, 1.5])
    with col_c:
        st.markdown("""
            <div style="text-align:center; margin-bottom: 24px;">
                <h1 style="color:#2E6B8A; font-size:2.5rem; margin-bottom:0;">🫁 MedStep</h1>
                <p style="color:#6B6B6B;">AI-Assisted Chest X-Ray Interpretation</p>
            </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            username = st.text_input("Username", key="login_user",
                                     placeholder="Enter your username")
            password = st.text_input("Password", type="password",
                                     key="login_pass",
                                     placeholder="Enter your password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Login", use_container_width=True, key="login_btn"):
                if not username or not password:
                    st.warning("Please fill in all fields.")
                else:
                    res = requests.post(f"{API_BASE}/login",
                                        params={"username": username,
                                                "password": password})
                    if res.status_code == 200:
                        st.session_state.token = res.json()["token"]
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error(res.json().get("detail", "Login failed"))

        with tab2:
            username = st.text_input("Username", key="reg_user",
                                     placeholder="Choose a username")
            email    = st.text_input("Email", key="reg_email",
                                     placeholder="Enter your email")
            password = st.text_input("Password", type="password",
                                     key="reg_pass",
                                     placeholder="Choose a password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Register", use_container_width=True, key="reg_btn"):
                if not username or not email or not password:
                    st.warning("Please fill in all fields.")
                else:
                    res = requests.post(f"{API_BASE}/register",
                                        params={"username": username,
                                                "password": password,
                                                "email": email})
                    if res.status_code == 200:
                        st.success("Registered successfully! Please log in.")
                    else:
                        st.error(res.json().get("detail", "Register failed"))

        st.markdown("""
            <p style="text-align:center; color:#6B6B6B; font-size:0.8rem; margin-top:24px;">
            For educational purposes only — not a substitute for professional medical diagnosis.
            </p>
        """, unsafe_allow_html=True)