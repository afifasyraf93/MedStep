import streamlit as st
import requests

API_BASE = "http://localhost:8000"

def show():
    st.title("🕓 History")

    res = requests.get(
        f"{API_BASE}/history",
        headers={"token": st.session_state.token}
    )

    if res.status_code != 200:
        st.error("Failed to load history.")
        return

    history = res.json()["history"]

    if not history:
        st.info("No analyses yet.")
        return

    st.write(f"Total analyses: **{len(history)}**")

    for h in history:
        with st.expander(f"🗂 {h['timestamp']}"):
            # TODO: st.markdown to show the report
            st.markdown(h["report"])
            # TODO: delete button — DELETE /history/{history_id}
            # on click: call API, st.rerun() on success
            if st.button("🗑 Delete", key=f"del_{h['history_id']}"):
                del_res = requests.delete(
                    f"{API_BASE}/history/{h['history_id']}",
                    headers={"token": st.session_state.token}
                )
                if del_res.status_code == 200:
                    st.rerun()