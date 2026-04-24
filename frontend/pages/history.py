import streamlit as st
import requests
from datetime import datetime

API_BASE = "http://localhost:8000"

def show():
    st.title("🕓 History")
    st.markdown("Your past chest X-ray analyses. Click to expand and view the full report.")
    st.markdown("---")

    res = requests.get(
        f"{API_BASE}/history",
        headers={"token": st.session_state.token}
    )

    if res.status_code != 200:
        st.error("Failed to load history.")
        return

    history = res.json()["history"]

    # Filter out second opinion entries
    analysis_history = [h for h in history
                        if not (h["report"] and
                                h["report"].startswith("SECOND_OPINION"))]

    if not analysis_history:
        st.info("No analyses yet. Go to Analysis to get started.")
        return

    # Summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Analyses", len(analysis_history))
    with col2:
        latest = analysis_history[0]["timestamp"][:10] if analysis_history else "—"
        st.metric("Latest Analysis", latest)

    st.markdown("<br>", unsafe_allow_html=True)

    # History entries
    for h in analysis_history:
        # Format timestamp
        try:
            dt = datetime.fromisoformat(h["timestamp"])
            label = dt.strftime("%d %b %Y, %I:%M %p")
        except:
            label = h["timestamp"]

        with st.expander(f"📋 {label}"):
            report = h["report"]

            if isinstance(report, dict):
                st.markdown("**Findings**")
                st.markdown(report.get("findings", ""))
                st.markdown("**Impression**")
                st.markdown(report.get("impression", ""))
            else:
                # Try to detect if it's a stringified dict
                if report and report.startswith("{"):
                    import ast
                    try:
                        report_dict = ast.literal_eval(report)
                        st.markdown("**Findings**")
                        st.markdown(report_dict.get("findings", ""))
                        st.markdown("**Impression**")
                        st.markdown(report_dict.get("impression", ""))
                    except:
                        st.markdown(report)
                else:
                    st.markdown(report if report else "No report available.")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Delete", key=f"del_{h['history_id']}"):
                del_res = requests.delete(
                    f"{API_BASE}/history/{h['history_id']}",
                    headers={"token": st.session_state.token}
                )
                if del_res.status_code == 200:
                    st.rerun()