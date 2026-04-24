import streamlit as st

def show():
    st.markdown("""
        <div style="text-align: center; padding: 60px 0 20px 0;">
            <h1 style="font-size: 3.5rem;">🫁 MedStep</h1>
            <h3 style="font-weight: normal; opacity: 0.7;">
                AI-Assisted Chest X-Ray Interpretation for Medical Students
            </h3>
        </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style="text-align:center; padding: 20px;">
                <h2>🔬</h2>
                <h4>Smart Detection</h4>
                <p style="opacity:0.7;">DenseNet121 trained on 75,000+ chest X-rays
                detecting 6 pathologies with AUC ≥ 0.70</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style="text-align:center; padding: 20px;">
                <h2>🗺️</h2>
                <h4>Visual Explanation</h4>
                <p style="opacity:0.7;">Grad-CAM heatmaps highlight exactly which
                regions influenced the AI's decision</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style="text-align:center; padding: 20px;">
                <h2>📚</h2>
                <h4>Case Library</h4>
                <p style="opacity:0.7;">Browse real cases filtered by pathology
                to build diagnostic confidence</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Stats row
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        st.metric("Training Images", "75,652")
    with col5:
        st.metric("Pathologies", "6")
    with col6:
        st.metric("Macro AUC", "0.788")
    with col7:
        st.metric("Model", "DenseNet121")

    st.markdown("---")

    # CTA button
    col_left, col_center, col_right = st.columns([2, 1, 2])
    with col_center:
        if st.button("🚀 Get Started", use_container_width=True):
            # TODO: set st.session_state.show_login = True
            st.session_state.show_login = True
            # TODO: st.rerun()
            st.rerun()
            pass