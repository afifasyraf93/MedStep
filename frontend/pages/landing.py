import streamlit as st

def show():
    # Hero section
    st.markdown("""
        <div style="text-align: center; padding: 60px 20px 30px 20px;">
            <h1 style="font-size: 3.5rem; color: #2E6B8A; margin-bottom: 0;">🫁 MedStep</h1>
            <p style="font-size: 1.3rem; color: #6B6B6B; margin-top: 8px;">
                AI-Assisted Chest X-Ray Interpretation for Medical Students
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Feature highlights — simple icon row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style="text-align:center; padding: 20px 10px;">
                <div style="font-size:2.5rem;">🔬</div>
                <h4 style="color:#2E6B8A;">Smart Detection</h4>
                <p style="color:#6B6B6B; font-size:0.9rem;">DenseNet121 trained on 75,000+ 
                chest X-rays detecting 6 pathologies with AUC ≥ 0.70</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style="text-align:center; padding: 20px 10px;">
                <div style="font-size:2.5rem;">🗺️</div>
                <h4 style="color:#2E6B8A;">Visual Explanation</h4>
                <p style="color:#6B6B6B; font-size:0.9rem;">Grad-CAM heatmaps highlight 
                exactly which regions influenced the AI decision</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style="text-align:center; padding: 20px 10px;">
                <div style="font-size:2.5rem;">📚</div>
                <h4 style="color:#2E6B8A;">Case Library</h4>
                <p style="color:#6B6B6B; font-size:0.9rem;">Browse real cases filtered 
                by pathology to build diagnostic confidence</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
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
    st.markdown("<br>", unsafe_allow_html=True)

    # CTA
    col_l, col_c, col_r = st.columns([2, 1, 2])
    with col_c:
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state.show_login = True
            st.rerun()

    # Disclaimer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <p style="text-align:center; color:#6B6B6B; font-size:0.8rem;">
        For educational purposes only — not a substitute for professional medical diagnosis.
        </p>
    """, unsafe_allow_html=True)