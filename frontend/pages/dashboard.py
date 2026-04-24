import streamlit as st
import requests
import plotly.graph_objects as go

API_BASE = "http://localhost:8000"
PATHOLOGIES = ["pneumonia", "cardiomegaly", "pleural_effusion",
               "pneumothorax", "atelectasis", "lung_mass"]

# Known AUC results from your experiment
AUC_SCORES = {
    "pneumonia":        0.728,
    "cardiomegaly":     0.856,
    "pleural_effusion": 0.856,
    "pneumothorax":     0.799,
    "atelectasis":      0.712,
    "lung_mass":        0.799,
}

def show():
    st.title("📊 Dashboard")

    # --- Chart 1: Model AUC per pathology ---
    st.subheader("Model Performance — AUC per Pathology")
    # TODO: create a bar chart using AUC_SCORES
    # hint: fig = go.Figure(go.Bar(x=list(...), y=list(...)))
    # add a horizontal line at y=0.70 (minimum threshold)
    # st.plotly_chart(fig, use_container_width=True)
    fig1 = go.Figure(go.Bar(
        x=list(AUC_SCORES.keys()),
        y=list(AUC_SCORES.values()),
        marker_color="steelblue"
    ))
    fig1.add_hline(y=0.70, line_dash="dash", line_color="red",
                   annotation_text="Minimum AUC 0.70")
    fig1.update_layout(yaxis_range=[0, 1], xaxis_title="Pathology",
                       yaxis_title="AUC Score")
    st.plotly_chart(fig1, use_container_width=True)

    # --- Chart 2: Dataset distribution ---
    st.subheader("Dataset Distribution")
    # TODO: create a bar chart showing train/test split
    # CheXpert: 48005 train / 11995 test
    # NIH:      12508 train / 3144 test
    # Combined: 60586 train / 15066 test
    dataset_names = ["CheXpert", "NIH", "Combined"]
    train_counts  = [48005, 12508, 60586]
    test_counts   = [11995, 3144, 15066]

    fig2 = go.Figure(data=[
        # TODO: add Bar trace for train_counts
        # TODO: add Bar trace for test_counts
        # hint: go.Bar(name="Train", x=dataset_names, y=train_counts)
        go.Bar(name="Train", x=dataset_names, y=train_counts),
        go.Bar(name="Test", x=dataset_names, y=test_counts)
    ])
    fig2.update_layout(barmode="group", xaxis_title="Dataset",
                       yaxis_title="Number of Images")
    st.plotly_chart(fig2, use_container_width=True)

    # --- Chart 3: Model comparison ---
    st.subheader("Model Comparison — Macro AUC")
    # TODO: create grouped bar chart for all 4 models × 3 datasets
    # use the results from your experiment summary
    models   = ["ResNet50", "DenseNet121", "EfficientNetB0", "MobileNetV3"]
    chexpert = [0.768, 0.766, 0.758, 0.745]
    nih      = [0.780, 0.782, 0.774, 0.764]
    combined = [0.788, 0.788, 0.779, 0.767]

    fig3 = go.Figure(data=[
        # TODO: one Bar trace per dataset
        # hint: go.Bar(name="CheXpert", x=models, y=chexpert)
        go.Bar(name="CheXpert", x=models, y=chexpert),
        go.Bar(name="NIH", x=models, y=nih),
        go.Bar(name="Combined", x=models, y=combined)
    ])
    fig3.update_layout(barmode="group", xaxis_title="Model",
                       yaxis_title="Macro AUC")
    st.plotly_chart(fig3, use_container_width=True)

    # --- Chart 4: User history trends ---
    st.subheader("Your Detection History")
    res = requests.get(
        f"{API_BASE}/history",
        headers={"token": st.session_state.token}
    )
    if res.status_code == 200 and res.json()["history"]:
        history = res.json()["history"]
        st.write(f"Total analyses: **{len(history)}**")
        # TODO: show a simple table of timestamps + report previews
        # hint: st.dataframe with list of dicts
        rows = [{"timestamp": h["timestamp"],
                 "report_preview": h["report"][:100] + "..."}
                for h in history]
        st.dataframe(rows)
    else:
        st.info("No analysis history yet.")