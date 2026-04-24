import streamlit as st
import requests
from streamlit_echarts import st_echarts

API_BASE = "http://localhost:8000"

AUC_SCORES = {
    "Pneumonia":        0.7279,
    "Cardiomegaly":     0.8436,
    "Pleural Effusion": 0.8544,
    "Pneumothorax":     0.7930,
    "Atelectasis":      0.7120,
    "Lung Mass":        0.7971,
}

models_auc = {
        "CheXpert":  [0.768, 0.766, 0.758, 0.745],   # from your experiment summary
        "NIH":       [0.780, 0.782, 0.774, 0.764],
        "Combined":  [
            round((0.7279+0.8436+0.8544+0.7930+0.7120+0.7971)/6, 4),  # DenseNet121
            round((0.7099+0.8559+0.8556+0.7992+0.7086+0.7994)/6, 4),  # ResNet50
            round((0.7167+0.8388+0.8405+0.7818+0.7066+0.7916)/6, 4),  # EfficientNetB0
            round((0.6976+0.8243+0.8283+0.7713+0.7028+0.7781)/6, 4),  # MobileNetV3
        ]
    }

def show():
    st.title("📊 Dashboard")

    # --- Progress Section ---
    res = requests.get(f"{API_BASE}/progress",
                       headers={"token": st.session_state.token})
    if res.status_code == 200:
        progress = res.json()
        st.subheader(f"👋 Welcome back, {progress['username']}!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyses", progress["analyses_count"])
        with col2:
            st.metric("Current Streak", f"{progress['streak_days']} 🔥")
        with col3:
            last = progress["last_active"]
            st.metric("Last Active", last[:10] if last else "Never")
        st.markdown("---")

    # --- Chart 1: AUC per Pathology ---
    st.subheader("🎯 Model Performance — AUC per Pathology")
    auc_option = {
        "tooltip": {"trigger": "axis"},
        "xAxis": {
            "type": "category",
            "data": list(AUC_SCORES.keys()),
            "axisLabel": {"rotate": 15}
        },
        "yAxis": {
            "type": "value",
            "min": 0, "max": 1,
            "name": "AUC Score"
        },
        "series": [{
            "type": "bar",
            "data": list(AUC_SCORES.values()),
            "itemStyle": {"color": "#2E6B8A"},
            "markLine": {
                "data": [{"yAxis": 0.70}],
                "lineStyle": {"color": "#E07B4F", "type": "dashed"},
                "label": {"formatter": "Min AUC 0.70"}
            }
        }]
    }
    st_echarts(auc_option, height="350px")

    # --- Chart 2: Dataset Distribution ---
    st.subheader("📦 Dataset Distribution")
    dataset_option = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["Train", "Test"]},
        "xAxis": {
            "type": "category",
            "data": ["CheXpert", "NIH", "Combined"]
        },
        "yAxis": {"type": "value", "name": "Images"},
        "series": [
            {
                "name": "Train",
                "type": "bar",
                "data": [48005, 12508, 60586],
                "itemStyle": {"color": "#4CAF7D"}
            },
            {
                "name": "Test",
                "type": "bar",
                "data": [11995, 3144, 15066],
                "itemStyle": {"color": "#2E6B8A"}
            }
        ]
    }
    st_echarts(dataset_option, height="350px")

    # --- Chart 3: Model Comparison ---
    st.subheader("🏆 Model Comparison — Macro AUC")
    model_option = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["CheXpert", "NIH", "Combined"]},
        "xAxis": {
            "type": "category",
            "data": ["ResNet50", "DenseNet121", "EfficientNetB0", "MobileNetV3"]
        },
        "yAxis": {
            "type": "value",
            "min": 0.7, "max": 0.80,
            "name": "Macro AUC"
        },
        "series": [
            {
                "name": "CheXpert",
                "type": "bar",
                "data": [0.766, 0.768, 0.758, 0.745],
                "itemStyle": {"color": "#E07B4F"}
            },
            {
                "name": "NIH",
                "type": "bar",
                "data": [0.782, 0.780, 0.774, 0.764],
                "itemStyle": {"color": "#4CAF7D"}
            },
            {
                "name": "Combined",
                "type": "bar",
                "data": models_auc["Combined"],
                "itemStyle": {"color": "#2E6B8A"}
            }
        ]
    }
    st_echarts(model_option, height="350px")

    # --- Chart 4: AUC Radar ---
    st.subheader("🕸️ Model Performance — AUC Radar")
    
    radar_data = {
        "DenseNet121":    [0.7279, 0.8436, 0.8544, 0.7930, 0.7120, 0.7971],
        "ResNet50":       [0.7099, 0.8559, 0.8556, 0.7992, 0.7086, 0.7994],
        "EfficientNetB0": [0.7167, 0.8388, 0.8405, 0.7818, 0.7066, 0.7916],
        "MobileNetV3":    [0.6976, 0.8243, 0.8283, 0.7713, 0.7028, 0.7781],
    }
    
    colors = {
        "DenseNet121":    "#2E6B8A",
        "ResNet50":       "#4CAF7D",
        "EfficientNetB0": "#E07B4F",
        "MobileNetV3":    "#9B59B6",
    }

    radar_option = {
        "tooltip": {},
        "legend": {"data": list(radar_data.keys())},
        "radar": {
            "indicator": [
                {"name": name, "max": 1}
                for name in AUC_SCORES.keys()
            ]
        },
        "series": [{
            "name": "AUC Score",
            "type": "radar",
            "data": [
                {
                    "value": values,
                    "name": model,
                    "areaStyle": {"opacity": 0.1},
                    "lineStyle": {"color": colors[model]},
                    "itemStyle": {"color": colors[model]}
                }
                for model, values in radar_data.items()
            ]
        }]
    }
    st_echarts(radar_option, height="450px")

    # --- Chart 5: Pathology Distribution ---
    st.subheader("🥧 Pathology Distribution in Case Library")
    stats_res = requests.get(f"{API_BASE}/library/stats")

    if stats_res.status_code == 200:
        stats = stats_res.json()
        pie_option = {
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} cases ({d}%)"},
            "legend": {"orient": "vertical", "left": "left"},
            "series": [{
                "type": "pie",
                "radius": ["40%", "70%"],
                "data": [
                    {"value": count, "name": pathology.replace("_", " ").title()}
                    for pathology, count in stats.items()
                ],
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0,0,0,0.5)"
                    }
                }
            }]
        }
        st_echarts(pie_option, height="400px")

    # --- Chart 6: Second Opinion Accuracy ---
    st.subheader("🎓 Your Diagnostic Accuracy — Second Opinion Mode")
    so_res = requests.get(f"{API_BASE}/second-opinion/stats",
                          headers={"token": st.session_state.token})

    if so_res.status_code == 200:
        accuracy = so_res.json()
        if any(v > 0 for v in accuracy.values()):
            so_option = {
                "tooltip": {"trigger": "axis",
                            "formatter": "{b}: {c}%"},
                "xAxis": {
                    "type": "category",
                    "data": [p.replace("_", " ").title()
                             for p in accuracy.keys()],
                    "axisLabel": {"rotate": 15}
                },
                "yAxis": {
                    "type": "value",
                    "min": 0, "max": 100,
                    "name": "Accuracy %"
                },
                "series": [{
                    "type": "bar",
                    "data": list(accuracy.values()),
                    "itemStyle": {
                        "color": "#4CAF7D"
                    },
                    "markLine": {
                        "data": [{"yAxis": 50}],
                        "lineStyle": {"color": "#E07B4F",
                                      "type": "dashed"},
                        "label": {"formatter": "50% baseline"}
                    }
                }]
            }
            st_echarts(so_option, height="350px")
        else:
            st.info("Use Second Opinion mode to track your diagnostic accuracy.")

    # --- Chart 7: Analysis Timeline ---
    st.subheader("📈 Your Analysis Timeline")
    hist_res = requests.get(f"{API_BASE}/history",
                            headers={"token": st.session_state.token})
    
    if hist_res.status_code == 200 and hist_res.json()["history"]:
        history = hist_res.json()["history"]
        st.write(f"Total analyses: **{len(history)}**")

        # Count analyses per day
        from collections import Counter
        date_counts = Counter(h["timestamp"][:10] for h in history)
        sorted_dates = sorted(date_counts.keys())

        if len(sorted_dates) > 1:
            # Line chart
            timeline_option = {
                "tooltip": {"trigger": "axis"},
                "xAxis": {
                    "type": "category",
                    "data": sorted_dates,
                    "axisLabel": {"rotate": 15}
                },
                "yAxis": {
                    "type": "value",
                    "name": "Analyses",
                    "minInterval": 1
                },
                "series": [{
                    "type": "line",
                    "data": [date_counts[d] for d in sorted_dates],
                    "smooth": True,
                    "itemStyle": {"color": "#2E6B8A"},
                    "areaStyle": {"color": "rgba(46,107,138,0.15)"}
                }]
            }
            st_echarts(timeline_option, height="300px")
        else:
            st.info("Run more analyses on different days to see your timeline.")

        # Keep the table below the chart
        st.markdown("**Recent Analyses**")
        rows = [{"timestamp": h["timestamp"],
                 "report_preview": h["report"][:100] + "..."
                 if h["report"] else ""}
                for h in history]
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No analysis history yet.")