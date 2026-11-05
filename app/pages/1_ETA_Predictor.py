"""ETA Predictor page — input a delivery job, get a predicted duration."""
import os, sys
import streamlit as st
import pydeck as pdk
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.model import predict, load_model
from src.geo import haversine_km

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="ETA Predictor", page_icon="🔮", layout="wide")
st.title("🔮 ETA Predictor")
st.markdown("Fill in the delivery job details to predict the actual delivery duration.")

# ── Check model exists ─────────────────────────────────────────────────────
model_path = os.path.join(ROOT, "models", "xgb_eta.pkl")
meta_path  = os.path.join(ROOT, "models", "xgb_meta.json")

if not os.path.exists(model_path):
    st.error("Model not found. Run `python src/model.py` first.")
    st.stop()

_, le, meta = load_model(model_path, meta_path)
vehicle_classes = meta.get("vehicle_classes", ["LARGE", "MEDIUM", "OTHER", "VAN"])

# ── Input form ─────────────────────────────────────────────────────────────
with st.form("predictor_form"):
    st.subheader("Origin (warehouse / depot)")
    c1, c2 = st.columns(2)
    orig_lat = c1.number_input("Latitude",  value=22.3699, format="%.5f", step=0.001)
    orig_lon = c2.number_input("Longitude", value=114.1356, format="%.5f", step=0.001)

    st.subheader("Destination")
    c3, c4 = st.columns(2)
    dest_lat = c3.number_input("Latitude",  value=22.2796, format="%.5f", step=0.001, key="dlat")
    dest_lon = c4.number_input("Longitude", value=114.1842, format="%.5f", step=0.001, key="dlon")

    st.subheader("Job details")
    c5, c6, c7 = st.columns(3)
    vehicle_cat      = c5.selectbox("Vehicle category", vehicle_classes, index=vehicle_classes.index("MEDIUM") if "MEDIUM" in vehicle_classes else 0)
    planned_dur_min  = c6.number_input("Planned duration (min)", min_value=5, max_value=600, value=180, step=5)
    start_hour       = c7.slider("Planned start hour", 0, 23, 8)

    c8, c9 = st.columns(2)
    day_of_week = c8.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    month       = c9.slider("Month", 1, 12, 3)

    submitted = st.form_submit_button("Predict ETA", use_container_width=True, type="primary")

# ── Run prediction ─────────────────────────────────────────────────────────
if submitted:
    dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    dow_int = dow_map[day_of_week]

    result = predict(
        orig_lat=orig_lat, orig_lon=orig_lon,
        dest_lat=dest_lat, dest_lon=dest_lon,
        vehicle_cat=vehicle_cat,
        planned_duration_min=float(planned_dur_min),
        hour=start_hour, day_of_week=dow_int, month=month,
        model_path=model_path, meta_path=meta_path,
    )

    ratio     = result["predicted_delay_ratio"]
    act_min   = result["predicted_actual_min"]
    low_min   = result["confidence_low_min"]
    high_min  = result["confidence_high_min"]
    hav_km    = haversine_km(orig_lat, orig_lon, dest_lat, dest_lon)

    st.divider()
    st.subheader("Prediction Result")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Straight-line distance", f"{hav_km:.1f} km")
    col2.metric("Delay ratio", f"{ratio:.2f}×",
                delta=f"{ratio - 1:.2f}× vs plan",
                delta_color="inverse")
    col3.metric("Predicted actual duration",
                f"{int(act_min // 60)}h {int(act_min % 60)}m")
    col4.metric("80% confidence range",
                f"{int(low_min)}–{int(high_min)} min")

    # ── Traffic context ────────────────────────────────────────────────────
    st.divider()
    peak_label = ""
    if 7 <= start_hour <= 9:
        peak_label = "⚠️ **Morning peak hour** — expect higher delays"
    elif 17 <= start_hour <= 20:
        peak_label = "⚠️ **Evening peak hour** — expect higher delays"
    else:
        peak_label = "✅ Off-peak hours — lower traffic impact expected"

    weekend_label = "🏖️ Weekend — lighter traffic" if dow_int >= 5 else "🗓️ Weekday"
    st.info(f"{peak_label}   ·   {weekend_label}")

    # ── Map ────────────────────────────────────────────────────────────────
    st.subheader("Route overview")
    arc_data = [{
        "orig_lat": orig_lat, "orig_lon": orig_lon,
        "dest_lat": dest_lat, "dest_lon": dest_lon,
    }]
    color = [220, 50, 50] if ratio > 2 else [50, 180, 100]

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arc_data,
        get_source_position=["orig_lon", "orig_lat"],
        get_target_position=["dest_lon", "dest_lat"],
        get_source_color=color,
        get_target_color=[50, 100, 220],
        get_width=4,
        pickable=True,
    )
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[
            {"lat": orig_lat, "lon": orig_lon, "label": "Origin"},
            {"lat": dest_lat, "lon": dest_lon, "label": "Destination"},
        ],
        get_position=["lon", "lat"],
        get_fill_color=[[255, 140, 0], [30, 144, 255]],
        get_radius=300,
        pickable=True,
    )
    mid_lat = (orig_lat + dest_lat) / 2
    mid_lon = (orig_lon + dest_lon) / 2
    view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=11, pitch=30)

    st.pydeck_chart(
        pdk.Deck(
            layers=[arc_layer, scatter_layer],
            initial_view_state=view,
            map_style="mapbox://styles/mapbox/dark-v10",
            tooltip={"text": "{label}"},
        )
    )
