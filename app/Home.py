"""
HK Logistics ETA Prediction System — Home page.
Run with:  streamlit run app/Home.py
"""
import os, sys, json
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="HK ETA Prediction",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model metrics ─────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meta_path = os.path.join(ROOT, "models", "xgb_meta.json")

metrics = {}
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    metrics = meta.get("metrics", {})

# ── Layout ─────────────────────────────────────────────────────────────────
st.title("🚚 HK Logistics ETA Prediction")
st.markdown(
    "An end-to-end machine learning system for predicting delivery durations "
    "in Hong Kong, built on ~10K real last-mile logistics records (2019–2020)."
)

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Training Rows", "9,950")
col2.metric("Model", "XGBoost")
col3.metric("R² Score", f"{metrics.get('r2', 0):.3f}" if metrics else "—")
col4.metric("MAE (delay ratio)", f"{metrics.get('mae', 0):.3f}" if metrics else "—")

st.divider()

st.subheader("About this project")
st.markdown(
    """
This project explores delivery ETA prediction for Hong Kong logistics operations.

**Dataset:**  65K+ delivery segment records from a last-mile logistics operator covering
multiple warehouses and vehicle types (9-tonne trucks, vans, 16-tonne trucks).

**Key insight:**
The ratio of *actual delivery time* to *planned delivery time* (**delay ratio**) is the most
predictive and generalisable target — it removes the trivial effect of route length and lets
the model focus on operational and traffic patterns.

**What the model learns:**
- Vehicle type is the dominant predictor (49 % importance) — larger trucks take longer per segment
- Origin location matters more than destination — warehouse-to-first-stop distance is a key factor
- Planned duration is a strong prior — planners have embedded route knowledge
- Morning peak (7–9 am) increases delays; weekends reduce them

**Navigate using the sidebar:**
| Page | What it does |
|---|---|
| 🔮 ETA Predictor | Input a delivery job, get a predicted actual duration |
| 📊 Analytics | Explore delay patterns by time, vehicle, and warehouse |
| 🗺️ Route Map | Interactive map of deliveries and delay hotspots |
    """
)

st.divider()
st.caption("Data: anonymised HK logistics records 2019–2020 · Model: XGBoost Regressor")
