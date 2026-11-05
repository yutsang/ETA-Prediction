"""Interactive route map — delivery density and delay hotspots across HK."""
import os, sys
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_PATH = os.path.join(ROOT, "data", "processed", "hk_features.parquet")

st.set_page_config(page_title="Route Map", page_icon="🗺️", layout="wide")
st.title("🗺️ HK Delivery Route Map")
st.markdown(
    "Explore where deliveries happen and which areas experience the highest delays. "
    "Use the sidebar to filter and toggle map layers."
)

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet(FEATURES_PATH)
    df = df.dropna(subset=["orig_lat", "orig_lon", "dest_lat", "dest_lon"])
    return df

if not os.path.exists(FEATURES_PATH):
    st.error("Features parquet not found. Run `python src/data_pipeline.py` first.")
    st.stop()

df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────
st.sidebar.header("Map Filters")
vehicles = ["All"] + sorted(df["vehicle_cat"].unique().tolist())
sel_vehicle = st.sidebar.selectbox("Vehicle category", vehicles)

warehouses = ["All"] + sorted(df["warehouse_labels"].unique().tolist())
sel_warehouse = st.sidebar.selectbox("Warehouse", warehouses)

show_heatmap    = st.sidebar.checkbox("Delivery density heatmap", value=True)
show_origins    = st.sidebar.checkbox("Origin points (delay-coloured)", value=True)
show_arcs       = st.sidebar.checkbox("Route arcs (sample)", value=False)
max_ratio_filter = st.sidebar.slider("Max delay ratio to show", 1.0, 20.0, 20.0, 0.5)

# ── Filter ─────────────────────────────────────────────────────────────────
dff = df.copy()
if sel_vehicle != "All":
    dff = dff[dff["vehicle_cat"] == sel_vehicle]
if sel_warehouse != "All":
    dff = dff[dff["warehouse_labels"] == sel_warehouse]
dff = dff[dff["delay_ratio"] <= max_ratio_filter]

st.caption(f"Plotting **{len(dff):,}** delivery records")

# ── Colour helper (delay ratio → RGB) ─────────────────────────────────────
def ratio_to_rgb(ratio, vmin=0.2, vmax=5.0):
    t = np.clip((ratio - vmin) / (vmax - vmin), 0, 1)
    r = int(255 * t)
    g = int(255 * (1 - t))
    return [r, g, 50, 180]

# ── Aggregate origin points by lat/lon ─────────────────────────────────────
orig_agg = (
    dff.groupby(["orig_lat", "orig_lon"])
    .agg(count=("delay_ratio", "count"), avg_ratio=("delay_ratio", "median"))
    .reset_index()
)
orig_agg["color"]  = orig_agg["avg_ratio"].apply(ratio_to_rgb)
orig_agg["radius"] = (orig_agg["count"] * 15).clip(upper=1500)

dest_agg = (
    dff.groupby(["dest_lat", "dest_lon"])
    .agg(count=("delay_ratio", "count"), avg_ratio=("delay_ratio", "median"))
    .reset_index()
)

# ── Build layers ────────────────────────────────────────────────────────────
layers = []

if show_heatmap:
    heatmap_data = dest_agg[["dest_lat", "dest_lon", "count"]].rename(
        columns={"dest_lat": "lat", "dest_lon": "lon"}
    )
    layers.append(
        pdk.Layer(
            "HeatmapLayer",
            data=heatmap_data,
            get_position=["lon", "lat"],
            get_weight="count",
            radius_pixels=40,
            opacity=0.55,
        )
    )

if show_origins:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=orig_agg,
            get_position=["orig_lon", "orig_lat"],
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            opacity=0.8,
        )
    )

if show_arcs:
    arc_sample = dff.sample(min(300, len(dff)), random_state=42)
    arc_sample = arc_sample.copy()
    arc_sample["src_color"] = arc_sample["delay_ratio"].apply(
        lambda r: ratio_to_rgb(r)[:3]
    )
    layers.append(
        pdk.Layer(
            "ArcLayer",
            data=arc_sample,
            get_source_position=["orig_lon", "orig_lat"],
            get_target_position=["dest_lon", "dest_lat"],
            get_source_color="src_color",
            get_target_color=[100, 100, 220],
            get_width=1.5,
            pickable=True,
            opacity=0.6,
        )
    )

# ── Render map ──────────────────────────────────────────────────────────────
view = pdk.ViewState(
    latitude=22.35,
    longitude=114.10,
    zoom=10,
    pitch=40,
    bearing=0,
)

tooltip = {
    "html": "<b>Trips:</b> {count}<br/><b>Median delay ratio:</b> {avg_ratio:.2f}×",
    "style": {"background": "#1a1a2e", "color": "white", "font-size": "12px"},
}

st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip=tooltip,
        height=600,
    ),
    use_container_width=True,
)

# ── Legend ──────────────────────────────────────────────────────────────────
st.markdown(
    """
**Origin point colours** (delay ratio):
🟢 Low delay (< 1×) &nbsp;&nbsp; 🟡 Moderate (1–3×) &nbsp;&nbsp; 🔴 High delay (> 3×)

**Heatmap** shows destination delivery density (brighter = more deliveries).
    """
)

# ── District summary table ──────────────────────────────────────────────────
st.divider()
st.subheader("Delay summary by origin district")
if "orig_district" in dff.columns:
    district_summary = (
        dff.groupby("orig_district")
        .agg(
            trips=("delay_ratio", "count"),
            median_delay=("delay_ratio", "median"),
            mean_actual_min=("actDurn_min", "mean"),
        )
        .reset_index()
        .sort_values("median_delay", ascending=False)
        .rename(columns={
            "orig_district": "District",
            "trips": "Trips",
            "median_delay": "Median delay ratio",
            "mean_actual_min": "Mean actual duration (min)",
        })
    )
    district_summary["Median delay ratio"] = district_summary["Median delay ratio"].round(2)
    district_summary["Mean actual duration (min)"] = district_summary["Mean actual duration (min)"].round(0).astype(int)
    st.dataframe(district_summary, use_container_width=True, hide_index=True)
