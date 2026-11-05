"""HK Traffic Analytics Dashboard page."""
import os, sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_PATH = os.path.join(ROOT, "data", "processed", "hk_features.parquet")

st.set_page_config(page_title="HK Analytics", page_icon="📊", layout="wide")
st.title("📊 HK Delivery Analytics")
st.markdown("Explore delay patterns across time, vehicle type, and warehouse using the cleaned HKG dataset.")

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet(FEATURES_PATH)
    df["planStart"] = pd.to_datetime(df["planStart"])
    df["year_month"] = df["planStart"].dt.to_period("M").astype(str)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df["day_name"] = df["day_of_week"].map(lambda d: day_names[d])
    return df

if not os.path.exists(FEATURES_PATH):
    st.error("Features parquet not found. Run `python src/data_pipeline.py` first.")
    st.stop()

df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────
st.sidebar.header("Filters")
vehicles = ["All"] + sorted(df["vehicle_cat"].unique().tolist())
sel_vehicle = st.sidebar.selectbox("Vehicle category", vehicles)

warehouses = ["All"] + sorted(df["warehouse_labels"].unique().tolist())
sel_warehouse = st.sidebar.selectbox("Warehouse", warehouses)

dff = df.copy()
if sel_vehicle != "All":
    dff = dff[dff["vehicle_cat"] == sel_vehicle]
if sel_warehouse != "All":
    dff = dff[dff["warehouse_labels"] == sel_warehouse]

st.caption(f"Showing **{len(dff):,}** delivery records")

# ── Row 1: KPI cards ────────────────────────────────────────────────────────
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Median delay ratio",  f"{dff['delay_ratio'].median():.2f}×")
k2.metric("Mean actual duration", f"{dff['actDurn_min'].mean():.0f} min")
k3.metric("Mean planned duration", f"{dff['planned_duration_min'].mean():.0f} min")
k4.metric("% delivered on time (ratio ≤ 1)",
          f"{100 * (dff['delay_ratio'] <= 1).mean():.0f}%")

st.divider()

# ── Row 2: Delay by hour + by day ──────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Delay ratio by hour of day")
    hourly = dff.groupby("hour_of_day")["delay_ratio"].median().reset_index()
    hourly.columns = ["Hour", "Median delay ratio"]
    fig = px.bar(
        hourly, x="Hour", y="Median delay ratio",
        color="Median delay ratio",
        color_continuous_scale="RdYlGn_r",
        labels={"Median delay ratio": "Median delay ratio"},
        template="plotly_dark",
    )
    fig.add_vrect(x0=6.5, x1=9.5, fillcolor="orange", opacity=0.15,
                  annotation_text="Morning peak", annotation_position="top left")
    fig.add_vrect(x0=16.5, x1=20.5, fillcolor="red", opacity=0.10,
                  annotation_text="Evening peak", annotation_position="top left")
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("Delay ratio by day of week")
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = dff.groupby("day_name")["delay_ratio"].median().reindex(day_order).reset_index()
    daily.columns = ["Day", "Median delay ratio"]
    fig2 = px.bar(
        daily, x="Day", y="Median delay ratio",
        color="Median delay ratio",
        color_continuous_scale="RdYlGn_r",
        template="plotly_dark",
    )
    fig2.update_layout(coloraxis_showscale=False, margin=dict(t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 3: Vehicle + warehouse ─────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Delay ratio by vehicle category")
    fig3 = px.box(
        dff, x="vehicle_cat", y="delay_ratio",
        color="vehicle_cat",
        labels={"vehicle_cat": "Vehicle", "delay_ratio": "Delay ratio"},
        template="plotly_dark",
        points=False,
    )
    fig3.update_layout(showlegend=False, yaxis_range=[0, 10], margin=dict(t=30, b=0))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Warehouse performance")
    wh = dff.groupby("warehouse_labels").agg(
        median_ratio=("delay_ratio", "median"),
        count=("delay_ratio", "count"),
    ).reset_index().sort_values("median_ratio")
    fig4 = px.bar(
        wh, x="warehouse_labels", y="median_ratio",
        text="count",
        color="median_ratio",
        color_continuous_scale="RdYlGn_r",
        labels={"warehouse_labels": "Warehouse", "median_ratio": "Median delay ratio"},
        template="plotly_dark",
    )
    fig4.update_traces(texttemplate="%{text:,} trips", textposition="outside")
    fig4.update_layout(coloraxis_showscale=False, margin=dict(t=30, b=0))
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 4: Monthly trend ───────────────────────────────────────────────────
st.subheader("Monthly trend — median delay ratio")
monthly = dff.groupby("year_month")["delay_ratio"].median().reset_index()
monthly.columns = ["Month", "Median delay ratio"]
fig5 = px.line(
    monthly, x="Month", y="Median delay ratio",
    markers=True,
    template="plotly_dark",
)
fig5.add_hline(y=1.0, line_dash="dash", line_color="grey",
               annotation_text="On-time baseline (ratio = 1)")
fig5.update_layout(margin=dict(t=30, b=0))
st.plotly_chart(fig5, use_container_width=True)

# ── Row 5: Delay distribution ──────────────────────────────────────────────
st.subheader("Distribution: actual vs planned duration (minutes)")
col5, col6 = st.columns(2)

with col5:
    fig6 = px.histogram(
        dff, x="actDurn_min",
        nbins=60,
        labels={"actDurn_min": "Actual duration (min)"},
        template="plotly_dark",
        color_discrete_sequence=["#00b4d8"],
    )
    fig6.update_layout(margin=dict(t=30, b=0))
    st.plotly_chart(fig6, use_container_width=True)

with col6:
    fig7 = px.histogram(
        dff, x="planned_duration_min",
        nbins=60,
        labels={"planned_duration_min": "Planned duration (min)"},
        template="plotly_dark",
        color_discrete_sequence=["#f77f00"],
    )
    fig7.update_layout(margin=dict(t=30, b=0))
    st.plotly_chart(fig7, use_container_width=True)

# ── Feature importance ─────────────────────────────────────────────────────
import json
meta_path = os.path.join(ROOT, "models", "xgb_meta.json")
if os.path.exists(meta_path):
    st.divider()
    st.subheader("Model — feature importance (XGBoost)")
    with open(meta_path) as f:
        meta = json.load(f)
    fi = meta.get("feature_importance", {})
    fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
    fi_df = fi_df.sort_values("Importance", ascending=True)
    fig8 = px.bar(
        fi_df, x="Importance", y="Feature",
        orientation="h",
        template="plotly_dark",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig8.update_layout(coloraxis_showscale=False, margin=dict(t=30, b=0))
    st.plotly_chart(fig8, use_container_width=True)
