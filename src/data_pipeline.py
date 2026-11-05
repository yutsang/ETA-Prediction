"""
Data pipeline for HK ETA prediction.

Loads df_hkg_cleansed.csv, cleans, engineers features, and saves a parquet
file ready for model training and dashboard use.

Usage:
    python src/data_pipeline.py
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.geo import haversine_km, parse_coord, classify_district

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "ETA-Prediction-Data", "df_hkg_cleansed.csv")
OUT_PATH = os.path.join(ROOT, "data", "processed", "hk_features.parquet")

# ---------------------------------------------------------------------------
# Vehicle category mapping
# ---------------------------------------------------------------------------
_VEHICLE_MAP = {
    "VAN-3_2": "VAN", "VAN-1_6": "VAN",
    "9T-17": "MEDIUM", "9T-13_6": "MEDIUM", "7_5T-6_8-H": "MEDIUM",
    "16T-20_4": "LARGE", "24T-20_4-H": "LARGE",
    "ROLLOUT_DUMMY": "OTHER", "LF-DUMMY": "OTHER",
}

def _map_vehicle(v: str) -> str:
    for prefix, cat in _VEHICLE_MAP.items():
        if str(v).startswith(prefix.split("-")[0]):
            return cat
    return "OTHER"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_features(raw_path: str = RAW_PATH) -> pd.DataFrame:
    print(f"Loading {raw_path} ...")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    # --- Parse durations -------------------------------------------------------
    df["actDurn_td"]  = pd.to_timedelta(df["actDurn"],  errors="coerce")
    df["planDurn_td"] = pd.to_timedelta(df["planDurn"], errors="coerce")
    df["actDurn_min"]  = df["actDurn_td"].dt.total_seconds()  / 60
    df["planDurn_min"] = df["planDurn_td"].dt.total_seconds() / 60

    # --- Keep rows with valid, positive durations ------------------------------
    df = df[(df["actDurn_min"] > 0) & (df["actDurn_min"] < 1440)
            & (df["planDurn_min"] > 5)].copy()
    print(f"  After duration filter: {len(df):,}")

    # --- Delay ratio -----------------------------------------------------------
    df["delay_ratio"] = df["actDurn_min"] / df["planDurn_min"]
    df = df[(df["delay_ratio"] >= 0.1) & (df["delay_ratio"] <= 20)].copy()
    print(f"  After ratio filter (0.1–20): {len(df):,}")

    # --- Parse timestamps ------------------------------------------------------
    df["planStart"] = pd.to_datetime(df["planStartTime"], errors="coerce", dayfirst=True)
    df = df[df["planStart"].notna()].copy()

    df["hour_of_day"]       = df["planStart"].dt.hour
    df["day_of_week"]       = df["planStart"].dt.dayofweek   # 0=Mon
    df["month"]             = df["planStart"].dt.month
    df["is_morning_peak"]   = df["hour_of_day"].between(7, 9).astype(int)
    df["is_evening_peak"]   = df["hour_of_day"].between(17, 20).astype(int)
    df["is_weekend"]        = (df["day_of_week"] >= 5).astype(int)

    # --- Geospatial features ---------------------------------------------------
    orig_parsed = df["origLoc"].apply(parse_coord)
    dest_parsed = df["destLoc"].apply(parse_coord)

    df["orig_lat"] = orig_parsed.apply(lambda x: x[0])
    df["orig_lon"] = orig_parsed.apply(lambda x: x[1])
    df["dest_lat"] = dest_parsed.apply(lambda x: x[0])
    df["dest_lon"] = dest_parsed.apply(lambda x: x[1])

    # Drop rows where we couldn't parse coords
    df = df[df["orig_lat"].notna() & df["dest_lat"].notna()].copy()

    df["haversine_km"] = df.apply(
        lambda r: haversine_km(r["orig_lat"], r["orig_lon"],
                               r["dest_lat"], r["dest_lon"]),
        axis=1
    )

    df["orig_district"] = df.apply(
        lambda r: classify_district(r["orig_lat"], r["orig_lon"]), axis=1
    )
    df["dest_district"] = df.apply(
        lambda r: classify_district(r["dest_lat"], r["dest_lon"]), axis=1
    )

    # --- Vehicle category ------------------------------------------------------
    df["vehicle_cat"] = df["Vehicle"].apply(_map_vehicle)

    # --- Warehouse encoding (label) --------------------------------------------
    wh_cats = df["warehouse"].astype("category")
    df["warehouse_enc"] = wh_cats.cat.codes
    df["warehouse_labels"] = df["warehouse"].fillna("UNKNOWN")

    # --- Log transform of target for model training ----------------------------
    df["log_delay_ratio"] = np.log(df["delay_ratio"])

    # --- Select final columns --------------------------------------------------
    feature_cols = [
        "haversine_km", "hour_of_day", "day_of_week", "month",
        "is_morning_peak", "is_evening_peak", "is_weekend",
        "planned_duration_min", "vehicle_cat", "warehouse_enc",
        "orig_lat", "orig_lon", "dest_lat", "dest_lon",
        "orig_district", "dest_district",
    ]
    df = df.rename(columns={"planDurn_min": "planned_duration_min"})

    keep = [
        "planStart", "Vehicle", "vehicle_cat", "warehouse_labels",
        "warehouse_enc", "haversine_km",
        "hour_of_day", "day_of_week", "month",
        "is_morning_peak", "is_evening_peak", "is_weekend",
        "planned_duration_min", "actDurn_min", "delay_ratio", "log_delay_ratio",
        "orig_lat", "orig_lon", "dest_lat", "dest_lon",
        "orig_district", "dest_district",
    ]
    df = df[keep].reset_index(drop=True)
    print(f"  Final rows: {len(df):,}")
    return df


def run(raw_path: str = RAW_PATH, out_path: str = OUT_PATH) -> pd.DataFrame:
    df = build_features(raw_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved → {out_path}")
    return df


if __name__ == "__main__":
    df = run()
    print("\nSample:")
    print(df[["haversine_km", "hour_of_day", "day_of_week",
              "vehicle_cat", "planned_duration_min",
              "actDurn_min", "delay_ratio"]].head())
    print("\ndelay_ratio stats:")
    print(df["delay_ratio"].describe())
