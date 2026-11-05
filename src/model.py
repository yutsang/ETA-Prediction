"""
XGBoost model for HK delivery ETA prediction.

Predicts log(delay_ratio) from route and time features.
delay_ratio = actual_duration / planned_duration

Usage:
    python src/model.py          # train and save
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(ROOT, "data", "processed", "hk_features.parquet")
MODEL_PATH     = os.path.join(ROOT, "models", "xgb_eta.pkl")
META_PATH      = os.path.join(ROOT, "models", "xgb_meta.json")

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "haversine_km",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_morning_peak",
    "is_evening_peak",
    "is_weekend",
    "planned_duration_min",
    "warehouse_enc",
    "orig_lat", "orig_lon",
    "dest_lat", "dest_lon",
]
CAT_FEATURE = "vehicle_cat"  # encoded separately
TARGET = "log_delay_ratio"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(features_path: str = FEATURES_PATH,
          model_path: str = MODEL_PATH,
          meta_path: str = META_PATH) -> dict:

    print("Loading features ...")
    df = pd.read_parquet(features_path)
    print(f"  Rows: {len(df):,}")

    # Encode vehicle_cat
    le = LabelEncoder()
    df["vehicle_enc"] = le.fit_transform(df["vehicle_cat"].astype(str))
    vehicle_classes = list(le.classes_)

    feature_cols = NUMERIC_FEATURES + ["vehicle_enc"]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- Evaluation -----------------------------------------------------------
    y_pred_log = model.predict(X_test)
    y_pred     = np.exp(y_pred_log)
    y_true     = np.exp(y_test)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    print(f"\nTest metrics (delay ratio space):")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    top5 = sorted(importance.items(), key=lambda x: -x[1])[:5]
    print(f"\nTop-5 features: {top5}")

    # --- Save -----------------------------------------------------------------
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le}, model_path)

    meta = {
        "feature_cols": feature_cols,
        "vehicle_classes": vehicle_classes,
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "feature_importance": importance,
        "best_iteration": int(model.best_iteration),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved → {model_path}")
    print(f"Meta  saved → {meta_path}")
    return meta


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def load_model(model_path: str = MODEL_PATH, meta_path: str = META_PATH):
    artifact = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return artifact["model"], artifact["label_encoder"], meta


def predict(orig_lat: float, orig_lon: float,
            dest_lat: float, dest_lon: float,
            vehicle_cat: str,
            planned_duration_min: float,
            hour: int, day_of_week: int, month: int,
            warehouse_enc: int = 0,
            model_path: str = MODEL_PATH,
            meta_path: str = META_PATH) -> dict:
    """
    Returns:
        predicted_delay_ratio  : float
        predicted_actual_min   : float
        confidence_low_min     : float  (approx. ±1 std in log space)
        confidence_high_min    : float
    """
    from src.geo import haversine_km

    model, le, meta = load_model(model_path, meta_path)
    std_log = 0.918  # empirical std of log_delay_ratio from training data

    hav = haversine_km(orig_lat, orig_lon, dest_lat, dest_lon)
    is_morning = int(7 <= hour <= 9)
    is_evening = int(17 <= hour <= 20)
    is_weekend = int(day_of_week >= 5)

    # Encode vehicle
    try:
        vehicle_enc = int(le.transform([vehicle_cat])[0])
    except ValueError:
        vehicle_enc = 0

    row = np.array([[
        hav, hour, day_of_week, month,
        is_morning, is_evening, is_weekend,
        planned_duration_min, warehouse_enc,
        orig_lat, orig_lon, dest_lat, dest_lon,
        vehicle_enc,
    ]], dtype=np.float32)

    log_ratio_pred = float(model.predict(row)[0])
    ratio = float(np.exp(log_ratio_pred))
    actual_min = planned_duration_min * ratio

    low_min  = planned_duration_min * np.exp(log_ratio_pred - std_log)
    high_min = planned_duration_min * np.exp(log_ratio_pred + std_log)

    return {
        "predicted_delay_ratio": ratio,
        "predicted_actual_min": actual_min,
        "confidence_low_min": float(low_min),
        "confidence_high_min": float(high_min),
    }


if __name__ == "__main__":
    train()
