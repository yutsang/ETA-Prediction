# HK Logistics ETA Prediction

A machine learning system for predicting delivery durations in Hong Kong, using anonymised last-mile logistics data.

---

## Overview

This project treats commercial logistics records as a proxy for urban traffic patterns in Hong Kong. Rather than predicting raw delivery duration, the system predicts the **delay ratio** — the ratio of actual to planned delivery time — which removes the trivial effect of route length and focuses the model on operational and traffic-driven variance.

Key finding: delivery delay ratios encode time-of-day congestion signals even though the data was never collected as traffic data. Vehicle category and origin depot are stronger predictors than distance or time-of-day, suggesting operational factors dominate over pure traffic in this dataset.

---

## Dataset

- **Source:** Anonymised last-mile logistics records from a regional operator (2019–2020)
- **Coverage:** Hong Kong (primary), with records also covering CHN, SGP, THA, IDN, MYS, PHL, TWN
- **Raw size:** 65,330 HKG delivery segments
- **Usable after cleaning:** 9,950 records (85% had missing actual completion times — itself a data quality finding)
- **Warehouses:** Multiple depots; identifiers remain in the private dataset
- **Vehicle types:** 9-tonne trucks (dominant), vans, 16-tonne trucks

### Key columns

| Column | Description |
|---|---|
| `origLoc` / `destLoc` | Origin and destination coordinates (lat, lon) |
| `planStartTime` | Scheduled departure time |
| `planDurn` | Planned segment duration |
| `actDurn` | Actual segment duration (frequently missing) |
| `Vehicle` | Vehicle type identifier |
| `warehouse` | Depot code |

---

## Methodology

### 1. Data Cleaning

Raw data had significant quality issues: negative durations, missing actual end times, and extreme outliers. Cleaning steps:

- Convert `planDurn` and `actDurn` from timedelta strings
- Keep only records with positive actual duration under 24 hours
- Filter planned duration > 5 minutes (removes placeholder entries)
- Remove rows where `delay_ratio` falls outside [0.1, 20]
- Parse `planStartTime` with day-first format (`D/M/YYYY`)

### 2. Feature Engineering

| Feature | Source | Notes |
|---|---|---|
| `haversine_km` | Origin/dest coords | Straight-line distance proxy |
| `hour_of_day` | planStartTime | Key traffic signal |
| `day_of_week` | planStartTime | 0 = Monday |
| `is_morning_peak` | hour 7–9 | Binary flag |
| `is_evening_peak` | hour 17–20 | Binary flag |
| `is_weekend` | day_of_week ≥ 5 | Binary flag |
| `planned_duration_min` | planDurn | Planner's prior estimate |
| `vehicle_cat` | Vehicle string | Bucketed: VAN / MEDIUM / LARGE / OTHER |
| `warehouse_enc` | warehouse | Label encoded |
| `orig_lat`, `orig_lon` | origLoc | Depot location signal |

**Target variable:** `log(delay_ratio)` where `delay_ratio = actDurn / planDurn`
Log-transforming the target stabilises variance and makes the regression well-behaved.

### 3. Model

**XGBoost Regressor** trained on log-transformed delay ratio.

- 80/20 train/test split
- 500 estimators, learning rate 0.05, max depth 6, early stopping (50 rounds)
- Predictions converted back: `predicted_actual_duration = planned_duration × exp(model_output)`

**Test set performance:**

| Metric | Value |
|---|---|
| R² | 0.672 |
| MAE (delay ratio space) | 0.320× |
| RMSE (delay ratio space) | 1.098× |

**Feature importance (top 5):**

1. `vehicle_enc` — 49.8% (vehicle category dominates)
2. `orig_lat` — 8.3% (depot location matters)
3. `planned_duration_min` — 7.5% (planner's estimate is a strong prior)
4. `orig_lon` — 6.7%
5. `warehouse_enc` — 4.7%

### 4. Geospatial Analysis

- Haversine distance computed between origin and destination
- HK districts classified by bounding box (Hong Kong Island, Kowloon, New Territories, Lantau)
- Cross-Harbour Tunnel route detection explored via OSMnx road network (see `2.Coding Trials/osm.ipynb`)
- Route optimisation experiments using Google OR-Tools VRP solver (see `2.Coding Trials/lp_solver.ipynb`)

---

## Application

A three-page Streamlit dashboard:

### Page 1 — ETA Predictor
Input a delivery job (origin, destination, vehicle type, planned start time) and receive:
- Predicted actual duration
- Delay ratio
- 80% confidence interval
- Route arc visualisation on a Pydeck map

### Page 2 — HK Traffic Analytics
- Delay ratio by hour of day (morning/evening peak annotations)
- Delay ratio by day of week
- Vehicle category performance comparison (box plots)
- Warehouse performance ranking
- Monthly trend of median delay ratio
- Actual vs planned duration distributions
- XGBoost feature importance chart

### Page 3 — Route Map
- Pydeck heatmap of destination delivery density across HK
- Origin scatter plot coloured by median delay ratio (green = low, red = high)
- Arc layer showing sampled origin-to-destination routes
- District-level delay summary table
- Filterable by vehicle type, warehouse, and delay ratio

---

## Project Structure

```
├── app/
│   ├── Home.py                    # Streamlit entry point
│   └── pages/
│       ├── 1_ETA_Predictor.py
│       ├── 2_Analytics.py
│       └── 3_Route_Map.py
├── src/
│   ├── geo.py                     # Haversine, coordinate parser, district classifier
│   ├── data_pipeline.py           # Data cleaning and feature engineering
│   └── model.py                   # XGBoost training and inference
├── models/
│   ├── xgb_eta.pkl                # Trained model artifact
│   └── xgb_meta.json              # Metrics and feature importance
├── data/processed/
│   └── hk_features.parquet        # Engineered feature table
├── 2.Coding Trials/               # Exploratory notebooks (OSM routing, VRP, data cleaning)
├── ETA-Prediction-Data/           # Raw data files
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt

# Build features
python src/data_pipeline.py

# Train model
python src/model.py

# Launch app
streamlit run app/Home.py
```

---

## Limitations

- 85% of raw records lacked actual completion times, limiting training data volume
- Delay ratio < 1 for the majority of records — planners systematically over-estimate, so the model partially learns planning behaviour rather than traffic
- No external traffic data (speed, incidents, weather) was incorporated
- Model generalises within HKG only; other regions would need retraining
