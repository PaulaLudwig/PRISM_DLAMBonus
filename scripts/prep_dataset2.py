from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data" / "etth1"
RAW_PATH = DATA_DIR / "raw" / "ETTh1.csv"
TRAIN_OUT = DATA_DIR / "prepared" / "train.csv" 
VAL_OUT = DATA_DIR / "prepared" / "validation_input.csv"
    

HORIZON = 336

# columns that don't exist in ETTh1, just filled with 0, no information to match input to our model 
UNUSED_NULLABLE_COLS = [
    "shock_risk",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
]

FINAL_COLUMNS = [
    "series_id", "timestamp",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "workload_intensity", "promotion_intensity", "maintenance_known",
    "demand_forecast", "staffing_forecast", "upstream_quality_forecast",
    "shock_risk", "unit_reliability_forecast", "queue_pressure_forecast",
    "network_pressure_forecast", "event_load_forecast",
    "service_irregularity_risk_forecast", "throughput_disruption_risk_forecast",
    "nominal_capacity", "zone_sin", "zone_cos",
]


def add_calendar_features(df):
    hour = df["timestamp"].dt.hour
    dow = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(float)
    return df


def rename_covariates(df):
    df["workload_intensity"] = df["HUFL"]
    df["promotion_intensity"] = df["HULL"]
    df["maintenance_known"] = df["MUFL"]
    df["demand_forecast"] = df["MULL"]
    df["staffing_forecast"] = df["LUFL"]
    df["upstream_quality_forecast"] = df["LULL"]
    return df


def load_and_prepare():
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])

    df["series_id"] = "etth1"
    df["timestamp"] = df["date"]
    df["target"] = df["OT"]

    df = add_calendar_features(df) 
    df = rename_covariates(df)

    for col in UNUSED_NULLABLE_COLS:
        df[col] = 0.0

    # single series, constant padding 
    df["nominal_capacity"] = 0.0
    df["zone_sin"] = 0.0
    df["zone_cos"] = 0.0

    return df


def split_train_val(df):
    train_df = df.iloc[:-HORIZON].copy()
    val_df = df.iloc[-HORIZON:].copy()
    val_df = val_df.drop(columns=["target"])
    return train_df, val_df


def main():
    df = load_and_prepare() 
    train_df, val_df = split_train_val(df)

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    train_cols = FINAL_COLUMNS + ["target"]
    train_df[train_cols].to_csv(TRAIN_OUT, index=False)
    val_df[FINAL_COLUMNS].to_csv(VAL_OUT, index=False)
    
    print(f"train rows: {len(train_df)}")
    print(f"validation rows: {len(val_df)}")

if __name__ == "__main__":
    main()