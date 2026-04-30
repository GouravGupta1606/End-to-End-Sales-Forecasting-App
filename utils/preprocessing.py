import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────
#  RAW DATA LOADER
# ─────────────────────────────────────────────

def load_raw_data(path: str, sample_size: int = 500_000) -> pd.DataFrame:
    """
    Load the Favorita-style train.csv, keep only relevant columns,
    clean dtypes, and sort chronologically.
    """
    df = pd.read_csv(path, nrows=sample_size, parse_dates=["date"])

    df = df[["date", "unit_sales", "onpromotion"]].copy()
    df.columns = ["date", "sales", "promo"]

    # Clip negative sales (returns) to zero — they break log transforms & MAPE
    df["sales"] = df["sales"].clip(lower=0)
    df["promo"] = df["promo"].fillna(0).astype(int)

    df = df.sort_values("date").reset_index(drop=True)

    print(f"[load_raw_data] shape={df.shape}  date range: {df['date'].min()} → {df['date'].max()}")
    return df


# ─────────────────────────────────────────────
#  AGGREGATION
# ─────────────────────────────────────────────

def load_aggregated_data(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse store-level rows to daily totals."""
    df_grouped = (
        df.groupby("date")
        .agg(sales=("sales", "sum"), promo=("promo", "sum"))
        .reset_index()
    )
    print(f"[load_aggregated_data] shape={df_grouped.shape}")
    return df_grouped


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal lag features, rolling statistics, and calendar features.
    Works on either raw or aggregated data (requires 'sales' and 'date' columns).
    """
    df = df.copy()

    # --- Lag features ---
    for lag in [1, 7, 14, 21]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    # --- Rolling statistics ---
    for w in [7, 14, 30]:
        df[f"rolling_mean_{w}"] = df["sales"].rolling(w).mean()
        df[f"rolling_std_{w}"]  = df["sales"].rolling(w).std()

    # --- Expanding (cumulative) mean ---
    df["expanding_mean"] = df["sales"].expanding().mean()

    # --- Calendar features ---
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["day_of_month"] = df["date"].dt.day
    df["month"]        = df["date"].dt.month
    df["quarter"]      = df["date"].dt.quarter
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"[feature_engineering] {before} → {len(df)} rows after dropping NaN from lags/rolling")
    return df


# ─────────────────────────────────────────────
#  SCALING
# ─────────────────────────────────────────────

def scale_data(df: pd.DataFrame):
    """
    MinMax-scale all numeric feature columns (everything except 'date').
    Returns (scaled_array, fitted_scaler, feature_columns).
    """
    features = df.drop(columns=["date"])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler, list(features.columns)


# ─────────────────────────────────────────────
#  SEQUENCE BUILDER
# ─────────────────────────────────────────────

def create_sequences(data: np.ndarray, time_steps: int = 30):
    """
    Convert a 2-D scaled array into supervised LSTM sequences.
    Target is always column 0 (sales) at t+1.
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps, 0])
    print(f"[create_sequences] X={np.array(X).shape}  y={np.array(y).shape}")
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
#  TRAIN / VALIDATION / TEST SPLIT
# ─────────────────────────────────────────────

def time_split(X: np.ndarray, y: np.ndarray, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Chronological split into train / validation / test sets.
    Default: 70 % / 15 % / 15 %.
    """
    n = len(X)
    i_train = int(n * train_frac)
    i_val   = int(n * (train_frac + val_frac))

    X_train, y_train = X[:i_train],        y[:i_train]
    X_val,   y_val   = X[i_train:i_val],   y[i_train:i_val]
    X_test,  y_test  = X[i_val:],          y[i_val:]

    print(f"[time_split] train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
