import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  MODEL EVALUATION
# ─────────────────────────────────────────────

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, MAPE, and R² between true and predicted values.
    Handles zero-valued ground truth gracefully to avoid division errors.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    # Avoid divide-by-zero for MAPE
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


# ─────────────────────────────────────────────
#  PRICING
# ─────────────────────────────────────────────

def price_elasticity(df: pd.DataFrame) -> float:
    """
    Pearson correlation between unit price and quantity as a proxy for
    price elasticity of demand.  Values closer to -1 indicate elastic demand.
    """
    if "Unit price" not in df.columns or "Quantity" not in df.columns:
        return float("nan")
    return float(df["Unit price"].corr(df["Quantity"]))


def price_elasticity_by_category(df: pd.DataFrame) -> pd.Series:
    """Per product-line price elasticity."""
    if "Product line" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("Product line").apply(
        lambda g: g["Unit price"].corr(g["Quantity"]) if len(g) > 1 else float("nan")
    ).rename("elasticity")


def profit_margin_analysis(df):
    """
    Returns aggregated metrics by product line:
    - avg_price
    - avg_quantity
    - avg_profit
    - total_revenue
    """

    # ✅ Normalize column names (IMPORTANT)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # ✅ Handle profit column naming
    if "gross_income" in df.columns:
        df["profit"] = df["gross_income"]
    elif "gross income" in df.columns:
        df["profit"] = df["gross income"]
    elif "profit" not in df.columns:
        raise ValueError("No profit/gross_income column found")

    # ✅ Ensure total_revenue column exists
    if "total" in df.columns:
        df["total_revenue"] = df["total"]
    elif "unit_price" in df.columns and "quantity" in df.columns:
        df["total_revenue"] = df["unit_price"] * df["quantity"]
    else:
        raise ValueError("Cannot compute total_revenue")

    # ✅ Grouping + aggregation
    margin_df = (
        df.groupby("product_line")
        .agg(
            avg_price=("unit_price", "mean"),
            avg_quantity=("quantity", "mean"),
            avg_profit=("profit", "mean"),
            total_revenue=("total_revenue", "sum")
        )
        .reset_index()
    )

    margin_df.columns = (
    margin_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    )

    return margin_df
# ─────────────────────────────────────────────
#  CUSTOMER ANALYTICS
# ─────────────────────────────────────────────

def customer_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    if "Customer type" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df.columns = df.columns.str.strip()

    #  Create Total if missing
    if "Total" not in df.columns:
        if "Unit price" in df.columns and "Quantity" in df.columns:
            df["Total"] = df["Unit price"] * df["Quantity"]
        else:
            return pd.DataFrame()

    return (
        df.groupby("Customer type")
        .agg(
            total_spend=("Total", "sum"),
            transactions=("Total", "count"),
        )
        .reset_index()
    )


def customer_rating_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Average rating and satisfaction flag per customer type and gender."""
    cols = [c for c in ["Customer type", "Gender"] if c in df.columns]
    if not cols or "Rating" not in df.columns:
        return pd.DataFrame()
    result = df.groupby(cols)["Rating"].agg(["mean", "std", "count"]).round(2).reset_index()
    result.columns = cols + ["avg_rating", "std_rating", "count"]
    return result


# ─────────────────────────────────────────────
#  WALMART / RETAIL STORE ANALYTICS
# ─────────────────────────────────────────────

def walmart_insights(df: pd.DataFrame) -> pd.DataFrame:
    """Mean weekly sales on holiday vs non-holiday weeks."""
    if "Holiday_Flag" not in df.columns or "Weekly_Sales" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("Holiday_Flag")["Weekly_Sales"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .assign(Holiday=lambda x: x["Holiday_Flag"].map({0: "Regular", 1: "Holiday"}))
    )


def walmart_store_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Total and average weekly sales ranked by store."""
    if "Store" not in df.columns or "Weekly_Sales" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("Store")["Weekly_Sales"]
        .agg(total_sales="sum", avg_weekly="mean", weeks="count")
        .round(2)
        .sort_values("total_sales", ascending=False)
        .reset_index()
    )


def walmart_macro_correlations(df: pd.DataFrame) -> pd.Series:
    """Correlation of macroeconomic indicators with weekly sales."""
    macro_cols = [c for c in ["Temperature", "Fuel_Price", "CPI", "Unemployment"] if c in df.columns]
    if not macro_cols or "Weekly_Sales" not in df.columns:
        return pd.Series(dtype=float)
    return df[macro_cols + ["Weekly_Sales"]].corr(numeric_only=True)["Weekly_Sales"].drop("Weekly_Sales").sort_values()


# ─────────────────────────────────────────────
#  TIME-SERIES HELPERS
# ─────────────────────────────────────────────

def moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def detect_anomalies(series: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    """Return boolean mask where |z-score| > z_thresh."""
    z = (series - series.mean()) / series.std()
    return z.abs() > z_thresh


def growth_rate(series: pd.Series, periods: int = 1) -> pd.Series:
    """Period-over-period percentage growth."""
    return series.pct_change(periods=periods) * 100
