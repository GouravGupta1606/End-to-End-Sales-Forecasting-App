"""
visualization.py — Clean, reusable Matplotlib/Seaborn plotting utilities
for EcoYield model diagnostics and EDA.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  SHARED STYLE
# ─────────────────────────────────────────────

PALETTE   = ["#2ECC71", "#E74C3C", "#3498DB", "#F39C12", "#9B59B6"]
BG_COLOR  = "#0F1117"
TEXT_COLOR = "#ECEFF1"

def _apply_dark_style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(BG_COLOR)
    ax.figure.patch.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=12, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


# ─────────────────────────────────────────────
#  FORECAST CHART
# ─────────────────────────────────────────────

def plot_series(actual: np.ndarray, pred: np.ndarray = None,
                title: str = "Sales — Actual vs Predicted",
                dates: pd.DatetimeIndex = None):
    fig, ax = plt.subplots(figsize=(12, 4))
    x = dates if dates is not None else np.arange(len(actual))
    ax.plot(x, actual, label="Actual", color=PALETTE[2], linewidth=1.5)
    if pred is not None:
        ax.plot(x, pred, label="Predicted", color=PALETTE[0],
                linewidth=1.5, linestyle="--")
    ax.legend(facecolor="#1E1E2E", labelcolor=TEXT_COLOR)
    _apply_dark_style(ax, title)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  TRAINING HISTORY
# ─────────────────────────────────────────────

def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     color=PALETTE[2], label="Train")
    axes[0].plot(history.history["val_loss"], color=PALETTE[1], label="Val")
    _apply_dark_style(axes[0], "Loss (MSE)", "Epoch", "MSE")
    axes[0].legend(facecolor="#1E1E2E", labelcolor=TEXT_COLOR)

    if "mae" in history.history:
        axes[1].plot(history.history["mae"],     color=PALETTE[2], label="Train")
        axes[1].plot(history.history["val_mae"], color=PALETTE[1], label="Val")
        _apply_dark_style(axes[1], "Mean Absolute Error", "Epoch", "MAE")
        axes[1].legend(facecolor="#1E1E2E", labelcolor=TEXT_COLOR)

    fig.patch.set_facecolor(BG_COLOR)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  SEASONAL DECOMPOSITION
# ─────────────────────────────────────────────

def plot_decomposition(series: pd.Series, period: int = 30):
    from statsmodels.tsa.seasonal import seasonal_decompose

    if len(series) < 2 * period:
        print(f"[plot_decomposition] Need ≥{2*period} points, got {len(series)}. Skipping.")
        return

    result = seasonal_decompose(series, period=period, model="additive", extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    labels = ["Observed", "Trend", "Seasonal", "Residual"]
    components = [result.observed, result.trend, result.seasonal, result.resid]

    for ax, comp, lbl in zip(axes, components, labels):
        ax.plot(comp, color=PALETTE[2], linewidth=1)
        _apply_dark_style(ax, lbl)

    fig.patch.set_facecolor(BG_COLOR)
    plt.suptitle("Seasonal Decomposition", color=TEXT_COLOR, fontsize=14)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  ACF / PACF
# ─────────────────────────────────────────────

def plot_autocorrelation(series: pd.Series, lags: int = 40):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    if len(series) < 10:
        print("[plot_autocorrelation] Not enough data. Skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf( series, lags=lags, ax=axes[0], color=PALETTE[0])
    plot_pacf(series, lags=lags, ax=axes[1], color=PALETTE[0])
    for ax, title in zip(axes, ["Autocorrelation (ACF)", "Partial ACF (PACF)"]):
        _apply_dark_style(ax, title, "Lag", "Correlation")

    fig.patch.set_facecolor(BG_COLOR)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  RESIDUAL DIAGNOSTICS
# ─────────────────────────────────────────────

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray):
    residuals = y_true - y_pred

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # Histogram
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(residuals, bins=50, color=PALETTE[2], edgecolor="none", alpha=0.85)
    _apply_dark_style(ax1, "Residual Distribution", "Residual", "Count")

    # Residuals vs predicted
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(y_pred, residuals, alpha=0.3, s=6, color=PALETTE[0])
    ax2.axhline(0, color=PALETTE[1], linewidth=1.2)
    _apply_dark_style(ax2, "Residuals vs Predicted", "Predicted", "Residual")

    # QQ plot
    ax3 = fig.add_subplot(gs[2])
    stats.probplot(residuals, plot=ax3)
    ax3.get_lines()[0].set(color=PALETTE[2], markersize=2, alpha=0.5)
    ax3.get_lines()[1].set(color=PALETTE[1])
    _apply_dark_style(ax3, "QQ Plot")

    fig.patch.set_facecolor(BG_COLOR)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  CORRELATION HEATMAP
# ─────────────────────────────────────────────

def correlation_heatmap(df: pd.DataFrame, title: str = "Feature Correlations"):
    corr = df.select_dtypes(include=np.number).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0,
        linewidths=0.5, linecolor="#1E1E2E",
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    _apply_dark_style(ax, title)
    fig.patch.set_facecolor(BG_COLOR)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  FEATURE IMPORTANCE (SHAP-free permutation)
# ─────────────────────────────────────────────

def plot_feature_importance(feature_names: list, importances: np.ndarray):
    """Bar chart for any importance scores (e.g., permutation importance)."""
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    ax.barh(df["feature"], df["importance"], color=PALETTE[0], edgecolor="none")
    _apply_dark_style(ax, "Feature Importance", "Importance Score", "Feature")
    plt.tight_layout()
    plt.show()
