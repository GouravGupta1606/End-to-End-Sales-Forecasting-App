"""
EcoYield — Enhanced Streamlit Dashboard
Run:  streamlit run app.py
"""

import os, sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _base)


def profit_margin_analysis(df):
    ...

from tensorflow.keras.models import load_model

# Support both layouts: project/utils/analytics.py and project/analytics.py
try:
    from utils.analytics import (
        price_elasticity, price_elasticity_by_category,
        profit_margin_analysis, customer_segmentation,
        customer_rating_analysis, walmart_insights,
        walmart_store_performance, walmart_macro_correlations,
        moving_average, detect_anomalies, growth_rate,
    )
except ModuleNotFoundError:
    from analytics import (
        price_elasticity, price_elasticity_by_category,
        profit_margin_analysis, customer_segmentation,
        customer_rating_analysis, walmart_insights,
        walmart_store_performance, walmart_macro_correlations,
        moving_average, detect_anomalies, growth_rate,
    )

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EcoYield Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL THEME / CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* Main background */
.stApp { background: #0A0D14; color: #C8D0E0; }
section[data-testid="stSidebar"] { background: #0D1118 !important; border-right: 1px solid #1C2333; }

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #2ECC71; }
.kpi-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1.4px; color: #6B7A99; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 600; color: #ECEFF4; font-family: 'DM Mono', monospace; }
.kpi-delta { font-size: 12px; margin-top: 4px; }
.kpi-delta.pos { color: #2ECC71; } .kpi-delta.neg { color: #E74C3C; }

/* Section header */
.section-title {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #2ECC71;
    border-left: 3px solid #2ECC71;
    padding-left: 10px;
    margin: 28px 0 14px;
}

/* Insight box */
.insight-box {
    background: #111827;
    border: 1px solid #1e2d45;
    border-left: 4px solid #2ECC71;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 13.5px;
    line-height: 1.7;
    color: #9BAEC8;
    margin-top: 16px;
}

/* Metric override */
[data-testid="stMetric"] { background: #111827; border-radius: 10px; padding: 12px 18px; border: 1px solid #1e2d45; }
[data-testid="stMetricLabel"] { font-size: 11px; color: #6B7A99 !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { font-size: 26px; font-weight: 600; color: #ECEFF4 !important; }
[data-testid="stMetricDelta"] { font-size: 12px; }

/* Sidebar radio */
div[data-testid="stRadio"] label { font-size: 14px; color: #9BAEC8; }

/* Tab styling */
button[data-baseweb="tab"] { font-size: 13px; color: #6B7A99; }
button[data-baseweb="tab"][aria-selected="true"] { color: #2ECC71 !important; border-bottom-color: #2ECC71 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-thumb { background: #2a3448; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY TEMPLATE
# ─────────────────────────────────────────────
CHART_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#0A0D14",
        plot_bgcolor="#111827",
        font=dict(family="DM Sans", color="#9BAEC8", size=12),
        xaxis=dict(gridcolor="#1C2333", zerolinecolor="#1C2333", showline=False),
        yaxis=dict(gridcolor="#1C2333", zerolinecolor="#1C2333", showline=False),
        legend=dict(bgcolor="#111827", bordercolor="#1C2333", borderwidth=1),
        colorway=["#2ECC71", "#3498DB", "#F39C12", "#E74C3C", "#9B59B6", "#1ABC9C"],
        margin=dict(t=40, l=10, r=10, b=10),
    )
)

def styled_fig(fig):
    fig.update_layout(**CHART_TEMPLATE["layout"])
    return fig

# ─────────────────────────────────────────────
#  PATHS  (edit these to match your machine)
# ─────────────────────────────────────────────
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH      = os.path.join(BASE, "model", "lstm_model.h5")
SCALER_PATH     = os.path.join(BASE, "model", "scaler.pkl")
FEATURE_PATH    = os.path.join(BASE, "model", "feature_count.pkl")
DATA_PATH       = os.path.join(BASE, "Data", "Raw", "train.csv")
WALMART_PATH    = os.path.join(BASE, "Data", "Raw", "Walmart.csv")
SUPERMARKET_PATH= os.path.join(BASE, "Data", "Raw", "SuperMarket Analysis.csv")

# ─────────────────────────────────────────────
#  RESOURCE LOADERS (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_all():
    model         = load_model(MODEL_PATH, compile=False)
    scaler        = joblib.load(SCALER_PATH)
    feature_count = joblib.load(FEATURE_PATH)
    return model, scaler, feature_count


@st.cache_data(show_spinner="Loading sales data…")
def load_sales():
    df = pd.read_csv(DATA_PATH, nrows=200_000, parse_dates=["date"])
    df["unit_sales"] = df["unit_sales"].clip(lower=0)
    return df


@st.cache_data(show_spinner="Loading Walmart data…")
def load_walmart():
    return pd.read_csv(WALMART_PATH)


@st.cache_data(show_spinner="Loading supermarket data…")
def load_supermarket():
    return pd.read_csv(SUPERMARKET_PATH)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 EcoYield")
    st.markdown("<div style='font-size:12px;color:#6B7A99;margin-bottom:20px;'>Retail Intelligence Platform</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📈 Forecast", "📊 Business Insights", "🏪 Walmart Analysis", "💰 Pricing Insights"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("<div style='font-size:11px;color:#444f66;'>Model: LSTM · Stacked · 2-layer<br>Scaler: MinMaxScaler<br>Sequence: 30-step window</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  PAGE 1 — FORECAST
# ═══════════════════════════════════════════════════════
if page == "📈 Forecast":
    model, scaler, feature_count = load_all()

    st.markdown("# 📈 Sales Forecast")
    st.markdown("Enter the **last 30 daily sales values** to generate a next-day prediction using the trained LSTM model.")

    st.divider()

    col_input, col_info = st.columns([2, 1])

    with col_input:
        st.markdown('<div class="section-title">Input Window</div>', unsafe_allow_html=True)
        vals_input = st.text_area(
            "30 comma-separated sales values",
            placeholder="e.g. 120, 134, 98, 145, …",
            height=120,
        )
        predict_btn = st.button("⚡ Generate Forecast", use_container_width=True)

    with col_info:
        st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
        The LSTM model accepts a <strong>30-day rolling window</strong>
        of sales figures and predicts the next period's demand.
        </div>
        """, unsafe_allow_html=True)

    if predict_btn:
        try:
            v = [float(x.strip()) for x in vals_input.split(",")]

            if len(v) != 30:
                st.error(f"❌ Expected exactly 30 values — got {len(v)}.")
            else:
                with st.spinner("Running inference…"):

                    # FIX 1: Correct input shape (ONLY sales feature)
                    dummy = np.zeros((1, 30, feature_count))
                    sales_index = 0  #  change if sales is not first feature
                    dummy[0, :, sales_index] = v

                    # Predict
                    pred_scaled = model.predict(dummy, verbose=0)

                    # DEBUG (remove later)
                    st.write("Scaled prediction:", pred_scaled)

                    # ✅ FIX 2: Correct inverse scaling
                    inv_dummy = np.zeros((1, feature_count))
                    inv_dummy[0, sales_index] = pred_scaled[0, 0]

                    pred_value = scaler.inverse_transform(inv_dummy)[0, sales_index]

                # Metrics
                avg_input = float(np.mean(v))
                trend_pct = (pred_value - avg_input) / (avg_input + 1e-8) * 100

                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Sales", f"{pred_value:,.2f}")
                c2.metric("Input Window Avg", f"{avg_input:,.2f}")
                c3.metric("Trend vs Avg", f"{trend_pct:+.1f}%")

                # Chart
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=list(range(30)),
                    y=v,
                    mode="lines",
                    name="Input",
                    line=dict(width=2)
                ))

                fig.add_trace(go.Scatter(
                    x=[29, 30],
                    y=[v[-1], pred_value],
                    mode="lines+markers",
                    name="Forecast"
                ))

                fig.update_layout(
                    **CHART_TEMPLATE["layout"],
                    title="Forecast",
                    height=320
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════
#  PAGE 2 — BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════
elif page == "📊 Business Insights":
    df_raw = load_sales()

    st.markdown("# 📊 Business Insights")

    # Aggregate to daily
    df_agg = (
        df_raw.groupby("date")["unit_sales"]
        .sum()
        .reset_index()
        .rename(columns={"unit_sales": "sales"})
        .sort_values("date")
    )
    df_agg["ma7"]    = moving_average(df_agg["sales"], 7)
    df_agg["ma30"]   = moving_average(df_agg["sales"], 30)
    df_agg["growth"] = growth_rate(df_agg["sales"])
    df_agg["anomaly"]= detect_anomalies(df_agg["sales"])

    total   = df_agg["sales"].sum()
    avg_day = df_agg["sales"].mean()
    max_day = df_agg["sales"].max()
    yoy_growth = df_agg["growth"].median()
    anomaly_count = df_agg["anomaly"].sum()

    # ── KPIs ──
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Sales",       f"{total:,.0f}")
    k2.metric("Avg Daily Sales",   f"{avg_day:,.0f}")
    k3.metric("Peak Sales Day",    f"{max_day:,.0f}")
    k4.metric("Median Daily Growth", f"{yoy_growth:+.2f}%")
    k5.metric("Anomaly Days",      f"{anomaly_count}")

    st.divider()

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend", "📅 Seasonality", "🔍 Anomalies", "📊 Distribution"])

    with tab1:
        fig = go.Figure()
        anomalies = df_agg[df_agg["anomaly"]]

        fig.add_trace(go.Scatter(
            x=df_agg["date"], y=df_agg["sales"],
            mode="lines", name="Daily Sales",
            line=dict(color="#3498DB", width=1.2),
            fill="tozeroy", fillcolor="rgba(52,152,219,0.05)",
        ))
        fig.add_trace(go.Scatter(
            x=df_agg["date"], y=df_agg["ma7"],
            mode="lines", name="7-Day MA",
            line=dict(color="#2ECC71", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df_agg["date"], y=df_agg["ma30"],
            mode="lines", name="30-Day MA",
            line=dict(color="#F39C12", width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=anomalies["date"], y=anomalies["sales"],
            mode="markers", name="Anomaly",
            marker=dict(color="#E74C3C", size=8, symbol="x"),
        ))
        fig.update_layout(**CHART_TEMPLATE["layout"], title="Daily Sales Trend with Moving Averages", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Growth rate chart
        fig2 = go.Figure(go.Bar(
            x=df_agg["date"], y=df_agg["growth"].clip(-100, 200),
            marker_color=np.where(df_agg["growth"] >= 0, "#2ECC71", "#E74C3C"),
            name="Daily Growth %",
        ))
        fig2.update_layout(**CHART_TEMPLATE["layout"], title="Day-over-Day Growth Rate (%)", height=250)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        df_agg["month"]      = df_agg["date"].dt.month
        df_agg["day_name"]   = df_agg["date"].dt.day_name()
        df_agg["quarter"]    = df_agg["date"].dt.quarter

        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        day_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

        monthly = (df_agg.groupby("month")["sales"].mean()
                   .reset_index().rename(columns={"sales": "avg_sales"}))
        weekly  = (df_agg.groupby("day_name")["sales"].mean()
                   .reset_index().rename(columns={"sales": "avg_sales"}))
        weekly["day_name"] = pd.Categorical(weekly["day_name"], categories=day_order, ordered=True)
        weekly = weekly.sort_values("day_name")

        col1, col2 = st.columns(2)
        with col1:
            fig_m = px.bar(monthly, x="month", y="avg_sales",
                           title="Average Sales by Month",
                           color="avg_sales", color_continuous_scale="Greens")
            fig_m.update_layout(**CHART_TEMPLATE["layout"])
            st.plotly_chart(fig_m, use_container_width=True)

        with col2:
            fig_w = px.bar(weekly, x="day_name", y="avg_sales",
                           title="Average Sales by Day of Week",
                           color="avg_sales", color_continuous_scale="Blues")
            fig_w.update_layout(**CHART_TEMPLATE["layout"])
            st.plotly_chart(fig_w, use_container_width=True)

        # Heatmap: month × weekday
        df_agg["weekday_num"] = df_agg["date"].dt.dayofweek
        pivot = df_agg.pivot_table(values="sales", index="month",
                                   columns="weekday_num", aggfunc="mean")
        pivot = pivot.reindex(columns=range(7), fill_value=0)
        pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        fig_heat = px.imshow(pivot, title="Sales Heatmap: Month × Weekday",
                             color_continuous_scale="Greens", aspect="auto")
        fig_heat.update_layout(**CHART_TEMPLATE["layout"])
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.markdown("Anomaly days are defined as dates where the absolute z-score > 3.0 standard deviations from the mean.")
        if anomaly_count > 0:
            anom_df = df_agg[df_agg["anomaly"]][["date","sales","ma7"]].copy()
            anom_df["deviation"] = anom_df["sales"] - anom_df["ma7"]
            anom_df = anom_df.reset_index(drop=True)
            st.dataframe(
                anom_df.style.format({"sales": "{:,.0f}", "ma7": "{:,.0f}", "deviation": "{:+,.0f}"}),
                use_container_width=True,
            )
        else:
            st.success("✅ No anomalies detected in the dataset.")

    with tab4:
        fig_hist = px.histogram(df_agg, x="sales", nbins=60,
                                title="Sales Distribution",
                                color_discrete_sequence=["#2ECC71"])
        fig_hist.update_layout(**CHART_TEMPLATE["layout"])
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_box = px.box(df_agg, x="month", y="sales",
                         title="Sales Distribution by Month",
                         color_discrete_sequence=["#3498DB"])
        fig_box.update_layout(**CHART_TEMPLATE["layout"])
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>💡 Key Takeaways</strong><br>
    • Moving averages smooth daily noise — the 30-day MA reveals macro momentum.<br>
    • Seasonal heatmaps expose the best days and months to run promotions.<br>
    • Anomaly flagging pinpoints stockouts, promotions, or data quality issues.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  PAGE 3 — WALMART ANALYSIS
# ═══════════════════════════════════════════════════════
elif page == "🏪 Walmart Analysis":
    df = load_walmart()

    st.markdown("# 🏪 Walmart Store Analysis")

    # Validate expected columns
    required = {"Store", "Weekly_Sales", "Temperature", "Fuel_Price",
                 "CPI", "Unemployment", "Holiday_Flag"}
    missing = required - set(df.columns)
    if missing:
        st.warning(f"Dataset missing columns: {missing}. Some charts may be unavailable.")

    # KPIs
    st.markdown('<div class="section-title">Portfolio Overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Stores",          f"{df['Store'].nunique()}" if "Store" in df.columns else "N/A")
    k2.metric("Total Weekly Sales",    f"${df['Weekly_Sales'].sum():,.0f}" if "Weekly_Sales" in df.columns else "N/A")
    k3.metric("Avg Weekly Sales / Store", f"${df['Weekly_Sales'].mean():,.0f}" if "Weekly_Sales" in df.columns else "N/A")
    k4.metric("Holiday Weeks",         f"{df['Holiday_Flag'].sum()}" if "Holiday_Flag" in df.columns else "N/A")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["🏬 Store Rankings", "🌡️ Macro Impact", "🎉 Holiday Effect", "🔗 Correlations"])

    with tab1:
        store_perf = walmart_store_performance(df)
        if not store_perf.empty:
            top_n = st.slider("Top N stores", 5, 45, 20)
            top_stores = store_perf.head(top_n)
            fig = px.bar(top_stores, x="Store", y="total_sales",
                         color="avg_weekly", color_continuous_scale="Greens",
                         title=f"Top {top_n} Stores by Total Sales")
            fig.update_layout(**CHART_TEMPLATE["layout"])
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                store_perf.head(top_n)
                .style.format({"total_sales": "${:,.0f}", "avg_weekly": "${:,.0f}"}),
                use_container_width=True,
            )

    with tab2:
        macro_cols = [c for c in ["Temperature","Fuel_Price","CPI","Unemployment"] if c in df.columns]
        if macro_cols and "Weekly_Sales" in df.columns:
            selected_macro = st.selectbox("Macro variable", macro_cols)
            fig = px.scatter(
                df, x=selected_macro, y="Weekly_Sales",
                color="Holiday_Flag" if "Holiday_Flag" in df.columns else None,
                trendline="ols",
                title=f"Weekly Sales vs {selected_macro}",
                opacity=0.6,
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(**CHART_TEMPLATE["layout"])
            st.plotly_chart(fig, use_container_width=True)

            corr = walmart_macro_correlations(df)
            if not corr.empty:
                fig_corr = go.Figure(go.Bar(
                    x=corr.values,
                    y=corr.index,
                    orientation="h",
                    marker_color=["#E74C3C" if v < 0 else "#2ECC71" for v in corr.values],
                ))
                fig_corr.update_layout(**CHART_TEMPLATE["layout"],
                                       title="Correlation with Weekly Sales", height=300)
                st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        holiday_df = walmart_insights(df)
        if not holiday_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig_h = px.bar(holiday_df, x="Holiday", y="mean",
                               error_y="std",
                               color="Holiday",
                               color_discrete_map={"Regular": "#3498DB", "Holiday": "#F39C12"},
                               title="Avg Weekly Sales: Holiday vs Regular")
                fig_h.update_layout(**CHART_TEMPLATE["layout"])
                st.plotly_chart(fig_h, use_container_width=True)
            with col2:
                # Percentage uplift
                h_vals = holiday_df.set_index("Holiday")["mean"]
                if "Holiday" in h_vals.index and "Regular" in h_vals.index:
                    uplift = (h_vals["Holiday"] - h_vals["Regular"]) / h_vals["Regular"] * 100
                    st.markdown(f"""
                    <div class="insight-box" style="margin-top:40px">
                    <strong>Holiday Sales Uplift</strong><br><br>
                    <span style="font-size:36px;color:#F39C12;font-family:'DM Mono'">{uplift:+.1f}%</span><br>
                    compared to regular weeks.
                    </div>
                    """, unsafe_allow_html=True)

    with tab4:
        num_df = df.select_dtypes(include=np.number)
        corr_matrix = num_df.corr()
        fig_hm = px.imshow(corr_matrix, title="Feature Correlation Matrix",
                           color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                           aspect="auto", text_auto=".2f")
        fig_hm.update_layout(**CHART_TEMPLATE["layout"])
        st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>💡 Key Takeaways</strong><br>
    • Top stores contribute disproportionate revenue — focus inventory efforts there.<br>
    • Holiday weeks drive measurable uplift; plan staffing and stock accordingly.<br>
    • CPI and Unemployment show inverse correlation with sales (economic sensitivity).
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  PAGE 4 — PRICING INSIGHTS
# ═══════════════════════════════════════════════════════
elif page == "💰 Pricing Insights":
    df = load_supermarket()

    st.markdown("# 💰 Pricing & Profitability Insights")

    profit_col = "gross income" if "gross income" in df.columns else "gross_income"

    # KPIs
    st.markdown('<div class="section-title">Supermarket Overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Transactions",    f"{len(df):,}")
    k2.metric("Avg Unit Price",        f"${df['Unit price'].mean():.2f}" if "Unit price" in df.columns else "N/A")
    k3.metric("Avg Gross Income",      f"${df[profit_col].mean():.2f}" if profit_col in df.columns else "N/A")
    k4.metric("Price Elasticity",      f"{price_elasticity(df):.3f}")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["💲 Price vs Profit", "📦 Product Lines", "👥 Customers", "🏷️ Elasticity"])

    with tab1:
        if "Unit price" in df.columns and profit_col in df.columns:
            color_col = "Product line" if "Product line" in df.columns else None
    
            fig = px.scatter(
                df,
                x="Unit price",
                y=profit_col,
                color=color_col,
                trendline="ols",
                title="Unit Price vs Gross Income",
                opacity=0.65,
                size="Quantity" if "Quantity" in df.columns else None,
            )

        fig.update_layout(**CHART_TEMPLATE["layout"])
        st.plotly_chart(fig, use_container_width=True)

        if "Payment" in df.columns:
            payment_split = df["Payment"].value_counts().reset_index()
            payment_split.columns = ["Method", "Count"]
            fig_pay = px.pie(payment_split, names="Method", values="Count",
                             title="Payment Method Breakdown",
                             color_discrete_sequence=["#2ECC71","#3498DB","#F39C12"])
            fig_pay.update_layout(**CHART_TEMPLATE["layout"])
            st.plotly_chart(fig_pay, use_container_width=True)

    with tab2:
        margin_df = profit_margin_analysis(df)
        if not margin_df.empty:
            fig_pl = px.bar(
                margin_df,
                x="product_line",   
                y="total_revenue",
                color="avg_profit",
                color_continuous_scale="Greens",
                title="Total Revenue by Product Line (coloured by avg profit)"
            )
            fig_pl.update_layout(**CHART_TEMPLATE["layout"])
            st.plotly_chart(fig_pl, use_container_width=True)

            fig_scatter = px.scatter(
                margin_df, x="avg_price", y="avg_quantity",
                size="total_revenue", color="avg_profit",
                text="product_line",
                title="Price vs Quantity Bubble Chart",
                color_continuous_scale="Viridis",
            )
            fig_scatter.update_traces(textposition="top center")
            fig_scatter.update_layout(**CHART_TEMPLATE["layout"])
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.dataframe(
                margin_df.style.format({
                    "avg_price": "${:.2f}", "avg_profit": "${:.2f}",
                    "total_revenue": "${:,.2f}", "avg_quantity": "{:.1f}",
                }),
                use_container_width=True,
            )

    with tab3:
        seg = customer_segmentation(df)
        rating = customer_rating_analysis(df)

        if not seg.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig_seg = px.bar(seg, x="Customer type", y="total_spend",
                                 color="Customer type",
                                 title="Total Spend by Customer Type",
                                 color_discrete_sequence=["#2ECC71","#3498DB"])
                fig_seg.update_layout(**CHART_TEMPLATE["layout"])
                st.plotly_chart(fig_seg, use_container_width=True)
            with col2:
                fig_tx = px.bar(seg, x="Customer type", y="transactions",
                                color="Customer type",
                                title="Transaction Count by Customer Type",
                                color_discrete_sequence=["#F39C12","#E74C3C"])
                fig_tx.update_layout(**CHART_TEMPLATE["layout"])
                st.plotly_chart(fig_tx, use_container_width=True)

        if not rating.empty:
            fig_rat = px.bar(rating, x=rating.columns[0], y="avg_rating",
                             color=rating.columns[1] if len(rating.columns) > 3 else None,
                             barmode="group",
                             error_y="std_rating",
                             title="Average Customer Rating",
                             color_discrete_sequence=["#2ECC71","#9B59B6"])
            fig_rat.update_layout(**CHART_TEMPLATE["layout"])
            st.plotly_chart(fig_rat, use_container_width=True)

    with tab4:
        elas_by_cat = price_elasticity_by_category(df)
        if not elas_by_cat.empty:
            elas_df = elas_by_cat.reset_index()
            elas_df.columns = ["product_line", "elasticity"]
            elas_df = elas_df.sort_values("elasticity")
            fig_el = go.Figure(go.Bar(
                x=elas_df["elasticity"],
                y=elas_df["product_line"],
                orientation="h",
                marker_color=["#E74C3C" if v < 0 else "#2ECC71" for v in elas_df["Elasticity"]],
            ))
            fig_el.update_layout(**CHART_TEMPLATE["layout"],
                                 title="Price Elasticity by Product Line",
                                 xaxis_title="Pearson Correlation (Price ↔ Quantity)",
                                 height=350)
            st.plotly_chart(fig_el, use_container_width=True)

        st.markdown(f"""
        <div class="insight-box">
        <strong>📐 Overall Price Elasticity: {price_elasticity(df):.3f}</strong><br><br>
        A value near <strong>0</strong> means price changes have little effect on quantity sold.<br>
        A value near <strong>–1</strong> indicates highly elastic (price-sensitive) demand.<br>
        A positive value can indicate a Veblen effect (prestige pricing) or data aggregation artifacts.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <strong>💡 Key Takeaways</strong><br>
    • Product lines with high average profit but low volume are candidates for promotional push.<br>
    • Member customers tend to spend more per transaction — loyalty programs pay off.<br>
    • Elastic product lines are risky to reprice; inelastic ones offer margin expansion room.
    </div>
    """, unsafe_allow_html=True)

