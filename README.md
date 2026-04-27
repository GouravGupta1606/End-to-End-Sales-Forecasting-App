# End-to-End-Sales-Forecasting-App
End-to-end machine learning-based sales forecasting system with data analysis and an interactive Streamlit dashboard for predicting future sales and generating business insights
A Streamlit-powered analytics and forecasting dashboard built on three public retail datasets,
using a trained LSTM model for sales prediction.

---

## 📁 Project Structure

```
EcoYield-Sales-Forecasting/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── walmart.csv
│   │   └── supermarket.csv
│   │
│   └── processed/
│       └── processed_data.csv   # (if you generate)
│
├── model/
│   ├── best_model.keras
│   ├── lstm_model.h5
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── feature_count.pkl
│   ├── cv_results.pkl
│   └── training_summary.pkl
│
├── notebooks/
│   └── model_training.ipynb
│
├── utils/
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── visualization.py
│   ├── analytics.py
│   └── __init__.py
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

## 📊 Datasets

### 1. Corporación Favorita Grocery Sales Forecasting
**Source:** https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data

A large-scale Ecuadorian grocery chain dataset used as the primary training source for the LSTM model.

| File | Description |
|------|-------------|
| `train.csv` | 125M+ rows of daily sales by store, item, date |
| `stores.csv` | Store metadata: city, state, type, cluster |
| `items.csv` | Item metadata: family, class, perishable |
| `transactions.csv` | Daily transaction count per store |
| `oil.csv` | Daily oil price (Ecuador is oil-dependent) |
| `holidays_events.csv` | National/local holidays with type & transferred flag |

**Key columns used in this project (`train.csv`):**

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Date of sale |
| `store_nbr` | int | Store identifier |
| `item_nbr` | int | Item identifier |
| `unit_sales` | float | Units sold (negative = returns; clipped to 0) |
| `onpromotion` | bool/NaN | Whether item was on promotion (~16% NaN → filled with 0) |

**Notes:**
- The dataset spans **January 2013 – August 2017**
- Zero-sales rows are absent (not recorded) — important for aggregation
- A magnitude 7.8 earthquake on **April 16, 2016** caused a visible spike in sales
- Public sector wages paid on the **15th and last day** of each month create bimonthly sales bumps

---

### 2. Walmart Store Sales Dataset
**Source:** https://www.kaggle.com/datasets/yasserh/walmart-dataset/data

Weekly sales data across 45 Walmart stores with macroeconomic context.

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `Store` | int | Store number (1–45) |
| `Date` | date | Week ending date |
| `Weekly_Sales` | float | Sales for that store/week (USD) |
| `Holiday_Flag` | int | 1 = holiday week, 0 = regular |
| `Temperature` | float | Regional temperature (°F) |
| `Fuel_Price` | float | Regional fuel price (USD/gallon) |
| `CPI` | float | Consumer Price Index |
| `Unemployment` | float | Regional unemployment rate (%) |

**Notable holidays covered:**
- Super Bowl, Labour Day, Thanksgiving, Christmas

**Notes:**
- Higher holiday-week sales are expected but vary significantly by store
- CPI and Unemployment show inverse correlation with sales (economic sensitivity)
- No item-level granularity — store-week aggregated only

---

### 3. Supermarket Sales Dataset
**Source:** https://www.kaggle.com/datasets/faresashraf1001/supermarket-sales/data

Transaction-level supermarket data across 3 branches, used for pricing and customer analytics.

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `Invoice ID` | str | Unique transaction ID |
| `Branch` | str | Store branch (A, B, C) |
| `City` | str | City of branch |
| `Customer type` | str | Member / Normal |
| `Gender` | str | Male / Female |
| `Product line` | str | 6 product categories |
| `Unit price` | float | Price per unit (USD) |
| `Quantity` | int | Number of units purchased |
| `Tax 5%` | float | 5% tax amount |
| `Total` | float | Total purchase value (incl. tax) |
| `Date` | date | Date of transaction |
| `Time` | str | Time of transaction |
| `Payment` | str | Cash / Credit card / Ewallet |
| `cogs` | float | Cost of goods sold |
| `gross margin percentage` | float | Fixed at ~4.76% |
| `gross income` | float | Profit per transaction |
| `Rating` | float | Customer satisfaction (1–10) |

**Notes:**
- 3 months of data (Jan–Mar 2019), 1000 transactions total
- All three branches have roughly equal transaction counts
- Useful for price elasticity, margin analysis, and customer segmentation

---

## 🧠 Model

### Architecture: Stacked LSTM

```
Input: (30 time steps × 9 features)
    ↓
LSTM(128, return_sequences=True) + BatchNorm + Dropout(0.2)
    ↓
LSTM(64, return_sequences=True) + BatchNorm + Dropout(0.2)
    ↓
LSTM(32) + BatchNorm + Dropout(0.2)
    ↓
Dense(32, relu)
    ↓
Dense(1)  ← predicted unit_sales (scaled)
```

### Features (9 total after engineering):

| Feature | Description |
|---------|-------------|
| `sales` | Target — daily aggregated unit sales |
| `promo` | Daily promotion count (sum of onpromotion flags) |
| `lag_1` | Sales 1 day ago |
| `lag_7` | Sales 7 days ago |
| `lag_14` | Sales 14 days ago |
| `rolling_mean_7` | 7-day rolling average |
| `rolling_std_7` | 7-day rolling std deviation |
| `day_of_week` | 0 (Monday) – 6 (Sunday) |
| `month` | 1–12 |

### Training Setup:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-3) |
| Loss | MSE |
| Batch size | 64 |
| Max epochs | 10 (EarlyStopping patience=3) |
| Train/Val split | 80/20 (chronological) |
| Sequence length | 30 days |
| Sample size | 200,000 rows from train.csv |

### Results:

| Metric | Value |
|--------|-------|
| RMSE | 20.36 |
| MAE | 9.44 |
| MAPE | ~241% *(inflated by near-zero sales days)* |

> ⚠️ The high MAPE is expected: the raw `train.csv` contains item-store level data where many
> combinations have very low unit_sales (< 1), making percentage errors explode.
> RMSE and MAE are more reliable metrics for this dataset.

---

## 🚀 Running the App

### 1. Install dependencies
```bash
pip install streamlit tensorflow joblib scikit-learn plotly pandas numpy
```

### 2. Update paths in `app/app.py`
```python
BASE = "C:/Users/GOURAV/EcoYield_Project"   # ← change this to your root directory
```

### 3. Run
```bash
streamlit run app/app.py
```

---

## 📈 Dashboard Pages

| Page | Data Source | Key Features |
|------|-------------|--------------|
| 📈 Forecast | Favorita LSTM model | 30-value input → next-day prediction with sparkline |
| 📊 Business Insights | Favorita train.csv | Trend + MA, seasonality heatmap, anomaly detection, distribution |
| 🏪 Walmart Analysis | Walmart.csv | Store rankings, macro correlations, holiday uplift, heatmap |
| 💰 Pricing Insights | SuperMarket Analysis.csv | Price vs profit, product-line margins, customer segmentation, elasticity |

---

## 🔧 Known Issues & Improvements

- **MAPE is unreliable** for this dataset due to near-zero item-level sales. Use RMSE/MAE/R².
- **Hardcoded paths** in `app.py` — use a `.env` file or `config.yaml` for portability.
- **Model saved as `.h5`** (legacy format) — migrate to `.keras` format for Keras 3 compatibility.
- **Feature count mismatch**: if you re-engineer features, regenerate `feature_count.pkl`.
- **Supermarket dataset** is small (1000 rows, 3 months) — elasticity estimates have wide confidence intervals.
