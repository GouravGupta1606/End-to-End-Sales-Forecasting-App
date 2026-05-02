[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# End-to-End-Sales-Forecasting-App
End-to-end machine learning-based sales forecasting system with data analysis and an interactive Streamlit dashboard for predicting future sales and generating business insights
A Streamlit-powered retail analytics and sales forecasting dashboard built on three public datasets, using a trained **Stacked LSTM** model to predict future sales.




---

## 📸 Dashboard Preview

| Page | Description |
|------|-------------|
| 📈 Forecast | Next-day prediction using a 30-step LSTM sequence |
| 📊 Business Insights | Trend analysis, seasonality heatmap, anomaly detection |
| 🏪 Walmart Analysis | Store rankings, macro correlations, holiday uplift |
| 💰 Pricing Insights | Price elasticity, margin analysis, customer segmentation |

---

## 📁 Project Structure

```
EcoYield-Sales-Forecasting/
│
├── app/
│   └── app.py                  # Streamlit dashboard entry point
│
├── data/
│   ├── raw/
│   │   ├── train.csv           # Favorita grocery sales (Kaggle)
│   │   ├── walmart.csv         # Walmart weekly sales (Kaggle)
│   │   └── supermarket.csv     # Supermarket transactions (Kaggle)
│   └── processed/
│       └── processed_data.csv  # Generated after preprocessing
│
├── model/
│   ├── best_model.keras        # Best checkpoint (Keras format)
│   ├── lstm_model.h5           # Legacy HDF5 format
│   ├── lstm_model.keras        # Keras 3 format
│   ├── scaler.pkl              # Fitted MinMaxScaler
│   ├── feature_names.pkl       # Feature column names
│   ├── feature_count.pkl       # Feature count for inverse transform
│   ├── cv_results.pkl          # Cross-validation results
│   └── training_summary.pkl    # Training history summary
│
├── notebooks/
│   └── model_training.ipynb    # Full training pipeline
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        # Data loading, feature engineering, scaling
│   ├── modeling.py             # LSTM architectures, callbacks, inference
│   ├── analytics.py            # Metrics, pricing, customer, Walmart analytics
│   └── visualization.py        # Matplotlib/Seaborn plotting utilities
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

## 📊 Datasets

### 1. Corporación Favorita Grocery Sales
**Source:** [Kaggle Competition](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)

Large-scale Ecuadorian grocery chain — primary training source for the LSTM.

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Date of sale |
| `unit_sales` | float | Units sold (negative = returns, clipped to 0) |
| `onpromotion` | bool | Item on promotion (~16% NaN → filled 0) |

> Spans **Jan 2013 – Aug 2017**. Notable: earthquake spike (Apr 16 2016), bimonthly wage-day bumps.

---

### 2. Walmart Store Sales
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset/data)

Weekly sales across 45 stores with macroeconomic context.

| Column | Type | Description |
|--------|------|-------------|
| `Store` | int | Store number (1–45) |
| `Weekly_Sales` | float | Sales for that store/week (USD) |
| `Holiday_Flag` | int | 1 = holiday week |
| `Temperature` / `Fuel_Price` / `CPI` / `Unemployment` | float | Macro indicators |

---

### 3. Supermarket Sales
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/faresashraf1001/supermarket-sales/data)

1,000 transactions across 3 branches (Jan–Mar 2019) — used for pricing and customer analytics.

| Column | Type | Description |
|--------|------|-------------|
| `Product line` | str | 6 product categories |
| `Unit price` / `Quantity` / `Total` | float | Transaction values |
| `gross income` | float | Profit per transaction |
| `Rating` | float | Customer satisfaction (1–10) |

---

## 🧠 Model Architecture

```
Input → (30 time steps × 9 features)
    ↓
LSTM(128, return_sequences=True) + BatchNorm + Dropout(0.2)
    ↓
LSTM(64,  return_sequences=True) + BatchNorm + Dropout(0.2)
    ↓
LSTM(32) + BatchNorm + Dropout(0.2)
    ↓
Dense(32, ReLU) → Dense(1)   ← predicted unit_sales (scaled)
```

### Engineered Features (9 total)

| Feature | Description |
|---------|-------------|
| `sales` | Target — daily aggregated unit sales |
| `promo` | Daily promotion count |
| `lag_1` / `lag_7` / `lag_14` | Lagged sales |
| `rolling_mean_7` / `rolling_std_7` | Rolling statistics |
| `day_of_week` | 0 (Mon) – 6 (Sun) |
| `month` | 1–12 |

### Training Config

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 1e-3) |
| Loss | MSE |
| Batch size | 64 |
| Max epochs | 10 (EarlyStopping patience = 3) |
| Train / Val split | 80 / 20 (chronological) |
| Sequence length | 30 days |
| Sample size | 200,000 rows |

### Results

| Metric | Value |
|--------|-------|
| RMSE | **20.36** |
| MAE | **9.44** |
| MAPE | ~241% *(see note)* |

> ⚠️ High MAPE is expected — item-store level data contains many near-zero sales values, inflating percentage errors. **RMSE and MAE are the reliable metrics here.**

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/EcoYield-Sales-Forecasting.git
cd EcoYield-Sales-Forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download datasets
Place the raw CSV files in `data/raw/`:
- [`train.csv`](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data) → Favorita
- [`walmart.csv`](https://www.kaggle.com/datasets/yasserh/walmart-dataset/data) → Walmart
- [`supermarket.csv`](https://www.kaggle.com/datasets/faresashraf1001/supermarket-sales/data) → Supermarket

### 4. Configure path in `app/app.py`
```python
BASE = "/path/to/EcoYield-Sales-Forecasting"   # ← update this
```

### 5. Run the app
```bash
streamlit run app/app.py
```

---

## 🔧 Known Issues & Roadmap

- [ ] Replace hardcoded `BASE` path with `.env` / `config.yaml`
- [ ] Migrate `lstm_model.h5` fully to `.keras` format (Keras 3 compatibility)
- [ ] Regenerate `feature_count.pkl` if features are re-engineered
- [ ] Add confidence intervals to forecast output
- [ ] Expand supermarket dataset for more reliable elasticity estimates

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Gourav** , **Shlok** ,**Aryan**
--
Feel free to connect on [LinkedIn](https://linkedin.com) or raise an issue for questions!
