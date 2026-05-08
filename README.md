# 📊 Bitcoin Market Sentiment & Trader Performance Analysis
### Primetrade.ai — Data Science Assignment Submission

---

## 🗂️ Project Overview

This project explores the relationship between **Bitcoin market sentiment** (Fear & Greed Index) and **trader performance** on the Hyperliquid decentralised exchange. Using 211,218 real trades from January–December 2024, the analysis uncovers how emotional market cycles influence trader behaviour, profitability, and position sizing — and builds machine learning models to predict market sentiment and trade outcomes.

---

## 📁 Repository Contents

```
📦 project/
├── 📓 bitcoin_sentiment_analysis.ipynb     ← Full Jupyter Notebook (submit this)
├── 🐍 bitcoin_sentiment_ml.py              ← Standalone Python script (same logic)
├── 📄 Bitcoin_Sentiment_Analysis_Report.docx  ← Professional written report
├── 📋 README.md                            ← This file
│
├── 📊 Figures/
│   ├── fig1_sentiment_distribution.png     ← Trade count & % profitable by sentiment
│   ├── fig2_pnl_distribution.png           ← PnL distribution across sentiment
│   ├── fig3_avg_pnl_and_volume.png         ← Avg PnL & trade size by sentiment
│   ├── fig4_buy_sell_ratio.png             ← Buy vs Sell ratio by sentiment
│   ├── fig5_monthly_trend.png              ← Monthly FG Index vs trade volume
│   ├── fig6_model_comparison.png           ← Model 1 accuracy comparison
│   ├── fig7_confusion_matrix.png           ← Random Forest confusion matrix
│   ├── fig8_feature_importance.png         ← Feature importances (Model 1)
│   ├── fig9_model2_comparison.png          ← Model 2 accuracy comparison
│   └── fig10_pnl_by_fg_bucket.png         ← Avg PnL by FG score bucket
│
└── 📂 Data/ (place your CSVs here)
    ├── historical_data.csv                 ← Hyperliquid trader data
    └── fear_greed_index.csv               ← Bitcoin Fear & Greed Index
```

---

## 📋 Datasets

### 1. Historical Trader Data (`historical_data.csv`)
- **Source:** Hyperliquid DEX
- **Period:** January 1, 2024 — December 31, 2024
- **Size:** 211,224 rows × 16 columns
- **Key columns:**

| Column | Description |
|--------|-------------|
| `Account` | Trader wallet address |
| `Coin` | Trading pair symbol |
| `Execution Price` | Price at trade execution |
| `Size USD` | Notional trade value in USD |
| `Side` | BUY or SELL |
| `Timestamp IST` | Trade timestamp (IST timezone) |
| `Closed PnL` | Realised profit/loss on trade close |
| `Fee` | Transaction fee paid |

### 2. Fear & Greed Index (`fear_greed_index.csv`)
- **Source:** Alternative.me Bitcoin Fear & Greed Index
- **Coverage:** 2018–2025 (filtered to 2024 for this analysis)
- **Key columns:**

| Column | Description |
|--------|-------------|
| `date` | Calendar date |
| `value` | Numeric score (0–100) |
| `classification` | Extreme Fear / Fear / Neutral / Greed / Extreme Greed |

---

## 🔧 Setup & Installation

### Requirements
- Python 3.8+
- Jupyter Notebook (for `.ipynb`)

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook

```bash
jupyter notebook bitcoin_sentiment_analysis.ipynb
```

### Run the Python Script

```bash
# Place both CSV files in the same directory, then:
python bitcoin_sentiment_ml.py
```

> ✅ No internet connection required. All data is loaded locally from CSVs.

---

## 🔍 Methodology

### Step 1 — Data Loading & Merging
Both datasets are joined on `date`. Each trade row is enriched with the Fear & Greed classification and numeric score for that day. Two derived features are created:
- `profit_flag` → 1 if `Closed PnL > 0`, else 0 (target for Model 2)
- `side_enc` → BUY = 1, SELL = 0 (ML feature)

### Step 2 — Exploratory Data Analysis (EDA)
Five visualisations explore trade frequency, profitability, PnL distribution, buy/sell behaviour, and monthly trends across all five sentiment categories.

### Step 3 — Machine Learning

Two classification problems are solved:

| | Model 1 | Model 2 |
|---|---|---|
| **Task** | Predict market sentiment (5-class) | Predict profitable trade (binary) |
| **Target** | `classification` | `profit_flag` |
| **Features** | Price, Size USD, Fee, Side | Price, Size USD, Fee, Side, FG Value |
| **Algorithms** | LR, Decision Tree, Random Forest | LR, Decision Tree, Random Forest |

---

## 📈 Results

### Model 1 — Market Sentiment Prediction (5-class)

| Model | Accuracy | vs. Baseline (20%) |
|-------|----------|--------------------|
| Logistic Regression | 29.5% | +9.5% |
| Decision Tree | 56.0% | +36.0% |
| **Random Forest** | **58.9%** | **+38.9%** ✅ |

### Model 2 — Profitable Trade Prediction (binary)

| Model | Accuracy | vs. Baseline (50%) |
|-------|----------|--------------------|
| Logistic Regression | 63.0% | +13.0% |
| Decision Tree | 75.6% | +25.6% |
| **Random Forest** | **79.5%** | **+29.5%** ✅ |

---

## 💡 Key Findings

- **Sentiment is the #1 predictor** of trade profitability. The Fear & Greed Index numeric value is the most important feature in the profit prediction model.
- **Extreme Greed = best returns.** Average PnL and win rate are both highest during Greed phases; traders take larger and more profitable positions.
- **Extreme Fear = worst returns.** Traders panic-sell at bottoms, leading to the lowest average PnL across all sentiment categories.
- **Traders follow emotion, not logic.** Buy/sell analysis shows traders buy during greed (high prices) and sell during fear (low prices) — the opposite of "buy low, sell high."
- **Trade volume spikes at extremes.** Both Fear and Greed phases see the most activity, confirming sentiment-driven participation.
- **Random Forest consistently wins.** Ensemble learning handles the non-linear relationships in this dataset significantly better than linear models.

---

## 📊 Visualisations Preview

| Figure | Description |
|--------|-------------|
| Fig 1 | Trade count & % profitable trades by sentiment |
| Fig 2 | PnL distribution histogram by sentiment |
| Fig 3 | Average PnL and trade size by sentiment |
| Fig 4 | Buy vs Sell ratio across sentiment categories |
| Fig 5 | Monthly Fear & Greed Index trend vs trade volume |
| Fig 6 | Model 1 accuracy comparison chart |
| Fig 7 | Confusion matrix — Random Forest (sentiment prediction) |
| Fig 8 | Feature importances — Random Forest |
| Fig 9 | Model 2 accuracy comparison chart |
| Fig 10 | Average Closed PnL by Fear & Greed score bucket |

---

## 🧠 Technologies Used

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading, merging, aggregation |
| `numpy` | Numerical operations |
| `matplotlib` | Chart generation |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | ML models, train/test split, evaluation metrics |

---

## 📄 Report

A full written report (`Bitcoin_Sentiment_Analysis_Report.docx`) is included with:
- Executive Summary
- Dataset Overview with tables
- EDA with all charts and written interpretation
- ML model methodology, results, and confusion matrix
- Model comparison table
- Key insights and strategic recommendations for Primetrade.ai

---

## 👤 Submission

- **Assignment:** Primetrade.ai Data Science Hiring Assessment
- **Contact:** Sonika — Primetrade.ai Hiring Team
- **Submission:** Via Google Form as specified in the assignment PDF

---

*This analysis was conducted purely for assessment purposes. All data is sourced from publicly available Hyperliquid trade records and the Alternative.me Fear & Greed Index.*
