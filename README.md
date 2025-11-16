# stockAnalysis_timeseries

“End-to-end stock forecasting system integrating financial news sentiment with time-series modeling. Includes news cleaning, VADER sentiment scoring, stock–sentiment merging, ARIMAX forecasting, Power BI exports, and an interactive Streamlit dashboard for analysis and prediction.”

# Stock Price Prediction Using News Sentiment (AAL)
A Time-Series Forecasting Project Integrating Financial News + ARIMAX Modeling + Streamlit Dashboard

This repository presents an end-to-end time-series forecasting system that integrates:

Financial news sentiment

Daily stock price movements

Technical indicators

ARIMAX statistical modeling

Interactive Streamlit dashboard

The project demonstrates how qualitative news sentiment can be combined with quantitative stock price data to improve short-term forecasting.

# Project Highlights
# 1. News Scraping & Cleaning

Raw financial news is processed using Polars for high-speed data manipulation.

HTML links removed, text normalized, duplicates dropped.

Clean daily news stored as AAL_news_clean.csv.

# 2. Sentiment Scoring (VADER)

News text scored using VADER sentiment analysis.

Sentiment labels: positive (1), neutral (0), negative (-1).

Output: AAL_news_sentiment.csv.

# 3. Data Integration

Sentiment scores merged with daily stock prices.

Computed technical indicators:

Moving averages (MA7, MA21)

Returns

Volatility (10-day)

Output: AAL_merged.csv.

# 4. ARIMAX Forecasting Model

Auto-ARIMA automatically selects best (p, d, q).

Exogenous regressors:

sentiment_avg

articles_count

Produces next 30-day forecast:

Saved as AAL_forecast.csv

Model stored at models/arimax_model.pkl

# 5. Power BI Export (Optional)

Exports:

Daily dataset

Monthly aggregation

Forecast outputs
into /powerbi folder for instant visual reporting.

# 6. Streamlit Dashboard

Interactive web app showing:

Daily stock price trends

Moving averages

Sentiment vs stock price

Customizable ARIMAX forecasting

Comparing model vs actual data

# Run:

streamlit run app.py

# Repository Structure
project_tsa/
│
├──# data/
│   ├── AAL_news.csv
│   ├── AAL_news_clean.csv
│   ├── AAL_news_sentiment.csv
│   ├── AAL_stock.csv
│   ├── AAL_merged.csv
│   └── AAL_forecast.csv
│
├──# scripts/
│   ├── step_clean_news.py
│   ├── step_sentiment_vader.py
│   ├── step_merge.py
│   ├── arimax_train.py
│   ├── arimax_forecast.py
│   ├── export_for_powerbi.py
│   └── show_stock_columns.py
│
├──# models/
│   └── arimax_model.pkl
│
├──# streamlit_app/
│   ├── app.py
│   └── utils.py
│
└──# powerbi/
    ├── AAL_daily.csv
    ├── AAL_monthly.csv
    └── AAL_forecast.csv

# Methodology Overview
# 1. Data Cleaning

Polars used for efficient large-scale text cleaning.

Date normalization, duplicate removal, text preprocessing.

# 2. Sentiment Extraction

VADER used due to its suitability for financial short texts.

Produces sentiment polarity per article.

Aggregated daily sentiment used as exogenous regressor.

# 3. Modeling

ARIMAX chosen for interpretability + support for exogenous features.

Auto-ARIMA identifies best order.

Forecasting includes:

Confidence intervals

Future exogenous assumptions

# 4. Visualization

Streamlit + Plotly for interactive UI.

Power BI exports support business-level reporting.

# Why This Project Matters

Combines NLP + Time-Series Modeling.

Demonstrates real-world market sentiment effects.

Shows how traditional econometric models (ARIMAX) still outperform deep learning when data is limited.

Provides a complete pipeline from raw news → forecast.

# Installation
pip install -r requirements.txt

# Running the Pipeline
Clean and process news
python -m scripts.step_clean_news
python -m scripts.step_sentiment_vader
python -m scripts.step_merge

# Train ARIMAX
python -m scripts.arimax_train

# Forecast
python -m scripts.arimax_forecast

# Export for Power BI
python -m scripts.export_for_powerbi

Launch Streamlit App
cd streamlit_app
streamlit run app.py

# Example Outputs

# 30-Day Forecast
AAL_forecast.csv

# Daily Dataset for Power BI
AAL_daily.csv

# Sentiment-Price Relationship Plots
Included inside Streamlit app.

# Future Improvements

Expand to more companies (multi-time-series ARIMAX).

Fine-tune sentiment using FinBERT.

Compare ARIMAX vs LSTM vs TFT models.

Add anomaly detection for news shocks.
