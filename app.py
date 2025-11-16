import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_daily, load_monthly, load_forecast, load_arimax, predict_future

# Paths
DAILY = "data/AAL_daily.csv"
MONTHLY = "data/AAL_monthly.csv"
FORECAST = "data/AAL_forecast.csv"
MODEL = "models/arimax_model.pkl"

# Load data
daily = load_daily(DAILY)
monthly = load_monthly(MONTHLY)
forecast = load_forecast(FORECAST)
model = load_arimax(MODEL)

st.title("📈 AAL Stock Price + News Sentiment + ARIMAX Forecast Dashboard")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "News & Sentiment", "ARIMAX Forecast"])

# -------------------------------------------------------------------
# 1️⃣ MAIN DASHBOARD
# -------------------------------------------------------------------
if page == "Dashboard":
    st.subheader("📊 AAL Daily Stock Price")

    fig = px.line(daily, x="date", y="Close", title="Daily Close Price")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 Moving Averages")
    fig = px.line(daily, x="date", y=["MA_7","MA_21"], title="Moving Averages")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Sentiment Trend")
    fig = px.line(daily, x="date", y="sentiment_avg", title="Daily Sentiment Score")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# 2️⃣ NEWS & SENTIMENT VIEW
# -------------------------------------------------------------------
if page == "News & Sentiment":
    st.subheader("📰 News Sentiment Analysis")

    st.write("Scatter Plot: Sentiment vs Close")
    fig = px.scatter(daily, x="sentiment_avg", y="Close", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Heat Map: Articles Count vs Close")
    fig = px.density_heatmap(daily, x="articles_count", y="Close")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# 3️⃣ ARIMAX FORECAST
# -------------------------------------------------------------------
if page == "ARIMAX Forecast":
    st.subheader("🔮 ARIMAX Forecast (Next 30 Days)")

    # Default forecast table from modeling
    st.write("Model Forecast Output")
    st.dataframe(forecast)

    fig = px.line(forecast, x="date", y="forecast_close", title="ARIMAX Forecast")
    st.plotly_chart(fig, use_container_width=True)

    # User prediction
    st.subheader("Make Your Own Forecast")
    days = st.slider("Forecast Days", 7, 60, 30)

    future = predict_future(model, daily, days)
    st.write(f"📅 Forecast for next {days} days")
    st.dataframe(future)

    fig2 = px.line(future, x="date", y="forecast", title="Custom ARIMAX Forecast")
    st.plotly_chart(fig2, use_container_width=True)
