import pandas as pd
import pickle


# -------------------------------------------------------------
# Load ARIMAX model
# -------------------------------------------------------------
def load_arimax(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# -------------------------------------------------------------
# Load CSV helper
# -------------------------------------------------------------
def load_csv(path):
    return pd.read_csv(path)


# -------------------------------------------------------------
# Specific loaders for Streamlit App
# -------------------------------------------------------------
def load_daily(path):
    return pd.read_csv(path)


def load_monthly(path):
    return pd.read_csv(path)


def load_forecast(path):
    return pd.read_csv(path)


# -------------------------------------------------------------
# Predict future using ARIMAX + exogenous inputs
# -------------------------------------------------------------
def predict_future(model, df, days):
    """
    model : SARIMAXResultsWrapper (from auto_arima.fit)
    df    : merged dataframe with sentiment + close
    days  : number of forecasting days
    """

    # Last available date
    last_date = pd.to_datetime(df["date"]).max()

    # Create future date index
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days
    )

    # Exogenous variables = last known sentiment & article count
    exog_future = pd.DataFrame({
        "sentiment_avg": [df["sentiment_avg"].iloc[-1]] * days,
        "articles_count": [df["articles_count"].iloc[-1]] * days
    })

    # Forecast N days
    forecast_obj = model.get_forecast(
        steps=days,
        exog=exog_future
    )

    forecast = forecast_obj.predicted_mean

    # Build final dataframe
    result = pd.DataFrame({
        "date": future_dates,
        "forecast_close": forecast
    })

    return result
