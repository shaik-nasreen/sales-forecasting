import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        url = 'marketing_sales.csv'
        data = pd.read_csv(url, header=0)
    
    # Strip any extra spaces in column names
    data.columns = data.columns.str.strip()
    return data

# Data Preprocessing function
def preprocess_data(data):
    if 'Sales' not in data.columns:
        st.error("Error: 'Sales' column not found in dataset! Please check your CSV file.")
        return None

    # Convert 'Date' column to datetime if it exists
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data['Sales'] = imputer.fit_transform(data[['Sales']])
    
    # Normalize Sales using Min-Max Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Sales'] = scaler.fit_transform(data[['Sales']])
    
    return data

# Train ARIMA model
def train_arima_model(data, order=(5, 1, 0)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

# Train Prophet model
def train_prophet_model(data):
    df_prophet = data.reset_index().rename(columns={'Date': 'ds', 'Sales': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df_prophet)
    return model

# Forecast using ARIMA
def forecast_arima(model, steps=12):
    return model.forecast(steps=steps)

# Forecast using Prophet
def forecast_prophet(model, periods=12):
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(periods)

# Streamlit App
def main():
    st.title("Retail Sales Forecasting using ARIMA and Prophet")
    st.write("""
    This application uses **ARIMA** and **Prophet** models for time series forecasting of retail sales data.
    Upload your CSV file and visualize the forecasts!
    """)

    uploaded_file = st.file_uploader("Upload your sales CSV file", type=["csv"])

    # Load data
    data = load_data(uploaded_file)
    if data is None:
        return
    
    st.subheader("Sample Sales Data")
    st.write(data.head())

    # Preprocess Data
    data = preprocess_data(data)
    if data is None:
        return

    # Train ARIMA Model
    st.subheader("Training ARIMA Model")
    arima_model = train_arima_model(data['Sales'])

    # Forecast using ARIMA
    arima_forecast = forecast_arima(arima_model)

    # Train Prophet Model
    st.subheader("Training Prophet Model")
    prophet_model = train_prophet_model(data)

    # Forecast using Prophet
    prophet_forecast = forecast_prophet(prophet_model, periods=12)
    prophet_values = prophet_forecast['yhat'].values  # Extracting values properly

    # Plot ARIMA Forecast
    st.subheader("ARIMA Forecast")
    fig_arima, ax_arima = plt.subplots(figsize=(10, 6))
    ax_arima.plot(data.index, data['Sales'], label='Historical Sales', color='blue')
    ax_arima.plot(pd.date_range(data.index[-1], periods=13, freq='M')[1:], arima_forecast, label='ARIMA Forecast', color='red')
    ax_arima.set_title('ARIMA Sales Forecast')
    ax_arima.set_xlabel('Date')
    ax_arima.set_ylabel('Normalized Sales (0-1)')
    ax_arima.legend()
    st.pyplot(fig_arima)  # FIXED DEPRECATION WARNING

    # Plot Prophet Forecast
    st.subheader("Prophet Forecast")
    future_dates = prophet_forecast['ds']
    
    fig_prophet, ax_prophet = plt.subplots(figsize=(10, 6))
    ax_prophet.plot(data.index, data['Sales'], label='Historical Sales', color='blue')
    ax_prophet.plot(future_dates, prophet_values, label='Prophet Forecast', color='green')
    ax_prophet.set_title('Prophet Sales Forecast')
    ax_prophet.set_xlabel('Date')
    ax_prophet.set_ylabel('Normalized Sales (0-1)')
    ax_prophet.legend()
    st.pyplot(fig_prophet)  # FIXED DEPRECATION WARNING

    # Model Evaluation
    st.subheader("Model Evaluation (ARIMA)")
    if len(data['Sales']) >= 12:  # Ensure enough data points exist
        arima_rmse = np.sqrt(mean_squared_error(data['Sales'][-12:], arima_forecast))
        arima_mae = mean_absolute_error(data['Sales'][-12:], arima_forecast)
        st.write(f"ARIMA RMSE: {arima_rmse:.2f}")
        st.write(f"ARIMA MAE: {arima_mae:.2f}")
    else:
        st.warning("Not enough historical data to evaluate ARIMA model performance.")

    # Evaluation for Prophet
    st.subheader("Model Evaluation (Prophet)")
    if len(data['Sales']) >= 12:
        prophet_rmse = np.sqrt(mean_squared_error(data['Sales'][-12:], prophet_values))
        prophet_mae = mean_absolute_error(data['Sales'][-12:], prophet_values)
        st.write(f"Prophet RMSE: {prophet_rmse:.2f}")
        st.write(f"Prophet MAE: {prophet_mae:.2f}")
    else:
        st.warning("Not enough historical data to evaluate Prophet model performance.")

    # Model Comparison
    st.subheader("Model Comparison")
    if len(data['Sales']) >= 12:
        st.write(f"ARIMA RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}")
        st.write(f"Prophet RMSE: {prophet_rmse:.2f}, MAE: {prophet_mae:.2f}")

if __name__ == "__main__":
    main()
