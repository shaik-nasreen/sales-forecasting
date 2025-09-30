import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

# Load the dataset
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, header=0, parse_dates=['Month'], index_col='Month')
    else:
        # url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
        url = 'bus_travellers.csv'
        data = pd.read_csv(url, header=0, parse_dates=['Month'], index_col='Month')
    return data

# Data Preprocessing function
def preprocess_data(data):
    imputer = SimpleImputer(strategy='mean')
    data['Passengers'] = imputer.fit_transform(data[['Passengers']])
    return data

# Train ARIMA model
def train_arima_model(data, order=(5, 1, 0)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

# Train Prophet model
def train_prophet_model(data):
    df_prophet = data.reset_index().rename(columns={'Month': 'ds', 'Passengers': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    return model

# Forecast using ARIMA
def forecast_arima(model, steps=12):
    forecast = model.forecast(steps=steps)
    return forecast

# Forecast using Prophet
def forecast_prophet(model, periods=12):
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return forecast['yhat']

# Main function for Streamlit app
def main():
    st.title("Retail Sales Forecasting using ARIMA and Prophet")
    st.write("""
    This application demonstrates the use of **ARIMA** and **Prophet** models for time series forecasting of retail sales data.
    We will load historical sales data, preprocess it, train the models, and visualize the forecasts.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    # Load and preprocess data
    data = load_data(uploaded_file)
    st.subheader("Historical Sales Data")
    st.write(data.head())

    # Preprocess data
    data = preprocess_data(data)

    # Train ARIMA Model
    st.subheader("Training ARIMA Model")
    arima_model = train_arima_model(data['Passengers'])

    # Forecast using ARIMA
    arima_forecast = forecast_arima(arima_model)

    # Train Prophet Model
    st.subheader("Training Prophet Model")
    prophet_model = train_prophet_model(data)

    # Forecast using Prophet
    prophet_forecast = forecast_prophet(prophet_model)

    # Plot the ARIMA Forecast
    st.subheader("ARIMA Forecast")
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Passengers'], label='Historical Sales', color='blue')
    plt.plot(pd.date_range(data.index[-1], periods=13, freq='M')[1:], arima_forecast, label='ARIMA Forecast', color='red')
    plt.title('ARIMA Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    st.pyplot(plt)

    # Plot the Prophet Forecast
    st.subheader("Prophet Forecast")
    future_dates = pd.date_range(data.index[-1], periods=13, freq='M')[1:]
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Passengers'], label='Historical Sales', color='blue')
    plt.plot(future_dates, prophet_forecast[-12:], label='Prophet Forecast', color='green')
    plt.title('Prophet Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    st.pyplot(plt)

    # Model Evaluation - RMSE & MAE (on ARIMA for simplicity)
    st.subheader("Model Evaluation (ARIMA)")
    arima_rmse = np.sqrt(mean_squared_error(data['Passengers'][-12:], arima_forecast))
    arima_mae = mean_absolute_error(data['Passengers'][-12:], arima_forecast)
    st.write(f"ARIMA Model RMSE: {arima_rmse:.2f}")
    st.write(f"ARIMA Model MAE: {arima_mae:.2f}")

    # Evaluation for Prophet - RMSE & MAE (on Prophet forecast for simplicity)
    st.subheader("Model Evaluation (Prophet)")
    prophet_rmse = np.sqrt(mean_squared_error(data['Passengers'][-12:], prophet_forecast[-12:]))
    prophet_mae = mean_absolute_error(data['Passengers'][-12:], prophet_forecast[-12:])
    st.write(f"Prophet Model RMSE: {prophet_rmse:.2f}")
    st.write(f"Prophet Model MAE: {prophet_mae:.2f}")

    # Compare the two models
    st.subheader("Model Comparison")
    st.write(f"ARIMA RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}")
    st.write(f"Prophet RMSE: {prophet_rmse:.2f}, MAE: {prophet_mae:.2f}")

if __name__ == "__main__":
    main()
