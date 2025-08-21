# src/dashboard_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# local imports
from export import export_forecasts
from preprocess import preprocess_pipeline
from train import run_training   

# -------------------------------
# Helper functions for visualization
# -------------------------------
def plot_monthly_trend(monthly):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["MonthPeriod"], monthly["Sales"], marker="o", linestyle="-", color="tab:blue")
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    ax.grid(True)
    st.pyplot(fig)


def plot_forecast(forecast):
    st.subheader("🔮 Sales Forecast")
    
    # forecast column auto detect
    if "yhat" in forecast.columns:
        ycol = "yhat"
    elif "Forecast" in forecast.columns:
        ycol = "Forecast"
    else:
        st.error("❌ No forecast column found in data")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(forecast["Date"], forecast[ycol], label="Forecast", color="blue")
    
    # confidence interval detect
    lower_col, upper_col = None, None
    for cand_l, cand_u in [("yhat_lower","yhat_upper"), ("Lower","Upper")]:
        if cand_l in forecast.columns and cand_u in forecast.columns:
            lower_col, upper_col = cand_l, cand_u
            break

    if lower_col and upper_col:
        ax.fill_between(forecast["Date"], forecast[lower_col], forecast[upper_col],
                        color="lightblue", alpha=0.3, label="Confidence Interval")
    
    ax.set_title("Forecast Horizon")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)



def plot_actual_vs_forecast(actual_vs_forecast):
    st.subheader("⚖️ Actual vs Forecast (Validation Period)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual_vs_forecast["Date"], actual_vs_forecast["y"], label="Actual", color="black")
    if "yhat" in actual_vs_forecast.columns:
        ax.plot(actual_vs_forecast["Date"], actual_vs_forecast["yhat"], label="Forecast", color="blue")
    if "yhat_lower" in actual_vs_forecast.columns and "yhat_upper" in actual_vs_forecast.columns:
        ax.fill_between(
            actual_vs_forecast["Date"],
            actual_vs_forecast["yhat_lower"],
            actual_vs_forecast["yhat_upper"],
            color="lightblue", alpha=0.3, label="Confidence Interval"
        )
    ax.set_title("Actual vs Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


# -------------------------------
# Load Data with caching
# -------------------------------
@st.cache_data
def load_data(selected_model, horizon):
    # run preprocessing pipeline
    daily, monthly, features = preprocess_pipeline()

    # run training dynamically
    forecast, actual_vs_forecast, metrics = run_training(model=selected_model, horizon=horizon)

    return daily, monthly, forecast, actual_vs_forecast, metrics


# -------------------------------
# Main App
# -------------------------------
def main():
    st.title("🤖 AI-Powered Sales Forecasting Dashboard")

    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    model_choice = st.sidebar.selectbox("Select Model", ["Prophet", "SARIMAX"])
    horizon = st.sidebar.slider("Forecast Horizon (days)", 30, 365, 90)

    # Load processed + forecasted data
    daily, monthly, forecast, actual_vs_forecast, metrics = load_data(model_choice, horizon)

    # Metrics Cards
    st.subheader("📊 Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
    col2.metric("MAE", f"{metrics['MAE']:.2f}")
    col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")

    # Tabs
    tabs = st.tabs(["Daily Trend", "Monthly Trend", "Forecast", "Actual vs Forecast", "Download Data"])

    with tabs[0]:
        st.subheader("📈 Daily Sales Trend")
        st.line_chart(daily.set_index("Date")["Sales"])

    with tabs[1]:
        plot_monthly_trend(monthly)

    with tabs[2]:
        plot_forecast(forecast)

    with tabs[3]:
        plot_actual_vs_forecast(actual_vs_forecast)

    with tabs[4]:
        st.subheader("⬇️ Download Data")
        # Forecast download
        csv1 = forecast.to_csv(index=False).encode("utf-8")
        st.download_button("Download Forecast CSV", csv1, "forecast.csv", "text/csv")

        # Actual vs Forecast download
        csv2 = actual_vs_forecast.to_csv(index=False).encode("utf-8")
        st.download_button("Download Actual vs Forecast CSV", csv2, "actual_vs_forecast.csv", "text/csv")


if __name__ == "__main__":
    main()
