# src/train.py — Model Training & Forecasting for FUTURE_ML_01

import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Prophet (preferred)
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

# SARIMAX (fallback)
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SARIMAX = True
except Exception:
    HAS_SARIMAX = False


# -------------------------------
# Utility functions (metrics)
# -------------------------------
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# -------------------------------
# Main training & forecasting function
# -------------------------------
def train_and_forecast(horizon: int = 90, plot: bool = True):
    """
    Train forecasting model (Prophet or SARIMAX fallback),
    evaluate on validation set, and save forecast results.

    Args:
        horizon (int): Forecast horizon in days.
        plot (bool): Whether to show plots.

    Returns:
        forecast_df (pd.DataFrame): Forecast results with confidence intervals.
        metrics (dict): Evaluation metrics on validation set.
        model_used (str): Model name used ("Prophet" or "SARIMAX").
    """

    # -------------------------------
    # Load cleaned data
    # -------------------------------
    candidates = ["../data/cleaned_sales.csv", "data/cleaned_sales.csv", "./cleaned_sales.csv"]
    cleaned_path = next((p for p in candidates if os.path.exists(p)), None)
    if cleaned_path is None:
        raise FileNotFoundError("cleaned_sales.csv not found. Run preprocess first.")

    df = pd.read_csv(cleaned_path)
    if "Date" not in df.columns or "Sales" not in df.columns:
        raise ValueError("cleaned_sales.csv must have columns: Date, Sales")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Date", "Sales"]).sort_values("Date").reset_index(drop=True)

    # Fill missing dates
    full_idx = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    df_full = (
        df.set_index("Date")
          .reindex(full_idx)
          .rename_axis("Date")
          .reset_index()
    )
    df_full["Sales"] = df_full["Sales"].fillna(0.0)

    # Train-validation split
    if len(df_full) <= horizon + 30:
        raise ValueError(f"Not enough history ({len(df_full)} rows). Reduce horizon={horizon} or add more data.")

    train = df_full.iloc[:-horizon].copy()
    valid = df_full.iloc[-horizon:].copy()

    results = {}
    forecast_df = None
    model_used = None

    # -------------------------------
    # Prophet
    # -------------------------------
    if HAS_PROPHET:
        df_p = train.rename(columns={"Date": "ds", "Sales": "y"})
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.5
        )
        m.fit(df_p)

        future = m.make_future_dataframe(periods=horizon, freq="D", include_history=True)
        fcst = m.predict(future)

        fidx = fcst.set_index("ds")
        fcst_valid = fidx.reindex(valid["Date"]).reset_index()
        fcst_valid = fcst_valid.rename(columns={"index": "Date", "yhat": "Forecast", "yhat_lower": "Lower", "yhat_upper": "Upper"})

        merged = valid.merge(fcst_valid[["Date", "Forecast", "Lower", "Upper"]], on="Date", how="left").dropna(subset=["Forecast"])

        results["RMSE"] = rmse(merged["Sales"], merged["Forecast"])
        results["MAE"] = mae(merged["Sales"], merged["Forecast"])
        results["MAPE"] = mape(merged["Sales"], merged["Forecast"])

        forecast_df = fcst.rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower", "yhat_upper": "Upper"})[
            ["Date", "Forecast", "Lower", "Upper"]
        ]
        model_used = "Prophet"

    # -------------------------------
    # SARIMAX (fallback)
    # -------------------------------
    elif HAS_SARIMAX:
        y_train = train.set_index("Date")["Sales"]
        y_valid = valid.set_index("Date")["Sales"]

        model = SARIMAX(
            y_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)

        pred_valid = res.get_prediction(start=y_valid.index[0], end=y_valid.index[-1], dynamic=False)
        pred_mean = pred_valid.predicted_mean
        pred_ci = pred_valid.conf_int(alpha=0.2)

        merged = pd.DataFrame({
            "Date": y_valid.index,
            "Sales": y_valid.values,
            "Forecast": pred_mean.values,
            "Lower": pred_ci.iloc[:, 0].values,
            "Upper": pred_ci.iloc[:, 1].values
        }).dropna(subset=["Forecast"])

        results["RMSE"] = rmse(merged["Sales"], merged["Forecast"])
        results["MAE"] = mae(merged["Sales"], merged["Forecast"])
        results["MAPE"] = mape(merged["Sales"], merged["Forecast"])

        last_date = df_full["Date"].max()
        fitted = res.get_prediction(start=df_full["Date"].min(), end=last_date, dynamic=False)
        fitted_mean = fitted.predicted_mean
        fitted_ci = fitted.conf_int(alpha=0.2)

        steps = horizon
        future_fore = res.get_forecast(steps=steps)
        future_mean = future_fore.predicted_mean
        future_ci = future_fore.conf_int(alpha=0.2)

        hist_part = pd.DataFrame({
            "Date": fitted_mean.index,
            "Forecast": fitted_mean.values,
            "Lower": fitted_ci.iloc[:, 0].values,
            "Upper": fitted_ci.iloc[:, 1].values
        })

        fut_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq="D")
        fut_part = pd.DataFrame({
            "Date": fut_idx,
            "Forecast": future_mean.values,
            "Lower": future_ci.iloc[:, 0].values,
            "Upper": future_ci.iloc[:, 1].values
        })

        forecast_df = pd.concat([hist_part, fut_part], ignore_index=True)
        model_used = "SARIMAX"

    else:
        raise ImportError("Neither Prophet nor statsmodels is installed.")

    # -------------------------------
    # Save outputs
    # -------------------------------
    data_dir = "../data" if os.path.isdir("../data") else "data"
    os.makedirs(data_dir, exist_ok=True)

    forecast_out = os.path.join(data_dir, "forecast_results.csv")
    forecast_df_out = forecast_df.rename(columns={"Date": "ds", "Forecast": "yhat", "Lower": "yhat_lower", "Upper": "yhat_upper"})
    forecast_df_out.to_csv(forecast_out, index=False)

    actual = df_full.rename(columns={"Date": "ds", "Sales": "y"})[["ds", "y"]]
    combo = actual.merge(forecast_df_out, on="ds", how="left")
    combo_out = os.path.join(data_dir, "actual_vs_forecast.csv")
    combo.to_csv(combo_out, index=False)

    # -------------------------------
    # Plot
    # -------------------------------
    if plot:
        plt.figure(figsize=(14, 5))
        lookback = min(len(df_full), 400)
        plot_actual = df_full.iloc[-lookback:].copy()
        plt.plot(plot_actual["Date"], plot_actual["Sales"], label="Actual Sales")

        f_slice = forecast_df[forecast_df["Date"].between(plot_actual["Date"].min(), plot_actual["Date"].max())]
        plt.plot(f_slice["Date"], f_slice["Forecast"], label=f"Forecast ({model_used})")

        if {"Lower", "Upper"}.issubset(f_slice.columns):
            lower = f_slice["Lower"].astype(float).fillna(method="bfill").fillna(method="ffill")
            upper = f_slice["Upper"].astype(float).fillna(method="bfill").fillna(method="ffill")
            plt.fill_between(f_slice["Date"], lower, upper, alpha=0.2, label="Confidence Band")

        plt.title("Actual vs Forecast — Recent Window")
        plt.xlabel("Date"); plt.ylabel("Sales")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.show()

    return forecast_df, results, model_used
def run_training(model="Prophet", horizon=90):
    """
    Wrapper for dashboard_app.py
    Always calls train_and_forecast and returns consistent format
    """
    forecast_df, results, model_used = train_and_forecast(horizon=horizon, plot=False)
    
    # actual_vs_forecast load karo
    data_dir = "../data" if os.path.isdir("../data") else "data"
    actual_vs_forecast = pd.read_csv(os.path.join(data_dir, "actual_vs_forecast.csv"))
    actual_vs_forecast.rename(columns={"ds": "Date", "y": "y"}, inplace=True)

    return forecast_df, actual_vs_forecast, results

