# src/export.py — Export & Visualize Forecasts for FUTURE_ML_01

import os
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------
# Helper: load CSV with normalized Date
# -----------------------------------
def load_with_date(path):
    df_temp = pd.read_csv(path)
    print(f"📂 {os.path.basename(path)} columns:", df_temp.columns.tolist())
    date_col = "Date" if "Date" in df_temp.columns else ("ds" if "ds" in df_temp.columns else None)
    if date_col is None:
        raise ValueError(f"No Date/ds column found in {path}")
    df = pd.read_csv(path, parse_dates=[date_col])
    return df.rename(columns={date_col: "Date"})


# -----------------------------------
# Main Export Function
# -----------------------------------
def export_forecasts(plot: bool = True):
    """
    Load forecast & actual_vs_forecast CSVs, visualize them, and export final CSVs.

    Args:
        plot (bool): Whether to show matplotlib plots.

    Returns:
        df_forecast (pd.DataFrame): Forecast results.
        df_actual_forecast (pd.DataFrame): Actual vs forecast combined.
    """

    # -------------------------------
    # Paths
    # -------------------------------
    forecast_path = "../data/forecast_results.csv" if os.path.exists("../data/forecast_results.csv") else "data/forecast_results.csv"
    actual_path   = "../data/actual_vs_forecast.csv" if os.path.exists("../data/actual_vs_forecast.csv") else "data/actual_vs_forecast.csv"

    if not os.path.exists(forecast_path):
        raise FileNotFoundError(f"{forecast_path} not found. Run train.py first.")
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"{actual_path} not found. Run train.py first.")

    # -------------------------------
    # Load CSVs
    # -------------------------------
    df_forecast = load_with_date(forecast_path)
    df_actual_forecast = load_with_date(actual_path)

    print("✅ Forecast loaded:", df_forecast.shape)
    print("✅ Actual vs Forecast loaded:", df_actual_forecast.shape)

    # -------------------------------
    # Plot forecast horizon
    # -------------------------------
    if plot:
        forecast_col = None
        for cand in ["Forecast", "yhat", "Prediction", "Predicted", "sales_forecast"]:
            if cand in df_forecast.columns:
                forecast_col = cand
                break
        if forecast_col is None:
            raise ValueError("❌ No forecast column found in df_forecast!")

        plt.figure(figsize=(14, 6))
        plt.plot(df_forecast["Date"], df_forecast[forecast_col], label="Forecast", color="blue")

        lower_col, upper_col = None, None
        for cand_l, cand_u in [("Lower","Upper"), ("yhat_lower","yhat_upper"), ("lower","upper")]:
            if cand_l in df_forecast.columns and cand_u in df_forecast.columns:
                lower_col, upper_col = cand_l, cand_u
                break

        if lower_col and upper_col:
            plt.fill_between(
                df_forecast["Date"],
                df_forecast[lower_col],
                df_forecast[upper_col],
                color="lightblue", alpha=0.3, label="Confidence Interval"
            )

        plt.title("Forecast Horizon")
        plt.xlabel("Date"); plt.ylabel("Sales")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.show()

    # -------------------------------
    # Plot actual vs forecast
    # -------------------------------
    if plot:
        actual_col, forecast_col = None, None
        for cand in ["Sales", "Actual", "y", "Observed", "Value"]:
            if cand in df_actual_forecast.columns:
                actual_col = cand
                break
        if actual_col is None:
            raise ValueError("❌ Actual sales column not found in df_actual_forecast!")

        for cand in ["Forecast", "yhat", "Prediction", "Predicted"]:
            if cand in df_actual_forecast.columns:
                forecast_col = cand
                break
        if forecast_col is None:
            raise ValueError("❌ Forecast column not found in df_actual_forecast!")

        plt.figure(figsize=(14, 6))
        plt.plot(df_actual_forecast["Date"], df_actual_forecast[actual_col], label="Actual", color="black")
        plt.plot(df_actual_forecast["Date"], df_actual_forecast[forecast_col], label="Forecast", color="blue")

        lower_col, upper_col = None, None
        for cand_l, cand_u in [("Lower","Upper"), ("yhat_lower","yhat_upper"), ("lower","upper")]:
            if cand_l in df_actual_forecast.columns and cand_u in df_actual_forecast.columns:
                lower_col, upper_col = cand_l, cand_u
                break

        if lower_col and upper_col:
            plt.fill_between(
                df_actual_forecast["Date"],
                df_actual_forecast[lower_col],
                df_actual_forecast[upper_col],
                color="lightblue", alpha=0.3, label="Confidence Interval"
            )

        plt.title("Actual vs Forecast (Validation Period)")
        plt.xlabel("Date"); plt.ylabel("Sales")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.show()

    # -------------------------------
    # Export final cleaned CSVs
    # -------------------------------
    out_forecast = "../data/final_forecast.csv" if os.path.isdir("../data") else "data/final_forecast.csv"
    out_actual   = "../data/final_actual_vs_forecast.csv" if os.path.isdir("../data") else "data/final_actual_vs_forecast.csv"

    df_forecast.to_csv(out_forecast, index=False)
    df_actual_forecast.to_csv(out_actual, index=False)

    print("💾 Exported final forecast →", out_forecast)
    print("💾 Exported final actual vs forecast →", out_actual)

    return df_forecast, df_actual_forecast
