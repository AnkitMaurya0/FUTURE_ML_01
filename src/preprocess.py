# src/preprocess.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)


def load_raw_data():
    """Load raw sales CSV from typical paths"""
    candidates = ["../data/raw_sales.csv", "data/raw_sales.csv", "./raw_sales.csv"]
    raw_path = next((p for p in candidates if os.path.exists(p)), None)

    if raw_path is None:
        raise FileNotFoundError("raw_sales.csv not found. Place it in FUTURE_ML_01/data/")

    df_raw = pd.read_csv(raw_path)
    print("✅ Loaded:", raw_path)
    print("Shape:", df_raw.shape)
    return df_raw


def detect_columns(df_raw):
    """Detect date and sales column names automatically"""
    date_col_candidates = ["Order Date", "order_date", "Date", "date"]
    sales_col_candidates = ["Sales", "sales", "Revenue", "revenue", "Amount", "amount"]

    date_col = next((c for c in date_col_candidates if c in df_raw.columns), None)
    sales_col = next((c for c in sales_col_candidates if c in df_raw.columns), None)

    if date_col is None:
        raise ValueError(f"No date column found. Expected one of: {date_col_candidates}")
    if sales_col is None:
        raise ValueError(f"No sales/revenue column found. Expected one of: {sales_col_candidates}")

    print(f"Using date column: {date_col}")
    print(f"Using sales column: {sales_col}")

    return date_col, sales_col


def clean_data(df_raw, date_col, sales_col):
    """Clean the raw data: parse dates, drop invalid rows, convert sales to numeric"""
    df = df_raw.copy()

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[date_col])
    after = len(df)

    # Sales column
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    df = df.dropna(subset=[sales_col])

    print(f"🧹 Dropped {before - after} rows with invalid dates.")
    print("Missing values now:\n", df[[date_col, sales_col]].isna().sum())
    print("Dtypes:\n", df[[date_col, sales_col]].dtypes)

    return df


def aggregate_daily(df, date_col, sales_col):
    """Aggregate sales data at daily level"""
    daily = (
        df.groupby(df[date_col].dt.normalize())[sales_col]
          .sum()
          .reset_index()
          .rename(columns={date_col: "Date", sales_col: "Sales"})
          .sort_values("Date")
          .reset_index(drop=True)
    )

    print("✅ Daily aggregation complete. Shape:", daily.shape)
    return daily


def save_cleaned_data(daily):
    """Save cleaned daily data to ../data/cleaned_sales.csv"""
    out_candidates = ["../data/cleaned_sales.csv", "data/cleaned_sales.csv"]
    cleaned_path = out_candidates[0] if os.path.isdir("../data") else out_candidates[1]

    daily.to_csv(cleaned_path, index=False)
    print("💾 Saved cleaned daily data →", cleaned_path)
    return cleaned_path


def generate_stats(daily):
    """Print quick stats of daily data"""
    print("📊 Quick Stats")
    print("Date range:", daily["Date"].min().date(), "→", daily["Date"].max().date())
    print("Rows (days):", len(daily))
    print("Average daily sales:", round(daily["Sales"].mean(), 2))
    print("Median  daily sales:", round(daily["Sales"].median(), 2))
    print("Max     daily sales:", round(daily["Sales"].max(), 2))


def plot_daily(daily):
    """Plot daily sales trend"""
    plt.figure(figsize=(14, 5))
    plt.plot(daily["Date"], daily["Sales"])
    plt.title("Daily Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def aggregate_monthly(daily):
    """Aggregate monthly sales"""
    monthly = (
        daily.assign(Year=daily["Date"].dt.year, Month=daily["Date"].dt.month)
             .groupby(["Year", "Month"], as_index=False)["Sales"].sum()
             .sort_values(["Year", "Month"])
    )

    monthly["MonthPeriod"] = pd.to_datetime(monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01")
    print("✅ Monthly summary ready. Shape:", monthly.shape)
    return monthly


def plot_monthly(monthly):
    """Plot monthly sales trend"""
    plt.figure(figsize=(12, 5))
    plt.plot(monthly["MonthPeriod"], monthly["Sales"], marker="o")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def feature_engineering(daily):
    """Create date-related features for modeling"""
    daily_fe = daily.copy()
    daily_fe["Year"] = daily_fe["Date"].dt.year
    daily_fe["Month"] = daily_fe["Date"].dt.month
    daily_fe["DayOfWeek"] = daily_fe["Date"].dt.dayofweek
    daily_fe["IsMonthStart"] = daily_fe["Date"].dt.is_month_start.astype(int)
    daily_fe["IsMonthEnd"] = daily_fe["Date"].dt.is_month_end.astype(int)
    return daily_fe


def save_monthly(monthly):
    """Save monthly sales summary for Power BI"""
    monthly_out = "../data/monthly_sales.csv" if os.path.isdir("../data") else "data/monthly_sales.csv"
    monthly[["MonthPeriod", "Sales"]].to_csv(monthly_out, index=False)
    print("💾 Saved monthly summary →", monthly_out)
    return monthly_out


def preprocess_pipeline():
    """Full pipeline to load, clean, aggregate, and save"""
    df_raw = load_raw_data()
    date_col, sales_col = detect_columns(df_raw)
    df = clean_data(df_raw, date_col, sales_col)
    daily = aggregate_daily(df, date_col, sales_col)
    save_cleaned_data(daily)
    generate_stats(daily)
    plot_daily(daily)
    monthly = aggregate_monthly(daily)
    plot_monthly(monthly)
    save_monthly(monthly)
    features = feature_engineering(daily)
    return daily, monthly, features
