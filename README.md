"""
multi_model_ensemble_20y_with_metrics.py

Purpose:
- Combine LSTM, SARIMAX, and GRU 20-year forecasts (daily & monthly)
  into a single ensemble forecast per ticker and date.
- Use RMSE-based weights to compute ensemble predictions.
- Approximate ensemble confidence intervals from model RMSEs.
- Aggregate and display performance metrics across models:
    * Prediction effectiveness: DirectionAccuracy, DirectionPrecision, DirectionRecall
    * Prediction accuracy: MAE, RMSE, R2
- Provide:
    * Performance summary table (saved to CSV)
    * RMSE bar chart by Ticker / Frequency / Model
    * Time series plots of actual vs ensemble forecast with timestamps and CI bands.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import yfinance as yf

FORECAST_DIR = Path("forecasts")


# ============================================================
# FORECAST LOADERS (PER MODEL)
# ============================================================

def load_lstm_forecast(freq: str) -> pd.DataFrame:
    """
    Load LSTM 20-year forecasts for a given frequency.

    Expected files:
        forecasts/all_tickers_20y_daily_forecast.csv
        forecasts/all_tickers_20y_monthly_forecast.csv

    Columns in LSTM CSV:
        Date, Ticker, PredictedPrice
    """
    fname = FORECAST_DIR / f"all_tickers_20y_{freq}_forecast.csv"
    df = pd.read_csv(fname, parse_dates=["Date"])
    df["Model"] = "LSTM"
    df["Frequency"] = freq
    return df[["Ticker", "Date", "Frequency", "Model", "PredictedPrice"]]


def load_sarimax_forecast(freq: str) -> pd.DataFrame:
    """
    Load SARIMAX 20-year forecasts for a given frequency.

    Expected files:
        sarimax_all_tickers_20y_daily_forecast.csv
        sarimax_all_tickers_20y_monthly_forecast.csv

    Columns:
        Date, Ticker, PredictedPrice
    """
    fname = Path(f"sarimax_all_tickers_20y_{freq}_forecast.csv")
    df = pd.read_csv(fname, parse_dates=["Date"])
    df["Model"] = "SARIMAX"
    df["Frequency"] = freq
    return df[["Ticker", "Date", "Frequency", "Model", "PredictedPrice"]]


def load_gru_forecast(freq: str) -> pd.DataFrame:
    """
    Load GRU 20-year forecasts for a given frequency.

    Expected files:
        forecasts/gru_20y_daily_forecast.csv
        forecasts/gru_20y_monthly_forecast.csv

    Columns:
        Date, Ticker, PredictedPrice
    """
    if freq == "daily":
        fname = FORECAST_DIR / "gru_20y_daily_forecast.csv"
    else:
        fname = FORECAST_DIR / "gru_20y_monthly_forecast.csv"

    df = pd.read_csv(fname, parse_dates=["Date"])
    df["Model"] = "GRU"
    df["Frequency"] = freq
    return df[["Ticker", "Date", "Frequency", "Model", "PredictedPrice"]]


# ============================================================
# METRIC LOADERS (PER MODEL)
# ============================================================

def load_lstm_metrics(freq: str) -> pd.DataFrame:
    """
    Load LSTM metrics for the specified frequency.

    Expected files:
        forecasts/all_tickers_daily_metrics.csv
        forecasts/all_tickers_monthly_metrics.csv

    Expected columns (from LSTM script):
        Ticker, Mode, TrainStart, TrainEnd, TestStart, TestEnd,
        RMSE, MAE, MAPE

    This function:
        - Normalizes column names
        - Adds missing columns (R2, direction metrics) as NaN
    """
    if freq == "daily":
        fname = FORECAST_DIR / "all_tickers_daily_metrics.csv"
    else:
        fname = FORECAST_DIR / "all_tickers_monthly_metrics.csv"

    df = pd.read_csv(fname)
    df["Model"] = "LSTM"
    df["frequency"] = df["Mode"].str.lower()
    df = df[df["frequency"] == freq]

    # Ensure these standard columns exist for later aggregation
    if "MAE" not in df.columns:
        df["MAE"] = np.nan
    if "RMSE" not in df.columns:
        df["RMSE"] = np.nan
    if "R2" not in df.columns:
        df["R2"] = np.nan

    # Direction metrics not provided by LSTM script -> NaN
    df["DirectionAccuracy"] = np.nan
    df["DirectionPrecision"] = np.nan
    df["DirectionRecall"] = np.nan

    # Keep a consistent set of columns
    return df[[
        "Ticker",
        "frequency",
        "Model",
        "MAE",
        "RMSE",
        "R2",
        "DirectionAccuracy",
        "DirectionPrecision",
        "DirectionRecall",
    ]]


def load_sarimax_metrics(freq: str) -> pd.DataFrame:
    """
    Load SARIMAX metrics for the specified frequency.

    Expected file:
        sarimax_performance_summary.csv

    Expected columns (from SARIMAX script):
        Ticker, Frequency, TrainStart, TrainEnd, TestStart, TestEnd,
        MAE, RMSE, R2, DirectionAccuracy, DirectionPrecision, DirectionRecall
    """
    fname = Path("sarimax_performance_summary.csv")
    df = pd.read_csv(fname)
    df["Model"] = "SARIMAX"
    df["frequency"] = df["Frequency"].str.lower()
    df = df[df["frequency"] == freq]

    # Ensure all standard columns exist
    for col in ["MAE", "RMSE", "R2", "DirectionAccuracy", "DirectionPrecision", "DirectionRecall"]:
        if col not in df.columns:
            df[col] = np.nan

    return df[[
        "Ticker",
        "frequency",
        "Model",
        "MAE",
        "RMSE",
        "R2",
        "DirectionAccuracy",
        "DirectionPrecision",
        "DirectionRecall",
    ]]


def load_gru_metrics(freq: str) -> pd.DataFrame:
    """
    Load GRU metrics for the specified frequency.

    Expected file:
        forecasts/gru_performance_summary.csv

    Expected columns (from GRU script):
        ticker, frequency, test_samples, MAE, RMSE, R2,
        direction_accuracy, direction_precision, direction_recall,
        future_prediction_date, future_predicted_price,
        future_ci_lower, future_ci_upper
    """
    fname = FORECAST_DIR / "gru_performance_summary.csv"
    df = pd.read_csv(fname)
    df["Model"] = "GRU"
    df["frequency"] = df["frequency"].str.lower()
    df = df[df["frequency"] == freq]

    # Rename to standard names
    df = df.rename(
        columns={
            "ticker": "Ticker",
            "direction_accuracy": "DirectionAccuracy",
            "direction_precision": "DirectionPrecision",
            "direction_recall": "DirectionRecall",
        }
    )

    # Ensure standard columns exist
    for col in ["MAE", "RMSE", "R2", "DirectionAccuracy", "DirectionPrecision", "DirectionRecall"]:
        if col not in df.columns:
            df[col] = np.nan

    return df[[
        "Ticker",
        "frequency",
        "Model",
        "MAE",
        "RMSE",
        "R2",
        "DirectionAccuracy",
        "DirectionPrecision",
        "DirectionRecall",
    ]]


def load_all_metrics(freq: str) -> pd.DataFrame:
    """
    Stack metrics from LSTM, SARIMAX, and GRU into a single table.

    Standardized columns:
        Ticker, frequency, Model,
        MAE, RMSE, R2,
        DirectionAccuracy, DirectionPrecision, DirectionRecall
    """
    m_lstm = load_lstm_metrics(freq)
    m_sari = load_sarimax_metrics(freq)
    m_gru = load_gru_metrics(freq)

    metrics_all = pd.concat([m_lstm, m_sari, m_gru], ignore_index=True, sort=False)
    return metrics_all


# ============================================================
# WEIGHTS & ENSEMBLE UNCERTAINTY
# ============================================================

def build_weight_table_from_metrics(metrics_all: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-(Ticker, frequency, Model) weights using RMSE:
        w_i ∝ 1 / RMSE_i^2

    Returns DataFrame:
        Ticker, Frequency, Model, Weight
    """
    rows = []
    for (ticker, freq), group in metrics_all.groupby(["Ticker", "frequency"]):
        # Drop models with missing RMSE
        group = group.dropna(subset=["RMSE"])
        if len(group) == 0:
            continue

        inv_sq = 1.0 / (group["RMSE"].values ** 2)
        w = inv_sq / inv_sq.sum()

        for model, weight in zip(group["Model"].values, w):
            rows.append(
                {
                    "Ticker": ticker,
                    "Frequency": freq,
                    "Model": model,
                    "Weight": weight,
                }
            )

    return pd.DataFrame(rows)


def compute_ensemble_sigma_lookup(
    metrics_all: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> dict:
    """
    Approximate ensemble standard deviation (sigma) of prediction errors per
    (Ticker, Frequency) using base model RMSEs and ensemble weights.

    Approximation (assuming uncorrelated errors):
        RMSE_ensemble^2 ≈ Σ_i (w_i^2 * RMSE_i^2)

    Returns:
        dict keyed by (Ticker, Frequency) -> sigma (float)
    """
    sigma_lookup = {}
    for (ticker, freq), w_group in weights_df.groupby(["Ticker", "Frequency"]):
        m_group = metrics_all[(metrics_all["Ticker"] == ticker) &
                              (metrics_all["frequency"] == freq)]

        merged = w_group.merge(
            m_group[["Ticker", "frequency", "Model", "RMSE"]],
            left_on=["Ticker", "Frequency", "Model"],
            right_on=["Ticker", "frequency", "Model"],
            how="left",
        )
        merged = merged.dropna(subset=["RMSE"])
        if len(merged) == 0:
            sigma_lookup[(ticker, freq)] = np.nan
            continue

        rmse_vals = merged["RMSE"].values
        w_vals = merged["Weight"].values
        mse_vals = rmse_vals ** 2

        ensemble_mse = np.sum((w_vals ** 2) * mse_vals)
        sigma_lookup[(ticker, freq)] = float(np.sqrt(ensemble_mse))

    return sigma_lookup


def build_ensemble_performance_summary(
    metrics_all: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a summary performance table that includes:

    - One row per base model (LSTM, SARIMAX, GRU) / ticker / frequency
    - One additional row per ticker / frequency for the ensemble:
        * Approximate RMSE_ensemble (as above)
        * Approximate MAE_ensemble = Σ_i w_i * MAE_i
        * R2 and direction metrics left as NaN (can be extended if desired)

    Returns:
        DataFrame with columns:
            Ticker, Frequency, Model,
            MAE, RMSE, R2,
            DirectionAccuracy, DirectionPrecision, DirectionRecall,
            Weight, IsEnsemble
    """
    rows = []

    for (ticker, freq), m_group in metrics_all.groupby(["Ticker", "frequency"]):
        # Add base models
        for _, row in m_group.iterrows():
            # Look up weight for this (ticker, freq, model)
            w_match = weights_df[
                (weights_df["Ticker"] == ticker)
                & (weights_df["Frequency"] == freq)
                & (weights_df["Model"] == row["Model"])
            ]
            weight_val = w_match["Weight"].iloc[0] if len(w_match) > 0 else np.nan

            rows.append(
                {
                    "Ticker": ticker,
                    "Frequency": freq,
                    "Model": row["Model"],
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "R2": row["R2"],
                    "DirectionAccuracy": row["DirectionAccuracy"],
                    "DirectionPrecision": row["DirectionPrecision"],
                    "DirectionRecall": row["DirectionRecall"],
                    "Weight": weight_val,
                    "IsEnsemble": False,
                }
            )

        # Add ensemble row
        w_group = weights_df[
            (weights_df["Ticker"] == ticker)
            & (weights_df["Frequency"] == freq)
        ]
        if len(w_group) == 0:
            continue

        merged = w_group.merge(
            m_group[["Ticker", "frequency", "Model", "RMSE", "MAE"]],
            left_on=["Ticker", "Frequency", "Model"],
            right_on=["Ticker", "frequency", "Model"],
            how="left",
        ).dropna(subset=["RMSE"])

        if len(merged) == 0:
            continue

        w_vals = merged["Weight"].values
        rmse_vals = merged["RMSE"].values
        mae_vals = merged["MAE"].values

        ensemble_rmse = float(np.sqrt(np.sum((w_vals ** 2) * (rmse_vals ** 2))))
        ensemble_mae = float(np.sum(w_vals * mae_vals))

        rows.append(
            {
                "Ticker": ticker,
                "Frequency": freq,
                "Model": "Ensemble",
                "MAE": ensemble_mae,
                "RMSE": ensemble_rmse,
                "R2": np.nan,
                "DirectionAccuracy": np.nan,
                "DirectionPrecision": np.nan,
                "DirectionRecall": np.nan,
                "Weight": 1.0,
                "IsEnsemble": True,
            }
        )

    summary_df = pd.DataFrame(rows)
    return summary_df


# ============================================================
# ENSEMBLE (DAILY / MONTHLY)
# ============================================================

def ensemble_for_frequency(freq: str) -> pd.DataFrame:
    """
    Build ensemble 20-year forecast for a given frequency ("daily" or "monthly").

    Steps:
    1) Load all model forecasts (LSTM, SARIMAX, GRU).
    2) Load metrics and compute RMSE-based weights.
    3) Compute per-(Ticker, Frequency) ensemble sigma (for CI).
    4) For each (Ticker, Date), combine model predictions:
        - Weighted average using computed weights.
        - If weights missing -> equal-weight average.
        - Attach approximate 95% confidence interval:
            [FinalPredictedPrice ± 1.96 * sigma_ensemble]
    """
    # Load forecasts
    lstm_fc = load_lstm_forecast(freq)
    sarima_fc = load_sarimax_forecast(freq)
    gru_fc = load_gru_forecast(freq)

    forecasts = pd.concat([lstm_fc, sarima_fc, gru_fc], ignore_index=True)

    # Load metrics & weights
    metrics_all = load_all_metrics(freq)
    weights_df = build_weight_table_from_metrics(metrics_all)
    sigma_lookup = compute_ensemble_sigma_lookup(metrics_all, weights_df)

    # Merge weights into forecasts
    forecasts_w = forecasts.merge(
        weights_df,
        on=["Ticker", "Frequency", "Model"],
        how="left",
    )

    def combine_group(group: pd.DataFrame) -> pd.Series:
        """
        Combine predictions across models for a single (Ticker, Date, Frequency).

        Returns:
            FinalPredictedPrice: weighted average of model predictions
            LowerCI, UpperCI   : approximate 95% CI using ensemble sigma
            ModelsUsed         : comma-separated list of contributing models
        """
        preds = group["PredictedPrice"].values
        models = group["Model"].values
        weights = group["Weight"].values

        valid_mask = ~np.isnan(preds)
        preds = preds[valid_mask]
        models = models[valid_mask]
        weights = weights[valid_mask]

        ticker = group["Ticker"].iloc[0]
        frequency = group["Frequency"].iloc[0]

        if len(preds) == 0:
            return pd.Series(
                {
                    "FinalPredictedPrice": np.nan,
                    "LowerCI": np.nan,
                    "UpperCI": np.nan,
                    "ModelsUsed": "",
                }
            )

        # If no RMSE-based weights (all NaN), fallback to equal weights
        if len(weights) == 0 or np.all(np.isnan(weights)):
            w = np.ones_like(preds) / len(preds)
        else:
            # Replace NaNs with 0, re-normalize
            weights = np.where(np.isnan(weights), 0.0, weights)
            if weights.sum() == 0:
                w = np.ones_like(preds) / len(preds)
            else:
                w = weights / weights.sum()

        final_pred = float(np.sum(w * preds))

        # Approximate CI using ensemble sigma for this Ticker/Frequency
        sigma = sigma_lookup.get((ticker, frequency), np.nan)
        if np.isnan(sigma):
            lower_ci = np.nan
            upper_ci = np.nan
        else:
            ci_mult = 1.96
            lower_ci = final_pred - ci_mult * sigma
            upper_ci = final_pred + ci_mult * sigma

        return pd.Series(
            {
                "FinalPredictedPrice": final_pred,
                "LowerCI": lower_ci,
                "UpperCI": upper_ci,
                "ModelsUsed": ",".join(models),
            }
        )

    grouped = forecasts_w.groupby(["Ticker", "Date", "Frequency"], as_index=False)
    ensemble_df = grouped.apply(combine_group)
    ensemble_df.reset_index(drop=True, inplace=True)

    return ensemble_df, metrics_all, weights_df


# ============================================================
# VISUALIZATION: PERFORMANCE & PRICE PATHS
# ============================================================

def plot_rmse_bar_chart(summary_df: pd.DataFrame, freq: str) -> None:
    """
    Bar chart of RMSE by (Ticker, Model) for given frequency.

    - Ensemble appears as an additional "model" row.
    """
    df = summary_df[summary_df["Frequency"] == freq].copy()
    if df.empty:
        return

    # Concise x-axis labels: "<Ticker> - <Model>"
    df["Label"] = df["Ticker"] + " - " + df["Model"]

    plt.figure(figsize=(10, 5))
    plt.bar(df["Label"], df["RMSE"])
    plt.title(f"RMSE by Ticker and Model ({freq.capitalize()})")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_actual_vs_ensemble(
    ensemble_df: pd.DataFrame,
    freq: str,
    max_tickers: int = 3,
) -> None:
    """
    For each ticker (up to `max_tickers`), plot:

    - Historical actual prices (downloaded from Yahoo Finance)
    - Ensemble 20-year forecast (FinalPredictedPrice)
    - Approximate 95% CI band around ensemble forecast

    This gives a time series view with timestamps showing:
        * where the historical data ends
        * where the forecast horizon begins
    """
    tickers = ensemble_df["Ticker"].unique()
    tickers = tickers[:max_tickers]  # limit to avoid too many plots

    for ticker in tickers:
        df_pred = ensemble_df[ensemble_df["Ticker"] == ticker].sort_values("Date")
        if df_pred.empty:
            continue

        # Forecast start is the first ensemble prediction date
        forecast_start = df_pred["Date"].min()

        # Download full historical prices from Yahoo Finance
        hist = yf.download(ticker, period="max", auto_adjust=True, progress=False)
        if hist.empty:
            continue

        if "Close" in hist.columns:
            hist_price = hist[["Close"]].rename(columns={"Close": "ActualPrice"})
        else:
            # Fallback if structure is different
            hist_price = hist.iloc[:, 0].to_frame("ActualPrice")

        hist_price = hist_price.sort_index()

        # Historical segment up to forecast start
        hist_segment = hist_price[hist_price.index <= forecast_start]

        plt.figure(figsize=(12, 5))

        # Actual historical prices
        plt.plot(
            hist_segment.index,
            hist_segment["ActualPrice"].values,
            label="Actual price (historical)",
        )

        # Ensemble forecast + CI
        plt.plot(
            df_pred["Date"],
            df_pred["FinalPredictedPrice"],
            label="Ensemble forecast",
            linestyle="--",
        )

        if "LowerCI" in df_pred.columns:
            plt.fill_between(
                df_pred["Date"],
                df_pred["LowerCI"],
                df_pred["UpperCI"],
                alpha=0.2,
                label="Approx. 95% CI (ensemble)",
            )

        # Vertical line marking forecast start
        plt.axvline(forecast_start, linestyle=":", label="Forecast start")

        plt.title(f"{ticker} - {freq.capitalize()} actual vs ensemble forecast")
        plt.xlabel("Timestamp")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ============================================================
# MAIN DRIVER
# ============================================================

def main():
    # =====================
    # DAILY ENSEMBLE
    # =====================
    print("=== Building ensemble forecast for DAILY frequency ===")
    daily_ensemble, daily_metrics_all, daily_weights = ensemble_for_frequency("daily")
    daily_out = FORECAST_DIR / "ensemble_20y_daily_forecast.csv"
    daily_ensemble.to_csv(daily_out, index=False)
    print(f"[Daily] Ensemble 20-year forecast saved to: {daily_out.resolve()}")

    # Build daily performance summary (including approximate ensemble metrics)
    daily_summary = build_ensemble_performance_summary(daily_metrics_all, daily_weights)

    # =====================
    # MONTHLY ENSEMBLE
    # =====================
    print("\n=== Building ensemble forecast for MONTHLY frequency ===")
    monthly_ensemble, monthly_metrics_all, monthly_weights = ensemble_for_frequency("monthly")
    monthly_out = FORECAST_DIR / "ensemble_20y_monthly_forecast.csv"
    monthly_ensemble.to_csv(monthly_out, index=False)
    print(f"[Monthly] Ensemble 20-year forecast saved to: {monthly_out.resolve()}")

    monthly_summary = build_ensemble_performance_summary(monthly_metrics_all, monthly_weights)

    # =====================
    # COMBINE SUMMARY & SAVE
    # =====================
    performance_summary = pd.concat([daily_summary, monthly_summary], ignore_index=True)
    perf_out = FORECAST_DIR / "ensemble_performance_summary.csv"
    performance_summary.to_csv(perf_out, index=False)
    print(f"\nOverall performance summary table saved to: {perf_out.resolve()}")

    # =====================
    # DISPLAY PERFORMANCE SUMMARY
    # =====================
    print("\n=== Performance Summary (Daily) ===")
    print(
        performance_summary[performance_summary["Frequency"] == "daily"]
        .sort_values(["Ticker", "IsEnsemble", "Model"])
        .to_string(index=False)
    )

    print("\n=== Performance Summary (Monthly) ===")
    print(
        performance_summary[performance_summary["Frequency"] == "monthly"]
        .sort_values(["Ticker", "IsEnsemble", "Model"])
        .to_string(index=False)
    )

    # =====================
    # PLOT RMSE CHARTS
    # =====================
    print("\nGenerating RMSE bar charts for Daily and Monthly frequencies...")
    plot_rmse_bar_chart(performance_summary, freq="daily")
    plot_rmse_bar_chart(performance_summary, freq="monthly")

    # =====================
    # VISUAL COMPARISON: ACTUAL VS ENSEMBLE
    # =====================
    print("\nGenerating actual vs ensemble forecast plots (with timestamps and CI)...")
    # Limit to a few tickers for readability; adjust max_tickers as desired.
    plot_actual_vs_ensemble(daily_ensemble, freq="daily", max_tickers=3)
    plot_actual_vs_ensemble(monthly_ensemble, freq="monthly", max_tickers=3)


if __name__ == "__main__":
    main()
