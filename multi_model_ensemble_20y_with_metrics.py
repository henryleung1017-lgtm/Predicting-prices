"""Ensemble price forecasting pipeline with CLI and validation.

This script combines LSTM, SARIMAX, and GRU 20-year forecasts into weighted
ensembles for daily and monthly frequencies. It validates inputs, builds
weights from RMSE metrics, estimates confidence intervals, and optionally
produces plots. Use the CLI flags to control which frequencies to process,
where outputs are written, and whether plots and network calls are allowed.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES & CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class InputFiles:
    """File layout used by the script.

    Paths are relative to the repository root; override via CLI if desired.
    """

    forecast_dir: Path = Path("forecasts")
    lstm_daily: Path = Path("forecasts/all_tickers_20y_daily_forecast.csv")
    lstm_monthly: Path = Path("forecasts/all_tickers_20y_monthly_forecast.csv")
    sarimax_daily: Path = Path("sarimax_all_tickers_20y_daily_forecast.csv")
    sarimax_monthly: Path = Path("sarimax_all_tickers_20y_monthly_forecast.csv")
    gru_daily: Path = Path("forecasts/gru_20y_daily_forecast.csv")
    gru_monthly: Path = Path("forecasts/gru_20y_monthly_forecast.csv")

    lstm_daily_metrics: Path = Path("forecasts/all_tickers_daily_metrics.csv")
    lstm_monthly_metrics: Path = Path("forecasts/all_tickers_monthly_metrics.csv")
    sarimax_metrics: Path = Path("sarimax_performance_summary.csv")
    gru_metrics: Path = Path("forecasts/gru_performance_summary.csv")

    def required_for_frequency(self, freq: str) -> Set[Path]:
        """Return the forecast and metric files required for the given frequency."""

        freq = _normalize_frequency(freq)
        forecast_files = {
            "daily": [self.lstm_daily, self.sarimax_daily, self.gru_daily],
            "monthly": [self.lstm_monthly, self.sarimax_monthly, self.gru_monthly],
        }[freq]

        metric_files = {
            "daily": [self.lstm_daily_metrics, self.sarimax_metrics, self.gru_metrics],
            "monthly": [self.lstm_monthly_metrics, self.sarimax_metrics, self.gru_metrics],
        }[freq]

        return set(forecast_files + metric_files)


REQUIRED_FORECAST_COLUMNS = {"Date", "Ticker", "PredictedPrice"}
REQUIRED_METRIC_COLUMNS = {
    "Ticker",
    "RMSE",
    "MAE",
}
METRIC_DATE_COLUMNS = ["TrainEnd", "TestEnd", "TrainEndDate", "TestEndDate"]
DIRECTIONAL_COLS = ["DirectionAccuracy", "DirectionPrecision", "DirectionRecall"]


# ============================================================
# UTILITIES
# ============================================================

def _require_columns(df: pd.DataFrame, required: Iterable[str], *, context: str) -> pd.DataFrame:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {context}")
    return df


def _validate_numeric_columns(df: pd.DataFrame, numeric_cols: Iterable[str], *, context: str) -> pd.DataFrame:
    """Ensure numeric columns are coercible to float and contain finite values."""

    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"Missing numeric column '{col}' in {context}")
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().any():
            raise ValueError(f"Non-numeric or missing values found in column '{col}' for {context}")
        df[col] = coerced.astype(float)
    return df


def _read_csv(path: Path, *, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def _normalize_frequency(freq: str) -> str:
    freq = freq.lower()
    if freq not in {"daily", "monthly"}:
        raise ValueError("Frequency must be 'daily' or 'monthly'")
    return freq


def validate_required_files(files: InputFiles, frequencies: Sequence[str]) -> None:
    """Ensure all required files exist before starting the pipeline.

    This catches missing inputs up-front instead of failing mid-pipeline with
    less clear stack traces.
    """

    missing: List[Path] = []
    for freq in {_normalize_frequency(f) for f in frequencies}:
        for path in files.required_for_frequency(freq):
            if not path.exists():
                missing.append(path)

    if missing:
        joined = "\n".join(str(p) for p in sorted(set(missing)))
        raise FileNotFoundError(
            "The following required inputs are missing:\n" + joined
        )


def validate_metric_recency(metrics_all: pd.DataFrame, *, max_age_days: Optional[int], context: str) -> None:
    """Warn if metrics look stale when date columns are present.

    If no known date columns exist, the function exits quietly because the
    script cannot assess recency.
    """

    if max_age_days is None:
        return

    date_cols_present = [c for c in METRIC_DATE_COLUMNS if c in metrics_all.columns]
    if not date_cols_present:
        LOGGER.warning(
            "Recency check skipped for %s: no date columns found among %s",
            context,
            METRIC_DATE_COLUMNS,
        )
        return

    now = pd.Timestamp(datetime.utcnow().date())
    recent_mask: List[bool] = []
    for col in date_cols_present:
        parsed = pd.to_datetime(metrics_all[col], errors="coerce")
        age_days = (now - parsed).dt.days
        recent_mask.append(age_days <= max_age_days)

    combined_recent = pd.concat(recent_mask, axis=1).any(axis=1)
    stale = metrics_all.loc[~combined_recent, ["Ticker", "frequency"]].drop_duplicates()
    if not stale.empty:
        stale_pairs = "; ".join(f"{t} ({f})" for t, f in stale.values)
        LOGGER.warning(
            "Metrics older than %s days for: %s (source columns: %s)",
            max_age_days,
            stale_pairs,
            ", ".join(date_cols_present),
        )


# ============================================================
# FORECAST LOADERS (PER MODEL)
# ============================================================


def load_lstm_forecast(freq: str, files: InputFiles) -> pd.DataFrame:
    freq = _normalize_frequency(freq)
    fname = files.lstm_daily if freq == "daily" else files.lstm_monthly
    df = _read_csv(fname, parse_dates=["Date"])
    _require_columns(df, REQUIRED_FORECAST_COLUMNS, context=f"LSTM {freq} forecast")
    _validate_numeric_columns(df, ["PredictedPrice"], context=f"LSTM {freq} forecast")
    df = df.assign(Model="LSTM", Frequency=freq)
    return df[["Ticker", "Date", "Frequency", "Model", "PredictedPrice"]]


def load_sarimax_forecast(freq: str, files: InputFiles) -> pd.DataFrame:
    freq = _normalize_frequency(freq)
    fname = files.sarimax_daily if freq == "daily" else files.sarimax_monthly
    df = _read_csv(fname, parse_dates=["Date"])
    _require_columns(df, REQUIRED_FORECAST_COLUMNS, context=f"SARIMAX {freq} forecast")
    _validate_numeric_columns(df, ["PredictedPrice"], context=f"SARIMAX {freq} forecast")
    df = df.assign(Model="SARIMAX", Frequency=freq)
    return df[["Ticker", "Date", "Frequency", "Model", "PredictedPrice"]]


def load_gru_forecast(freq: str, files: InputFiles) -> pd.DataFrame:
    freq = _normalize_frequency(freq)
    fname = files.gru_daily if freq == "daily" else files.gru_monthly
    df = _read_csv(fname, parse_dates=["Date"])
    _require_columns(df, REQUIRED_FORECAST_COLUMNS, context=f"GRU {freq} forecast")
    _validate_numeric_columns(df, ["PredictedPrice"], context=f"GRU {freq} forecast")
    df = df.assign(Model="GRU", Frequency=freq)
    return df[["Ticker", "Date", "Frequency", "Model", "PredictedPrice"]]


# ============================================================
# METRIC LOADERS (PER MODEL)
# ============================================================


def _ensure_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["MAE", "RMSE", "R2", "DirectionAccuracy", "DirectionPrecision", "DirectionRecall"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _validate_metric_numeric(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    return _validate_numeric_columns(df, ["MAE", "RMSE"], context=context)


def load_lstm_metrics(freq: str, files: InputFiles) -> pd.DataFrame:
    freq = _normalize_frequency(freq)
    fname = files.lstm_daily_metrics if freq == "daily" else files.lstm_monthly_metrics
    df = _read_csv(fname)
    _require_columns(df, REQUIRED_METRIC_COLUMNS | {"Mode"}, context=f"LSTM {freq} metrics")
    df["frequency"] = df["Mode"].str.lower()
    df = df[df["frequency"] == freq]
    df = _ensure_metric_columns(df)
    df = _validate_metric_numeric(df, context=f"LSTM {freq} metrics")
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
    ]].assign(Model="LSTM")


def load_sarimax_metrics(freq: str, files: InputFiles) -> pd.DataFrame:
    freq = _normalize_frequency(freq)
    df = _read_csv(files.sarimax_metrics)
    _require_columns(df, REQUIRED_METRIC_COLUMNS | {"Frequency"}, context=f"SARIMAX {freq} metrics")
    df["frequency"] = df["Frequency"].str.lower()
    df = df[df["frequency"] == freq]
    df = _ensure_metric_columns(df)
    df = _validate_metric_numeric(df, context=f"SARIMAX {freq} metrics")
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
    ]].assign(Model="SARIMAX")


def load_gru_metrics(freq: str, files: InputFiles) -> pd.DataFrame:
    freq = _normalize_frequency(freq)
    df = _read_csv(files.gru_metrics)
    rename_map = {
        "ticker": "Ticker",
        "direction_accuracy": "DirectionAccuracy",
        "direction_precision": "DirectionPrecision",
        "direction_recall": "DirectionRecall",
    }
    df = df.rename(columns=rename_map)
    _require_columns(df, REQUIRED_METRIC_COLUMNS | {"frequency"}, context=f"GRU {freq} metrics")
    df["frequency"] = df["frequency"].str.lower()
    df = df[df["frequency"] == freq]
    df = _ensure_metric_columns(df)
    df = _validate_metric_numeric(df, context=f"GRU {freq} metrics")
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
    ]].assign(Model="GRU")


def load_all_metrics(freq: str, files: InputFiles) -> pd.DataFrame:
    metrics_frames = [
        load_lstm_metrics(freq, files),
        load_sarimax_metrics(freq, files),
        load_gru_metrics(freq, files),
    ]
    metrics = pd.concat(metrics_frames, ignore_index=True, sort=False)
    return metrics


def warn_on_missing_direction_metrics(metrics_all: pd.DataFrame, freq: str) -> None:
    has_dir = metrics_all[DIRECTIONAL_COLS].notna().any(axis=1)
    missing = metrics_all.loc[~has_dir, ["Ticker", "frequency", "Model"]]
    if missing.empty:
        return
    pairs = "; ".join(sorted({f"{t} ({f}) [{m}]" for t, f, m in missing.values}))
    LOGGER.warning("Directional metrics missing for %s: %s", freq, pairs)


def warn_on_metric_coverage(forecasts: pd.DataFrame, metrics_all: pd.DataFrame, freq: str) -> None:
    forecast_pairs = set(zip(forecasts["Ticker"], forecasts["Frequency"]))
    metric_pairs = set(zip(metrics_all["Ticker"], metrics_all["frequency"]))
    missing_pairs = forecast_pairs - metric_pairs
    if missing_pairs:
        LOGGER.warning(
            "Forecasts lack metrics for %s: %s",
            freq,
            "; ".join(sorted(f"{t} ({f})" for t, f in missing_pairs)),
        )


def warn_on_nonpositive_metrics(metrics_all: pd.DataFrame, freq: str) -> None:
    invalid = metrics_all[(metrics_all["RMSE"] <= 0) | (metrics_all["MAE"] <= 0)]
    if invalid.empty:
        return
    pairs = "; ".join(sorted({f"{t} ({f}) [{m}]" for t, f, m in invalid[["Ticker", "frequency", "Model"]].values}))
    LOGGER.warning("Non-positive MAE/RMSE values found for %s: %s", freq, pairs)


# ============================================================
# WEIGHTS & ENSEMBLE UNCERTAINTY
# ============================================================


def build_weight_table_from_metrics(metrics_all: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (ticker, freq), group in metrics_all.groupby(["Ticker", "frequency"]):
        group = group.dropna(subset=["RMSE"])
        if group.empty:
            LOGGER.warning("Skipping weights for %s (%s): no RMSE values", ticker, freq)
            continue

        inv_sq = 1.0 / (group["RMSE"].values ** 2)
        weights = inv_sq / inv_sq.sum()
        for model, weight in zip(group["Model"].values, weights):
            rows.append({
                "Ticker": ticker,
                "Frequency": freq,
                "Model": model,
                "Weight": weight,
            })
    return pd.DataFrame(rows)


def compute_ensemble_sigma_lookup(
    metrics_all: pd.DataFrame, weights_df: pd.DataFrame, *, assumed_correlation: float
) -> Dict[Tuple[str, str], float]:
    sigma_lookup: Dict[Tuple[str, str], float] = {}
    if assumed_correlation < 0 or assumed_correlation > 1:
        raise ValueError("assumed_correlation must be between 0 and 1")

    for (ticker, freq), w_group in weights_df.groupby(["Ticker", "Frequency"]):
        m_group = metrics_all[(metrics_all["Ticker"] == ticker) & (metrics_all["frequency"] == freq)]
        merged = w_group.merge(
            m_group[["Ticker", "frequency", "Model", "RMSE"]],
            left_on=["Ticker", "Frequency", "Model"],
            right_on=["Ticker", "frequency", "Model"],
            how="left",
        ).dropna(subset=["RMSE"])

        if merged.empty:
            LOGGER.warning("No RMSE values available for CI on %s (%s)", ticker, freq)
            sigma_lookup[(ticker, freq)] = np.nan
            continue

        rmse_vals = merged["RMSE"].to_numpy()
        w_vals = merged["Weight"].to_numpy()
        diag_var = float(np.sum((w_vals ** 2) * (rmse_vals ** 2)))

        if rmse_vals.size > 1 and assumed_correlation > 0:
            outer = np.outer(w_vals * rmse_vals, w_vals * rmse_vals)
            cross_terms = float(outer.sum() - np.sum((w_vals ** 2) * (rmse_vals ** 2)))
            ensemble_var = diag_var + assumed_correlation * cross_terms
        else:
            ensemble_var = diag_var

        sigma_lookup[(ticker, freq)] = float(np.sqrt(ensemble_var))
    return sigma_lookup


def build_ensemble_performance_summary(metrics_all: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for (ticker, freq), m_group in metrics_all.groupby(["Ticker", "frequency"]):
        for _, row in m_group.iterrows():
            weight_series = weights_df[
                (weights_df["Ticker"] == ticker)
                & (weights_df["Frequency"] == freq)
                & (weights_df["Model"] == row["Model"])
            ]["Weight"]
            weight_val = float(weight_series.iloc[0]) if not weight_series.empty else np.nan
            rows.append({
                "Ticker": ticker,
                "Frequency": freq,
                "Model": row["Model"],
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
                "R2": row["R2"],
                "DirectionAccuracy": row["DirectionAccuracy"],
                "DirectionPrecision": row["DirectionPrecision"],
                "DirectionRecall": row["DirectionRecall"],
                "DirectionMetricCoverage": float(pd.notna(row[DIRECTIONAL_COLS]).mean()),
                "Weight": float(weight_val) if pd.notna(weight_val) else np.nan,
                "IsEnsemble": False,
            })

        w_group = weights_df[(weights_df["Ticker"] == ticker) & (weights_df["Frequency"] == freq)]
        if w_group.empty:
            continue

        merged = w_group.merge(
            m_group[["Ticker", "frequency", "Model", "RMSE", "MAE"]],
            left_on=["Ticker", "Frequency", "Model"],
            right_on=["Ticker", "frequency", "Model"],
            how="left",
        ).dropna(subset=["RMSE", "MAE"])

        if merged.empty:
            continue

        w_vals = merged["Weight"].to_numpy()
        rmse_vals = merged["RMSE"].to_numpy()
        mae_vals = merged["MAE"].to_numpy()

        ensemble_rmse = float(np.sqrt(np.sum((w_vals ** 2) * (rmse_vals ** 2))))
        ensemble_mae = float(np.sum(w_vals * mae_vals))

        dir_metrics: Dict[str, float] = {}
        coverage_values: List[float] = []
        for col in DIRECTIONAL_COLS:
            available_mask = merged[col].notna().to_numpy()
            coverage_values.append(float(available_mask.mean()))
            if available_mask.any():
                weights_subset = w_vals[available_mask]
                metrics_subset = merged.loc[available_mask, col].to_numpy()
                dir_metrics[col] = float(np.sum((weights_subset / weights_subset.sum()) * metrics_subset))
            else:
                dir_metrics[col] = np.nan

        rows.append({
            "Ticker": ticker,
            "Frequency": freq,
            "Model": "Ensemble",
            "MAE": ensemble_mae,
            "RMSE": ensemble_rmse,
            "R2": np.nan,
            "DirectionAccuracy": dir_metrics["DirectionAccuracy"],
            "DirectionPrecision": dir_metrics["DirectionPrecision"],
            "DirectionRecall": dir_metrics["DirectionRecall"],
            "DirectionMetricCoverage": float(np.mean(coverage_values)) if coverage_values else np.nan,
            "Weight": 1.0,
            "IsEnsemble": True,
        })

    return pd.DataFrame(rows)


# ============================================================
# ENSEMBLE (DAILY / MONTHLY)
# ============================================================


def ensemble_for_frequency(
    freq: str,
    files: InputFiles,
    *,
    metric_recency_days: Optional[int],
    assumed_correlation: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    freq = _normalize_frequency(freq)
    lstm_fc = load_lstm_forecast(freq, files)
    sarima_fc = load_sarimax_forecast(freq, files)
    gru_fc = load_gru_forecast(freq, files)

    forecasts = pd.concat([lstm_fc, sarima_fc, gru_fc], ignore_index=True)
    metrics_all = load_all_metrics(freq, files)
    warn_on_missing_direction_metrics(metrics_all, freq)
    warn_on_nonpositive_metrics(metrics_all, freq)
    warn_on_metric_coverage(forecasts, metrics_all, freq)
    validate_metric_recency(metrics_all, max_age_days=metric_recency_days, context=f"{freq} metrics")

    weights_df = build_weight_table_from_metrics(metrics_all)
    sigma_lookup = compute_ensemble_sigma_lookup(metrics_all, weights_df, assumed_correlation=assumed_correlation)

    weight_pairs = set(zip(weights_df["Ticker"], weights_df["Frequency"]))
    forecast_pairs = set(zip(forecasts["Ticker"], forecasts["Frequency"]))
    missing_weights = sorted(forecast_pairs - weight_pairs)
    if missing_weights:
        LOGGER.warning(
            "No RMSE weights for %d ticker/frequency pairs: %s",
            len(missing_weights),
            "; ".join(f"{t} ({f})" for t, f in missing_weights),
        )

    forecasts_w = forecasts.merge(weights_df, on=["Ticker", "Frequency", "Model"], how="left")

    def combine_group(group: pd.DataFrame) -> pd.Series:
        preds = group["PredictedPrice"].to_numpy()
        models = group["Model"].to_numpy()
        weights = group["Weight"].to_numpy()

        valid_mask = ~np.isnan(preds)
        preds = preds[valid_mask]
        models = models[valid_mask]
        weights = weights[valid_mask]

        ticker = group["Ticker"].iloc[0]
        frequency = group["Frequency"].iloc[0]

        if preds.size == 0:
            return pd.Series({"FinalPredictedPrice": np.nan, "LowerCI": np.nan, "UpperCI": np.nan, "ModelsUsed": ""})

        if weights.size == 0 or np.all(np.isnan(weights)):
            w = np.full_like(preds, 1 / preds.size, dtype=float)
        else:
            weights = np.where(np.isnan(weights), 0.0, weights)
            w = weights / weights.sum() if weights.sum() else np.full_like(preds, 1 / preds.size, dtype=float)

        final_pred = float(np.sum(w * preds))
        sigma = sigma_lookup.get((ticker, frequency), np.nan)
        if np.isnan(sigma):
            lower_ci = upper_ci = np.nan
        else:
            ci_mult = 1.96
            lower_ci = final_pred - ci_mult * sigma
            upper_ci = final_pred + ci_mult * sigma

        return pd.Series({
            "FinalPredictedPrice": final_pred,
            "LowerCI": lower_ci,
            "UpperCI": upper_ci,
            "ModelsUsed": ",".join(models),
        })

    grouped = forecasts_w.groupby(["Ticker", "Date", "Frequency"], as_index=False)
    ensemble_df = grouped.apply(combine_group).reset_index(drop=True)
    return ensemble_df, metrics_all, weights_df


# ============================================================
# VISUALIZATION: PERFORMANCE & PRICE PATHS
# ============================================================


def plot_rmse_bar_chart(summary_df: pd.DataFrame, freq: str) -> None:
    df = summary_df[summary_df["Frequency"] == _normalize_frequency(freq)].copy()
    if df.empty:
        LOGGER.info("No RMSE data to plot for %s", freq)
        return

    df["Label"] = df["Ticker"] + " - " + df["Model"]
    plt.figure(figsize=(10, 5))
    plt.bar(df["Label"], df["RMSE"])
    plt.title(f"RMSE by Ticker and Model ({freq.capitalize()})")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def fetch_price_history(
    ticker: str,
    *,
    history_start: Optional[pd.Timestamp],
    cache_dir: Optional[Path],
    allow_network: bool,
) -> pd.DataFrame:
    """Fetch historical prices with optional caching and date bounding."""

    cache_path = cache_dir / f"{ticker}.csv" if cache_dir else None
    if cache_path and cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if history_start is not None:
                cached = cached[cached.index >= history_start]
            return cached
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to read cache for %s: %s", ticker, exc)

    if not allow_network:
        LOGGER.info("Skipping download for %s because allow_network=False and cache missing", ticker)
        return pd.DataFrame()

    download_kwargs = {"period": "max", "auto_adjust": True, "progress": False}
    if history_start is not None:
        download_kwargs["start"] = history_start

    hist = yf.download(ticker, **download_kwargs)
    if hist.empty:
        return pd.DataFrame()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(cache_path)

    return hist


def plot_actual_vs_ensemble(
    ensemble_df: pd.DataFrame,
    freq: str,
    *,
    max_tickers: int = 3,
    allow_network: bool = True,
    history_start: Optional[pd.Timestamp] = None,
    cache_dir: Optional[Path] = None,
) -> None:
    if not allow_network and cache_dir is None:
        LOGGER.info("Skipping Yahoo Finance downloads because allow_network=False and no cache provided")
        return

    tickers = ensemble_df["Ticker"].unique()[:max_tickers]
    for ticker in tickers:
        df_pred = ensemble_df[ensemble_df["Ticker"] == ticker].sort_values("Date")
        if df_pred.empty:
            continue
        forecast_start = df_pred["Date"].min()

        hist = fetch_price_history(ticker, history_start=history_start, cache_dir=cache_dir, allow_network=allow_network)
        if hist.empty:
            LOGGER.warning("No historical data available for %s", ticker)
            continue

        price_col = "Close" if "Close" in hist.columns else hist.columns[0]
        hist_price = hist[[price_col]].rename(columns={price_col: "ActualPrice"})
        hist_segment = hist_price[hist_price.index <= forecast_start]

        plt.figure(figsize=(12, 5))
        plt.plot(hist_segment.index, hist_segment["ActualPrice"], label="Actual price (historical)")
        plt.plot(df_pred["Date"], df_pred["FinalPredictedPrice"], label="Ensemble forecast", linestyle="--")
        if {"LowerCI", "UpperCI"}.issubset(df_pred.columns):
            plt.fill_between(df_pred["Date"], df_pred["LowerCI"], df_pred["UpperCI"], alpha=0.2, label="Approx. 95% CI")
        plt.axvline(forecast_start, linestyle=":", label="Forecast start")
        plt.title(f"{ticker} - {freq.capitalize()} actual vs ensemble forecast")
        plt.xlabel("Timestamp")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ============================================================
# MAIN DRIVER & CLI
# ============================================================


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Saved %s rows to %s", len(df), path.resolve())


def run_pipeline(
    frequencies: Sequence[str],
    files: InputFiles,
    *,
    output_dir: Path,
    plot: bool,
    max_tickers: int,
    allow_network: bool,
    metric_recency_days: Optional[int],
    assumed_correlation: float,
    history_start: Optional[pd.Timestamp],
    history_cache_dir: Optional[Path],
) -> None:
    validate_required_files(files, frequencies)
    all_summaries: List[pd.DataFrame] = []
    for freq in frequencies:
        LOGGER.info("Building ensemble forecast for %s", freq)
        ensemble_df, metrics_all, weights_df = ensemble_for_frequency(
            freq,
            files,
            metric_recency_days=metric_recency_days,
            assumed_correlation=assumed_correlation,
        )
        freq_dir = output_dir

        out_path = freq_dir / f"ensemble_20y_{freq}_forecast.csv"
        _save_dataframe(ensemble_df, out_path)

        summary_df = build_ensemble_performance_summary(metrics_all, weights_df)
        all_summaries.append(summary_df)

        if plot:
            plot_rmse_bar_chart(summary_df, freq=freq)
            plot_actual_vs_ensemble(
                ensemble_df,
                freq=freq,
                max_tickers=max_tickers,
                allow_network=allow_network,
                history_start=history_start,
                cache_dir=history_cache_dir,
            )

    if all_summaries:
        performance_summary = pd.concat(all_summaries, ignore_index=True)
        _save_dataframe(performance_summary, output_dir / "ensemble_performance_summary.csv")
    else:
        LOGGER.warning("No performance summaries generated; check input data.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weighted ensemble forecasts with optional plots.")
    parser.add_argument("--frequencies", nargs="+", default=["daily", "monthly"], help="Frequencies to process: daily monthly")
    parser.add_argument("--output-dir", type=Path, default=Path("forecasts"), help="Directory to write ensemble outputs.")
    parser.add_argument("--plot", action="store_true", help="Generate RMSE and price-path plots.")
    parser.add_argument("--max-tickers", type=int, default=3, help="Max tickers to plot when --plot is set.")
    parser.add_argument("--allow-network", action="store_true", help="Allow network calls (Yahoo Finance) when plotting.")
    parser.add_argument("--history-start", type=str, default=None, help="Optional YYYY-MM-DD start date for historical downloads.")
    parser.add_argument("--history-cache-dir", type=Path, default=None, help="Directory for caching Yahoo Finance history to avoid repeat downloads.")
    parser.add_argument("--metric-recency-days", type=int, default=None, help="Warn when metrics are older than this many days (requires date columns).")
    parser.add_argument(
        "--ci-correlation",
        type=float,
        default=0.5,
        help="Assumed pairwise correlation (0-1) between model errors when estimating CI; higher widens intervals.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s:%(name)s:%(message)s")

    files = InputFiles()
    frequencies = [_normalize_frequency(f) for f in args.frequencies]
    history_start = pd.to_datetime(args.history_start).tz_localize(None) if args.history_start else None

    try:
        run_pipeline(
            frequencies,
            files,
            output_dir=args.output_dir,
            plot=args.plot,
            max_tickers=args.max_tickers,
            allow_network=args.allow_network,
            metric_recency_days=args.metric_recency_days,
            assumed_correlation=args.ci_correlation,
            history_start=history_start,
            history_cache_dir=args.history_cache_dir,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
