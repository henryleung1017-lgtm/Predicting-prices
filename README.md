# Predicting-prices
## Purpose of `multi_model_ensemble_20y_with_metrics.py`
This repository tracks a single orchestration script that combines multiple price-forecasting models into a publishable ensemble.

- Combines 20-year forecasts from LSTM, SARIMAX, and GRU models (daily and monthly) into a weighted ensemble per ticker and date.
- Validates required forecast/metric CSVs up front, including column-level schema and numeric checks, and warns when metrics are missing for any ticker/frequency or contain non-positive errors.
- Derives weights from RMSE metrics, defaulting to equal weights when RMSE data is missing, and surfaces directional metric coverage per row so unsupported tickers/frequencies are explicit.
- Approximates ensemble uncertainty using weighted RMSE with a configurable error-correlation assumption and optional empirical correlation backtests derived from residual CSVs to avoid understating confidence intervals.
- Saves ensemble forecasts and performance summaries to `forecasts/` and produces RMSE bar charts plus plots comparing historical Yahoo Finance prices with ensemble predictions. Plotting can optionally bound/download history by date, clamp downloads to the latest train/test dates in the metrics, enforce lookback windows, parallelize Yahoo downloads, and cache Yahoo responses to minimize repeated network calls.

### Where to copy the code
The full, copy/pasteable pipeline lives in `multi_model_ensemble_20y_with_metrics.py` at the repository root.

## How to run the ensemble pipeline
1. Place the required forecast and metric CSVs in the paths expected by the script (see `InputFiles` defaults inside `multi_model_ensemble_20y_with_metrics.py`).
2. Execute the pipeline (daily and monthly) and write outputs to `forecasts/`:
   ```bash
   python multi_model_ensemble_20y_with_metrics.py
   ```
3. To enable plotting and limit the number of tickers shown:
   ```bash
   python multi_model_ensemble_20y_with_metrics.py --plot --max-tickers 2 --allow-network --history-start 2010-01-01 --history-cache-dir .cache/yahoo
   ```
4. Use `--frequencies` to process a subset (e.g., daily only) and `--output-dir` to redirect outputs:
   ```bash
   python multi_model_ensemble_20y_with_metrics.py --frequencies daily --output-dir artifacts/
   ```
5. To widen confidence intervals when you suspect correlated model errors and to warn on stale metrics (when date columns exist):
   ```bash
   python multi_model_ensemble_20y_with_metrics.py --ci-correlation 0.7 --metric-recency-days 90
   ```
6. To use empirically estimated error correlations, backtested residual correlations, and bound historical downloads to training spans with lookback/parallel downloads:
   ```bash
   python multi_model_ensemble_20y_with_metrics.py \
     --correlation-matrix correlations.csv \
     --residuals-daily forecasts/residuals_daily.csv \
     --history-end-from-metrics \
     --history-lookback-days 365 \
     --history-workers 4
   ```
