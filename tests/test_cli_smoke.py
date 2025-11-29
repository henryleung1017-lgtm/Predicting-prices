import subprocess
import sys
from pathlib import Path

def test_cli_smoke_runs_with_temporary_inputs(tmp_path):
    forecasts_dir = tmp_path / "forecasts"
    forecasts_dir.mkdir(parents=True)

    # Forecast files
    daily_forecasts = [
        (forecasts_dir / "all_tickers_20y_daily_forecast.csv", "LSTM"),
        (forecasts_dir / "gru_20y_daily_forecast.csv", "GRU"),
    ]
    sarimax_daily = tmp_path / "sarimax_all_tickers_20y_daily_forecast.csv"
    for path, model in daily_forecasts:
        path.write_text("Date,Ticker,PredictedPrice\n2020-01-01,ABC,10.0\n")
    sarimax_daily.write_text("Date,Ticker,PredictedPrice\n2020-01-01,ABC,12.0\n")

    # Metric files
    (forecasts_dir / "all_tickers_daily_metrics.csv").write_text(
        "Ticker,Mode,RMSE,MAE\nABC,daily,1.0,0.5\n"
    )
    (forecasts_dir / "gru_performance_summary.csv").write_text(
        "ticker,frequency,RMSE,MAE,R2,direction_accuracy,direction_precision,direction_recall\nABC,daily,1.2,0.6,0.1,0.5,0.4,0.3\n"
    )
    (tmp_path / "sarimax_performance_summary.csv").write_text(
        "Ticker,Frequency,RMSE,MAE,R2,DirectionAccuracy,DirectionPrecision,DirectionRecall\nABC,daily,1.4,0.7,0.2,0.6,0.5,0.4\n"
    )

    # Residuals for empirical correlation
    residuals_daily = tmp_path / "residuals_daily.csv"
    residuals_daily.write_text(
        "Ticker,Date,Model,Residual\nABC,2020-01-01,LSTM,1\nABC,2020-01-01,GRU,1\nABC,2020-01-02,LSTM,-1\nABC,2020-01-02,GRU,-1\n"
    )

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "multi_model_ensemble_20y_with_metrics.py"),
        "--frequencies",
        "daily",
        "--output-dir",
        str(forecasts_dir),
        "--input-root",
        str(tmp_path),
        "--residuals-daily",
        str(residuals_daily),
    ]

    subprocess.run(cmd, check=True)

    assert (forecasts_dir / "ensemble_20y_daily_forecast.csv").exists()
    assert (forecasts_dir / "ensemble_performance_summary.csv").exists()
