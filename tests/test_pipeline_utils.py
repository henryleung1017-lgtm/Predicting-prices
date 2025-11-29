import pandas as pd
import numpy as np
import pytest

from multi_model_ensemble_20y_with_metrics import (
    _validate_numeric_columns,
    build_history_end_lookup,
    build_weight_table_from_metrics,
    compute_ensemble_sigma_lookup,
    build_ensemble_performance_summary,
    estimate_pairwise_correlations_from_residuals,
    fetch_price_history,
)


def test_validate_numeric_columns_rejects_negative():
    df = pd.DataFrame({"A": [1, -2, 3]})
    try:
        _validate_numeric_columns(df, ["A"], context="test")
    except ValueError as exc:
        assert "Negative values" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative numeric column")


def test_compute_sigma_with_correlation_matrix():
    metrics_all = pd.DataFrame(
        {
            "Ticker": ["ABC", "ABC"],
            "frequency": ["daily", "daily"],
            "Model": ["A", "B"],
            "RMSE": [2.0, 4.0],
            "MAE": [1.0, 2.0],
        }
    )
    weights_df = build_weight_table_from_metrics(metrics_all)
    correlation_lookup = {("A", "B"): 1.0}
    sigma_lookup = compute_ensemble_sigma_lookup(
        metrics_all,
        weights_df,
        assumed_correlation=0.0,
        correlation_lookup=correlation_lookup,
    )
    assert sigma_lookup[("ABC", "daily")] == pytest.approx(3.0)


def test_history_end_lookup_uses_latest_metric_date():
    metrics_all = pd.DataFrame(
        {
            "Ticker": ["ABC", "ABC", "XYZ"],
            "frequency": ["daily", "daily", "monthly"],
            "TrainEnd": ["2020-01-01", "2021-06-01", "2019-12-31"],
            "TestEnd": ["2020-06-01", None, "2020-01-15"],
        }
    )
    lookup = build_history_end_lookup(metrics_all)
    assert pd.Timestamp("2021-06-01") == lookup[("ABC", "daily")]
    assert pd.Timestamp("2020-01-15") == lookup[("XYZ", "monthly")]


def test_directional_coverage_flagged_in_summary():
    metrics_all = pd.DataFrame(
        {
            "Ticker": ["ABC", "ABC"],
            "frequency": ["daily", "daily"],
            "Model": ["A", "B"],
            "MAE": [1.0, 2.0],
            "RMSE": [1.5, 2.5],
            "R2": [0.8, 0.7],
            "DirectionAccuracy": [np.nan, 0.6],
            "DirectionPrecision": [np.nan, 0.5],
            "DirectionRecall": [np.nan, 0.4],
        }
    )
    weights_df = build_weight_table_from_metrics(metrics_all)
    summary = build_ensemble_performance_summary(metrics_all, weights_df)
    ensemble_row = summary[(summary["Model"] == "Ensemble") & (summary["Ticker"] == "ABC")].iloc[0]
    assert ensemble_row["DirectionMetricCoverage"] < 1.0
    assert summary[summary["Model"] == "A"]["DirectionalSupported"].iloc[0] is False


def test_empirical_correlation_estimation_clips_and_averages():
    residuals = pd.DataFrame(
        {
            "Ticker": ["ABC"] * 4,
            "Date": pd.date_range("2020-01-01", periods=4, freq="D"),
            "Model": ["A", "B", "A", "B"],
            "Residual": [1.0, 1.0, -1.0, -1.0],
        }
    )
    lookup = estimate_pairwise_correlations_from_residuals(residuals)
    assert lookup[("A", "B")] == 1.0


def test_fetch_price_history_validates_lookback_bounds(monkeypatch):
    # No network calls because allow_network=False and no cache
    with pytest.raises(ValueError):
        fetch_price_history(
            "TEST",
            history_start=pd.Timestamp("2021-01-01"),
            history_end=pd.Timestamp("2020-01-01"),
            cache_dir=None,
            allow_network=False,
            lookback_days=None,
            forecast_start=None,
        )
