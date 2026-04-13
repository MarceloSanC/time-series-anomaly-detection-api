from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for synthetic concept-drift analysis."""

    sampling_hz: int = 1
    training_hours: int = 2
    evaluation_hours: int = 6
    rolling_window: int = 300
    instability_fpr_threshold: float = 0.05
    random_seed: int = 42
    base_mean: float = 50.0
    base_std: float = 1.0
    drift_start_hour: int = 1
    drift_total_shift: float = 8.0
    anomaly_probability: float = 0.005
    anomaly_magnitude: float = 8.0


def _simulate_training_series(config: AnalysisConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate stationary baseline training data (healthy regime)."""
    n_train = config.training_hours * 3600 * config.sampling_hz
    return rng.normal(loc=config.base_mean, scale=config.base_std, size=n_train).astype(float)


def _simulate_evaluation_stream(
    config: AnalysisConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate drifting stream plus anomaly labels for evaluation."""
    n_eval = config.evaluation_hours * 3600 * config.sampling_hz
    time_index = np.arange(n_eval, dtype=float)
    drift_start = config.drift_start_hour * 3600 * config.sampling_hz

    drift = np.zeros(n_eval, dtype=float)
    if drift_start < n_eval:
        drift[drift_start:] = np.linspace(0.0, config.drift_total_shift, n_eval - int(drift_start), dtype=float)

    noise = rng.normal(loc=0.0, scale=config.base_std, size=n_eval).astype(float)
    values = config.base_mean + drift + noise

    anomaly_mask = rng.random(n_eval) < config.anomaly_probability
    values = values + anomaly_mask.astype(float) * config.anomaly_magnitude

    _ = time_index  # reserved for future timestamp-aware extensions
    return values, anomaly_mask


def _baseline_thresholds(train_values: np.ndarray, n_eval: int) -> np.ndarray:
    """Build constant threshold array using train mean/std."""
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    threshold = mean + 3.0 * std
    return np.full(shape=n_eval, fill_value=threshold, dtype=float)


def _rolling_thresholds(
    train_values: np.ndarray,
    eval_values: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Build causal rolling thresholds (mean + 3*std) for each eval point."""
    if window_size <= 1:
        raise ValueError("rolling_window must be greater than 1")

    seed_window = deque(train_values[-window_size:].tolist(), maxlen=window_size)
    thresholds = np.zeros(len(eval_values), dtype=float)

    for idx, value in enumerate(eval_values):
        window_arr = np.array(seed_window, dtype=float)
        mean = float(np.mean(window_arr))
        std = float(np.std(window_arr))
        thresholds[idx] = mean + 3.0 * std
        seed_window.append(float(value))

    return thresholds


def _alerts(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return values > thresholds


def _false_positive_rate(alerts: np.ndarray, anomaly_mask: np.ndarray) -> float:
    normal_mask = ~anomaly_mask
    normal_count = int(np.sum(normal_mask))
    if normal_count == 0:
        return 0.0
    false_positives = int(np.sum(alerts & normal_mask))
    return float(false_positives / normal_count)


def _alerts_per_hour(alerts: np.ndarray, total_hours: int) -> float:
    if total_hours <= 0:
        return 0.0
    return float(np.sum(alerts) / total_hours)


def _threshold_stability(thresholds: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(thresholds)),
        "std": float(np.std(thresholds)),
        "min": float(np.min(thresholds)),
        "max": float(np.max(thresholds)),
        "range": float(np.max(thresholds) - np.min(thresholds)),
    }


def _time_to_instability_hours(
    alerts: np.ndarray,
    anomaly_mask: np.ndarray,
    sampling_hz: int,
    fpr_limit: float,
) -> float | None:
    """Return first hour where windowed FPR exceeds limit; None when stable."""
    points_per_hour = 3600 * sampling_hz
    if points_per_hour <= 0:
        return None

    n = len(alerts)
    hour_idx = 0
    while hour_idx * points_per_hour < n:
        start = hour_idx * points_per_hour
        end = min((hour_idx + 1) * points_per_hour, n)
        hour_alerts = alerts[start:end]
        hour_anomaly = anomaly_mask[start:end]
        hour_fpr = _false_positive_rate(hour_alerts, hour_anomaly)
        if hour_fpr > fpr_limit:
            return float(hour_idx + 1)
        hour_idx += 1
    return None


def _hourly_false_positive_rates(
    alerts: np.ndarray,
    anomaly_mask: np.ndarray,
    sampling_hz: int,
) -> list[float]:
    """Return per-hour FPR slices over the evaluation window."""
    points_per_hour = 3600 * sampling_hz
    if points_per_hour <= 0:
        return []

    rates: list[float] = []
    n = len(alerts)
    hour_idx = 0
    while hour_idx * points_per_hour < n:
        start = hour_idx * points_per_hour
        end = min((hour_idx + 1) * points_per_hour, n)
        hour_alerts = alerts[start:end]
        hour_anomaly = anomaly_mask[start:end]
        rates.append(_false_positive_rate(hour_alerts, hour_anomaly))
        hour_idx += 1
    return rates


def _save_plots(
    output_dir: Path,
    eval_values: np.ndarray,
    baseline_thresholds: np.ndarray,
    rolling_thresholds: np.ndarray,
    baseline_alerts: np.ndarray,
    rolling_alerts: np.ndarray,
    anomaly_mask: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, str]:
    """Save visual drift-analysis artifacts and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    points_per_hour = 3600 * config.sampling_hz
    x_hours = np.arange(len(eval_values), dtype=float) / float(points_per_hour)

    # Time-series panel with both thresholds and alert overlays.
    fig_ts, ax_ts = plt.subplots(figsize=(12, 4.5))
    ax_ts.plot(x_hours, eval_values, color="#4c78a8", linewidth=1.0, alpha=0.8, label="evaluation value")
    ax_ts.plot(
        x_hours,
        baseline_thresholds,
        color="#e45756",
        linestyle="--",
        linewidth=1.5,
        label="baseline threshold",
    )
    ax_ts.plot(
        x_hours,
        rolling_thresholds,
        color="#72b7b2",
        linestyle="-",
        linewidth=1.4,
        label="rolling threshold",
    )
    ax_ts.scatter(
        x_hours[baseline_alerts],
        eval_values[baseline_alerts],
        color="#e45756",
        s=7,
        alpha=0.35,
        label="baseline alerts",
    )
    ax_ts.scatter(
        x_hours[rolling_alerts],
        eval_values[rolling_alerts],
        color="#72b7b2",
        s=7,
        alpha=0.35,
        label="rolling alerts",
    )
    ax_ts.scatter(
        x_hours[anomaly_mask],
        eval_values[anomaly_mask],
        color="#f58518",
        s=10,
        alpha=0.55,
        label="injected anomalies",
    )
    ax_ts.set_title("Concept Drift Stream: Values, Thresholds, and Alerts")
    ax_ts.set_xlabel("Evaluation Time (hours)")
    ax_ts.set_ylabel("Value")
    ax_ts.grid(alpha=0.25)
    ax_ts.legend(loc="upper left", ncol=2, fontsize=8)
    fig_ts.tight_layout()
    ts_path = output_dir / "drift_analysis_timeseries.png"
    fig_ts.savefig(ts_path, dpi=150)
    plt.close(fig_ts)

    # Per-hour FPR comparison panel.
    baseline_fpr_hourly = _hourly_false_positive_rates(
        alerts=baseline_alerts,
        anomaly_mask=anomaly_mask,
        sampling_hz=config.sampling_hz,
    )
    rolling_fpr_hourly = _hourly_false_positive_rates(
        alerts=rolling_alerts,
        anomaly_mask=anomaly_mask,
        sampling_hz=config.sampling_hz,
    )
    hours = np.arange(1, len(baseline_fpr_hourly) + 1)
    bar_width = 0.38

    fig_fpr, ax_fpr = plt.subplots(figsize=(10, 4.2))
    ax_fpr.bar(hours - bar_width / 2.0, baseline_fpr_hourly, width=bar_width, color="#e45756", label="baseline")
    ax_fpr.bar(hours + bar_width / 2.0, rolling_fpr_hourly, width=bar_width, color="#72b7b2", label="rolling")
    ax_fpr.axhline(
        y=config.instability_fpr_threshold,
        color="#444444",
        linestyle=":",
        linewidth=1.3,
        label=f"instability threshold ({config.instability_fpr_threshold:.0%})",
    )
    ax_fpr.set_title("Hourly False Positive Rate Under Drift")
    ax_fpr.set_xlabel("Evaluation Hour")
    ax_fpr.set_ylabel("False Positive Rate")
    ax_fpr.set_xticks(hours)
    ax_fpr.set_ylim(bottom=0.0)
    ax_fpr.grid(axis="y", alpha=0.25)
    ax_fpr.legend(loc="upper left", fontsize=8)

    # Numeric annotations to make low rolling FPR values visible at a glance.
    baseline_max = max(baseline_fpr_hourly) if baseline_fpr_hourly else 0.0
    for idx, hour in enumerate(hours):
        base_val = baseline_fpr_hourly[idx]
        roll_val = rolling_fpr_hourly[idx]
        ax_fpr.text(
            hour - bar_width / 2.0,
            base_val + (0.008 * max(baseline_max, 0.01)),
            f"{base_val * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#7f1d1d",
        )
        ax_fpr.text(
            hour + bar_width / 2.0,
            roll_val + (0.002 * max(baseline_max, 0.01)),
            f"{roll_val * 100:.3f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#134e4a",
        )

    fig_fpr.tight_layout()
    fpr_path = output_dir / "drift_analysis_fpr_hourly.png"
    fig_fpr.savefig(fpr_path, dpi=150)
    plt.close(fig_fpr)

    # One-sided detection illustration: lower-tail anomalies are missed by design.
    demo_x = np.arange(10, dtype=float)
    mean_eval = float(np.mean(eval_values))
    std_eval = float(np.std(eval_values))
    upper = mean_eval + 3.0 * std_eval
    lower = mean_eval - 3.0 * std_eval
    demo_values = np.array(
        [
            mean_eval - 4.2 * std_eval,  # lower-tail extreme (missed by one-sided)
            mean_eval - 3.5 * std_eval,  # lower-tail extreme (missed by one-sided)
            mean_eval - 1.0 * std_eval,
            mean_eval + 0.2 * std_eval,
            mean_eval + 1.2 * std_eval,
            mean_eval + 3.4 * std_eval,  # upper-tail anomaly (detected)
            mean_eval + 4.1 * std_eval,  # upper-tail anomaly (detected)
            mean_eval + 0.0 * std_eval,
            mean_eval - 0.6 * std_eval,
            mean_eval + 0.8 * std_eval,
        ],
        dtype=float,
    )
    detected_by_current = demo_values > upper
    lower_tail_extreme = demo_values < lower

    fig_one, ax_one = plt.subplots(figsize=(10.5, 4.2))
    ax_one.scatter(demo_x, demo_values, color="#4c78a8", s=36, label="sample points")
    ax_one.scatter(
        demo_x[detected_by_current],
        demo_values[detected_by_current],
        color="#e45756",
        s=55,
        label="detected by current rule (value > mean+3std)",
    )
    ax_one.scatter(
        demo_x[lower_tail_extreme],
        demo_values[lower_tail_extreme],
        facecolors="none",
        edgecolors="#f58518",
        linewidths=1.6,
        s=80,
        label="lower-tail extreme (missed by current rule)",
    )
    ax_one.axhline(mean_eval, color="#666666", linestyle="-", linewidth=1.1, label="mean")
    ax_one.axhline(upper, color="#e45756", linestyle="--", linewidth=1.3, label="upper bound (+3std)")
    ax_one.axhline(lower, color="#f58518", linestyle="--", linewidth=1.3, label="lower bound (-3std)")
    ax_one.set_title("One-Sided Baseline: Lower-Tail Extremes Are Not Flagged")
    ax_one.set_xlabel("Demo Point Index")
    ax_one.set_ylabel("Value")
    ax_one.grid(alpha=0.25)
    ax_one.legend(loc="upper right", fontsize=8)
    fig_one.tight_layout()
    one_sided_path = output_dir / "drift_analysis_one_sided_demo.png"
    fig_one.savefig(one_sided_path, dpi=150)
    plt.close(fig_one)

    return {
        "timeseries_plot": str(ts_path.as_posix()),
        "hourly_fpr_plot": str(fpr_path.as_posix()),
        "one_sided_demo_plot": str(one_sided_path.as_posix()),
    }


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    """Run synthetic drift analysis for baseline vs rolling detectors."""
    rng = np.random.default_rng(config.random_seed)
    train_values = _simulate_training_series(config=config, rng=rng)
    eval_values, anomaly_mask = _simulate_evaluation_stream(config=config, rng=rng)

    baseline_thresholds = _baseline_thresholds(train_values=train_values, n_eval=len(eval_values))
    rolling_thresholds = _rolling_thresholds(
        train_values=train_values,
        eval_values=eval_values,
        window_size=config.rolling_window,
    )

    baseline_alerts = _alerts(eval_values, baseline_thresholds)
    rolling_alerts = _alerts(eval_values, rolling_thresholds)

    baseline_fpr = _false_positive_rate(baseline_alerts, anomaly_mask)
    rolling_fpr = _false_positive_rate(rolling_alerts, anomaly_mask)

    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "config": {
            "sampling_hz": config.sampling_hz,
            "training_hours": config.training_hours,
            "evaluation_hours": config.evaluation_hours,
            "rolling_window": config.rolling_window,
            "instability_fpr_threshold": config.instability_fpr_threshold,
            "random_seed": config.random_seed,
            "base_mean": config.base_mean,
            "base_std": config.base_std,
            "drift_start_hour": config.drift_start_hour,
            "drift_total_shift": config.drift_total_shift,
            "anomaly_probability": config.anomaly_probability,
            "anomaly_magnitude": config.anomaly_magnitude,
        },
        "workload": {
            "training_points": int(len(train_values)),
            "evaluation_points": int(len(eval_values)),
        },
        "detectors": {
            "baseline": {
                "fpr": baseline_fpr,
                "alerts_per_hour": _alerts_per_hour(baseline_alerts, config.evaluation_hours),
                "threshold_stability": _threshold_stability(baseline_thresholds),
                "time_to_instability_hours": _time_to_instability_hours(
                    baseline_alerts,
                    anomaly_mask,
                    sampling_hz=config.sampling_hz,
                    fpr_limit=config.instability_fpr_threshold,
                ),
            },
            "rolling": {
                "fpr": rolling_fpr,
                "alerts_per_hour": _alerts_per_hour(rolling_alerts, config.evaluation_hours),
                "threshold_stability": _threshold_stability(rolling_thresholds),
                "time_to_instability_hours": _time_to_instability_hours(
                    rolling_alerts,
                    anomaly_mask,
                    sampling_hz=config.sampling_hz,
                    fpr_limit=config.instability_fpr_threshold,
                ),
            },
        },
        "comparison": {
            "fpr_relative_change_percent": float(
                ((baseline_fpr - rolling_fpr) / rolling_fpr) * 100.0 if rolling_fpr > 0 else 0.0
            ),
            "alerts_per_hour_delta": float(
                _alerts_per_hour(baseline_alerts, config.evaluation_hours)
                - _alerts_per_hour(rolling_alerts, config.evaluation_hours)
            ),
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic concept-drift analysis baseline vs rolling.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/drift_analysis_results.json"),
        help="Path to save drift analysis results.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("docs/assets"),
        help="Directory to save generated drift plots.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visual artifacts (timeseries + hourly FPR) in --plot-dir.",
    )
    parser.add_argument("--sampling-hz", type=int, default=1)
    parser.add_argument("--training-hours", type=int, default=2)
    parser.add_argument("--evaluation-hours", type=int, default=6)
    parser.add_argument("--rolling-window", type=int, default=300)
    parser.add_argument("--instability-fpr-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AnalysisConfig(
        sampling_hz=args.sampling_hz,
        training_hours=args.training_hours,
        evaluation_hours=args.evaluation_hours,
        rolling_window=args.rolling_window,
        instability_fpr_threshold=args.instability_fpr_threshold,
        random_seed=args.seed,
    )

    if config.training_hours <= 0 or config.evaluation_hours <= 0:
        raise SystemExit("training_hours and evaluation_hours must be > 0")
    if config.sampling_hz <= 0:
        raise SystemExit("sampling_hz must be > 0")
    if config.rolling_window <= 1:
        raise SystemExit("rolling_window must be > 1")

    rng = np.random.default_rng(config.random_seed)
    train_values = _simulate_training_series(config=config, rng=rng)
    eval_values, anomaly_mask = _simulate_evaluation_stream(config=config, rng=rng)

    baseline_thresholds = _baseline_thresholds(train_values=train_values, n_eval=len(eval_values))
    rolling_thresholds = _rolling_thresholds(
        train_values=train_values,
        eval_values=eval_values,
        window_size=config.rolling_window,
    )
    baseline_alerts = _alerts(eval_values, baseline_thresholds)
    rolling_alerts = _alerts(eval_values, rolling_thresholds)

    baseline_fpr = _false_positive_rate(baseline_alerts, anomaly_mask)
    rolling_fpr = _false_positive_rate(rolling_alerts, anomaly_mask)
    result = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "config": {
            "sampling_hz": config.sampling_hz,
            "training_hours": config.training_hours,
            "evaluation_hours": config.evaluation_hours,
            "rolling_window": config.rolling_window,
            "instability_fpr_threshold": config.instability_fpr_threshold,
            "random_seed": config.random_seed,
            "base_mean": config.base_mean,
            "base_std": config.base_std,
            "drift_start_hour": config.drift_start_hour,
            "drift_total_shift": config.drift_total_shift,
            "anomaly_probability": config.anomaly_probability,
            "anomaly_magnitude": config.anomaly_magnitude,
        },
        "workload": {
            "training_points": int(len(train_values)),
            "evaluation_points": int(len(eval_values)),
        },
        "detectors": {
            "baseline": {
                "fpr": baseline_fpr,
                "alerts_per_hour": _alerts_per_hour(baseline_alerts, config.evaluation_hours),
                "threshold_stability": _threshold_stability(baseline_thresholds),
                "time_to_instability_hours": _time_to_instability_hours(
                    baseline_alerts,
                    anomaly_mask,
                    sampling_hz=config.sampling_hz,
                    fpr_limit=config.instability_fpr_threshold,
                ),
            },
            "rolling": {
                "fpr": rolling_fpr,
                "alerts_per_hour": _alerts_per_hour(rolling_alerts, config.evaluation_hours),
                "threshold_stability": _threshold_stability(rolling_thresholds),
                "time_to_instability_hours": _time_to_instability_hours(
                    rolling_alerts,
                    anomaly_mask,
                    sampling_hz=config.sampling_hz,
                    fpr_limit=config.instability_fpr_threshold,
                ),
            },
        },
        "comparison": {
            "fpr_relative_change_percent": float(
                ((baseline_fpr - rolling_fpr) / rolling_fpr) * 100.0 if rolling_fpr > 0 else 0.0
            ),
            "alerts_per_hour_delta": float(
                _alerts_per_hour(baseline_alerts, config.evaluation_hours)
                - _alerts_per_hour(rolling_alerts, config.evaluation_hours)
            ),
        },
    }

    if args.plot:
        result["artifacts"] = _save_plots(
            output_dir=args.plot_dir,
            eval_values=eval_values,
            baseline_thresholds=baseline_thresholds,
            rolling_thresholds=rolling_thresholds,
            baseline_alerts=baseline_alerts,
            rolling_alerts=rolling_alerts,
            anomaly_mask=anomaly_mask,
            config=config,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
