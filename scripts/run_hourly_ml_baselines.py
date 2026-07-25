#!/usr/bin/env python3
"""Strong linear and tree baselines for the hourly deep-learning experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


HOURS = 18
CONTEXT = 28


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("hourly_experiment_data.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/hourly"))
    parser.add_argument("--tree-sample", type=int, default=700_000)
    return parser.parse_args()


def date_index(dates: np.ndarray, value: str) -> int:
    return int(
        np.flatnonzero(
            dates.astype("datetime64[D]") == np.datetime64(value)
        )[0]
    )


def target_range(dates: np.ndarray, start: str, end: str) -> np.ndarray:
    return np.arange(
        max(CONTEXT, date_index(dates, start)),
        date_index(dates, end) + 1,
    )


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - actual
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(absolute.sum() / actual.sum()),
        "smape": float(
            np.mean(
                2
                * absolute
                / np.maximum(np.abs(actual) + np.abs(prediction), 1)
            )
        ),
        "bias": float(error.mean()),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.9)),
        "hourly_mae": [
            float(absolute[hour::HOURS].mean())
            for hour in range(HOURS)
        ],
    }


class PointFrame:
    def __init__(self, raw, train_end: int):
        self.dates = raw["dates"]
        self.demand = raw["demand"].astype(np.float32)
        self.station = raw["series_station_id"].astype(np.int32)
        self.line = raw["series_line_id"].astype(np.int32)
        self.direction = raw["series_direction_id"].astype(np.int32)
        self.weekday = raw["weekday"].astype(np.int32)
        self.doy = np.stack(
            [raw["doy_sin"], raw["doy_cos"]], axis=-1
        ).astype(np.float32)
        self.forecast = raw["forecast_weather"].astype(np.float32)
        self.series_count = self.demand.shape[1]
        self.scale = (
            np.median(self.demand[: train_end + 1], axis=(0, 2)) + 1
        ).astype(np.float32)
        self.z = np.log1p(
            self.demand / self.scale[None, :, None]
        ).astype(np.float32)
        self.weather_center = np.nanmedian(
            self.forecast[: train_end + 1], axis=(0, 1)
        )
        self.weather_scale = np.maximum(
            np.nanstd(self.forecast[: train_end + 1], axis=(0, 1)),
            1e-3,
        )

    def build(self, target_days: np.ndarray) -> dict[str, np.ndarray]:
        days = np.repeat(target_days, self.series_count * HOURS)
        series = np.tile(
            np.repeat(np.arange(self.series_count), HOURS),
            len(target_days),
        )
        hour = np.tile(
            np.arange(HOURS), len(target_days) * self.series_count
        )
        lags = np.stack(
            [self.z[days - lag, series, hour] for lag in (1, 7, 14, 21, 28)],
            axis=-1,
        )
        same_weekday = np.mean(lags[:, 1:], axis=1, keepdims=True)
        rolling7 = np.stack(
            [self.z[days - lag, series, hour] for lag in range(1, 8)],
            axis=-1,
        )
        rolling_mean = rolling7.mean(axis=1, keepdims=True)
        rolling_std = rolling7.std(axis=1, keepdims=True)
        weather = self.forecast[days, hour].copy()
        missing = (~np.isfinite(weather)).astype(np.float32)
        weather = np.where(
            np.isfinite(weather), weather, self.weather_center
        )
        weather[:, :6] = (
            weather[:, :6] - self.weather_center[:6]
        ) / self.weather_scale[:6]
        numeric = np.concatenate(
            [
                lags,
                same_weekday,
                rolling_mean,
                rolling_std,
                np.log1p(self.scale[series])[:, None],
                self.doy[days],
                weather,
                missing,
            ],
            axis=1,
        ).astype(np.float32)
        return {
            "numeric": numeric,
            "station": self.station[series],
            "line": self.line[series],
            "direction": self.direction[series],
            "hour": hour.astype(np.int32),
            "weekday": self.weekday[days],
            "target_z": self.z[days, series, hour],
            "target": self.demand[days, series, hour],
            "scale": self.scale[series],
            "days": days,
            "series": series,
        }


def standardize_numeric(
    train: dict[str, np.ndarray],
    *others: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    center = train["numeric"].mean(axis=0)
    scale = np.maximum(train["numeric"].std(axis=0), 1e-4)
    for frame in (train, *others):
        frame["numeric_scaled"] = (
            frame["numeric"] - center
        ) / scale
    return center, scale


def ridge_matrix(
    frame: dict[str, np.ndarray],
    category_sizes: tuple[int, int, int, int, int],
) -> sparse.csr_matrix:
    numeric = sparse.csr_matrix(frame["numeric_scaled"])
    row_count = len(frame["target"])
    rows = np.repeat(np.arange(row_count), 5)
    offsets = np.cumsum((0, *category_sizes[:-1]))
    columns = np.column_stack(
        [
            frame["station"] + offsets[0],
            frame["line"] + offsets[1],
            frame["direction"] + offsets[2],
            frame["hour"] + offsets[3],
            frame["weekday"] + offsets[4],
        ]
    ).reshape(-1)
    categories = sparse.csr_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            (rows, columns),
        ),
        shape=(row_count, sum(category_sizes)),
    )
    return sparse.hstack([numeric, categories], format="csr")


def inverse(z: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, scale * np.expm1(z))


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(args.data)
    dates = raw["dates"]
    train_days = target_range(dates, "2025-01-29", "2025-10-31")
    validation_days = target_range(
        dates, "2025-11-01", "2025-11-14"
    )
    test_days = target_range(dates, "2025-11-15", "2025-12-31")
    builder = PointFrame(raw, int(train_days.max()))
    print("Building flattened ML frames", flush=True)
    train = builder.build(train_days)
    validation = builder.build(validation_days)
    test = builder.build(test_days)
    standardize_numeric(train, validation, test)
    results = []
    predictions = {
        "actual": test["target"].astype(np.float32),
        "day_index": test["days"].astype(np.int16),
        "series_index": test["series"].astype(np.int16),
        "hour_index": test["hour"].astype(np.int8),
    }

    sizes = (
        int(builder.station.max() + 1),
        int(builder.line.max() + 1),
        int(builder.direction.max() + 1),
        HOURS,
        7,
    )
    print("Building sparse Ridge matrices", flush=True)
    x_train = ridge_matrix(train, sizes)
    x_validation = ridge_matrix(validation, sizes)
    x_test = ridge_matrix(test, sizes)
    started = time.time()
    ridge = Ridge(alpha=10.0, solver="lsqr", max_iter=500)
    ridge.fit(x_train, train["target_z"])
    for split_name, frame, matrix in (
        ("validation", validation, x_validation),
        ("test", test, x_test),
    ):
        prediction = inverse(ridge.predict(matrix), frame["scale"])
        results.append(
            {
                "name": "ridge_one_hot",
                "family": "machine_learning",
                "split": split_name,
                "alpha": 10.0,
                "trainingRows": int(len(train["target"])),
                "elapsedSeconds": float(time.time() - started),
                **metrics(frame["target"], prediction),
            }
        )
        if split_name == "test":
            predictions["ridge_one_hot"] = prediction.astype(np.float32)
    del x_train, x_validation, x_test

    # Station identity is omitted from the tree categorical set because the
    # 273 levels exceed HistGradientBoosting's 255-bin categorical limit.
    def tree_matrix(frame):
        return np.column_stack(
            [
                frame["numeric_scaled"],
                frame["line"],
                frame["direction"],
                frame["hour"],
                frame["weekday"],
            ]
        ).astype(np.float32)

    generator = np.random.default_rng(42)
    sample_size = min(args.tree_sample, len(train["target"]))
    sample = generator.choice(
        len(train["target"]), size=sample_size, replace=False
    )
    x_tree_train = tree_matrix(train)[sample]
    y_tree_train = train["target_z"][sample]
    categorical_indices = list(
        range(
            train["numeric_scaled"].shape[1],
            train["numeric_scaled"].shape[1] + 4,
        )
    )
    started = time.time()
    tree = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.07,
        max_iter=140,
        max_leaf_nodes=31,
        l2_regularization=10.0,
        categorical_features=categorical_indices,
        random_state=42,
    )
    print(f"Fitting HistGradientBoosting on {sample_size:,} rows", flush=True)
    tree.fit(x_tree_train, y_tree_train)
    for split_name, frame in (
        ("validation", validation),
        ("test", test),
    ):
        prediction = inverse(
            tree.predict(tree_matrix(frame)), frame["scale"]
        )
        results.append(
            {
                "name": "hist_gradient_boosting",
                "family": "machine_learning",
                "split": split_name,
                "trainingRows": int(sample_size),
                "elapsedSeconds": float(time.time() - started),
                **metrics(frame["target"], prediction),
            }
        )
        if split_name == "test":
            predictions["hist_gradient_boosting"] = prediction.astype(
                np.float32
            )

    save_json(args.output_dir / "ml_results.json", results)
    np.savez_compressed(
        args.output_dir / "ml_predictions.npz", **predictions
    )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
