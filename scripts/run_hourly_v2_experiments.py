#!/usr/bin/env python3
"""V2 leakage-aware residual deep-learning experiments for hourly demand.

The strong four-week same-hour median is used as an exact anchor.  A causal
per-hour temporal encoder may add only a bounded, gated correction.  Calendar
event corrections are isolated from the ordinary-day correction so that a
holiday signal cannot silently shift every ordinary prediction.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CONTEXT_DAYS = 56
WEEKLY_LAGS = (7, 14, 21, 28)
ALL_LAGS = (1, 7, 14, 21, 28, 35, 42, 49, 56)
HOLIDAYS_2025 = (
    "2025-01-01",
    "2025-01-27",
    "2025-01-28",
    "2025-01-29",
    "2025-01-30",
    "2025-03-01",
    "2025-03-03",
    "2025-05-05",
    "2025-05-06",
    "2025-06-03",
    "2025-06-06",
    "2025-08-15",
    "2025-10-03",
    "2025-10-05",
    "2025-10-06",
    "2025-10-07",
    "2025-10-08",
    "2025-10-09",
    "2025-12-25",
)
EVENT_VALIDATION_DATES = (
    "2025-08-15",
    "2025-10-03",
    "2025-10-04",
    "2025-10-05",
    "2025-10-06",
    "2025-10-07",
    "2025-10-08",
    "2025-10-09",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("hourly_experiment_data.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/hourly_v2"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument(
        "--suite",
        choices=("core", "seeds", "folds", "all", "smoke"),
        default="core",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class ModelConfig:
    name: str
    temporal: bool = True
    weather: bool = False
    calendar: bool = False
    cross_hour: bool = True
    gate: bool = True
    station_context: bool = True
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def date_index(dates: np.ndarray, value: str) -> int:
    day_dates = dates.astype("datetime64[D]")
    matches = np.flatnonzero(day_dates == np.datetime64(value))
    if not len(matches):
        raise ValueError(f"Date not found: {value}")
    return int(matches[0])


def target_range(dates: np.ndarray, start: str, end: str) -> np.ndarray:
    return np.arange(
        max(CONTEXT_DAYS, date_index(dates, start)),
        date_index(dates, end) + 1,
        dtype=np.int64,
    )


def calendar_array(dates: np.ndarray) -> np.ndarray:
    """Return holiday, previous-day, next-day, and run-length features."""
    day_dates = dates.astype("datetime64[D]")
    holiday_dates = np.asarray(HOLIDAYS_2025, dtype="datetime64[D]")
    holiday = np.isin(day_dates, holiday_dates)
    before = np.isin(day_dates + np.timedelta64(1, "D"), holiday_dates)
    after = np.isin(day_dates - np.timedelta64(1, "D"), holiday_dates)
    run_length = np.zeros(len(day_dates), dtype=np.float32)
    for index in np.flatnonzero(holiday):
        left, right = index, index
        while left > 0 and holiday[left - 1]:
            left -= 1
        while right + 1 < len(holiday) and holiday[right + 1]:
            right += 1
        run_length[index] = min(right - left + 1, 5) / 5.0
    return np.stack(
        [
            holiday.astype(np.float32),
            before.astype(np.float32),
            after.astype(np.float32),
            run_length,
        ],
        axis=-1,
    )


def normalize_weather(
    values: np.ndarray, train_days: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(values)
    normalized = values.astype(np.float32).copy()
    center = np.zeros(values.shape[-1], dtype=np.float32)
    scale = np.ones(values.shape[-1], dtype=np.float32)
    categorical = {6, 7}
    for feature in range(values.shape[-1]):
        train_values = values[train_days, ..., feature]
        fill = 0.0 if feature in categorical else float(np.nanmedian(train_values))
        if feature not in categorical:
            center[feature] = fill
            scale[feature] = max(float(np.nanstd(train_values)), 1e-3)
        normalized[..., feature] = np.where(
            mask[..., feature], normalized[..., feature], fill
        )
        if feature not in categorical:
            normalized[..., feature] = (
                normalized[..., feature] - center[feature]
            ) / scale[feature]
    return normalized, mask.astype(np.float32), center, scale


class ExperimentData:
    def __init__(self, path: Path, scale_train_days: np.ndarray):
        raw = np.load(path)
        self.dates = raw["dates"]
        self.demand = raw["demand"].astype(np.float32)
        if not np.isfinite(self.demand).all():
            raise ValueError("Demand tensor contains non-finite values")
        self.hours = raw["service_hours"].astype(np.int64)
        self.weekday = raw["weekday"].astype(np.int64)
        self.doy = np.stack([raw["doy_sin"], raw["doy_cos"]], axis=-1).astype(
            np.float32
        )
        self.calendar = calendar_array(self.dates)
        self.station_id = raw["series_station_id"].astype(np.int64)
        self.line_id = raw["series_line_id"].astype(np.int64)
        self.direction_id = raw["series_direction_id"].astype(np.int64)
        self.series_names = raw["series_station"]
        self.series_lines = raw["series_line"]
        self.series_directions = raw["series_direction"]
        self.forecast_lead_hours = raw["forecast_lead_hours"].astype(np.float32)
        self.n_series = self.demand.shape[1]
        self.n_hours = self.demand.shape[2]
        self.n_stations = int(self.station_id.max() + 1)
        self.n_lines = int(self.line_id.max() + 1)
        self.n_directions = int(self.direction_id.max() + 1)

        self.series_hour_scale = (
            np.median(self.demand[scale_train_days], axis=0) + 1.0
        ).astype(np.float32)
        self.global_target_mean = float(
            max(self.demand[scale_train_days].mean(), 1.0)
        )
        (
            self.forecast,
            self.forecast_mask,
            self.weather_center,
            self.weather_scale,
        ) = normalize_weather(
            raw["forecast_weather"].astype(np.float32), scale_train_days
        )

    def make_batch(
        self,
        sample_indices: np.ndarray,
        target_days: np.ndarray,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        days = target_days[sample_indices // self.n_series]
        series = sample_indices % self.n_series
        history_days = days[:, None] - np.arange(
            CONTEXT_DAYS, 0, -1, dtype=np.int64
        )[None, :]
        weekly = np.stack(
            [self.demand[days - lag, series] for lag in WEEKLY_LAGS], axis=0
        )
        anchor = np.median(weekly, axis=0).astype(np.float32)
        arrays = {
            "hist_raw": self.demand[history_days, series[:, None], :],
            "anchor_raw": anchor,
            "target_raw": self.demand[days, series, :],
            "scale": self.series_hour_scale[series],
            "future": self.forecast[days],
            "future_mask": self.forecast_mask[days],
            "target_weekday": self.weekday[days],
            "target_doy": self.doy[days],
            "calendar": self.calendar[days],
            "station_id": self.station_id[series],
            "line_id": self.line_id[series],
            "direction_id": self.direction_id[series],
            "days": days,
            "series": series,
        }
        return {
            key: torch.from_numpy(np.asarray(value)).to(device)
            for key, value in arrays.items()
        }


class StaticContext(nn.Module):
    def __init__(self, data: ExperimentData):
        super().__init__()
        self.station = nn.Embedding(data.n_stations, 20)
        self.line = nn.Embedding(data.n_lines, 6)
        self.direction = nn.Embedding(data.n_directions, 4)
        self.projection = nn.Sequential(
            nn.Linear(30, 32),
            nn.GELU(),
            nn.LayerNorm(32),
        )

    def forward(
        self, batch: dict[str, torch.Tensor], enabled: bool
    ) -> torch.Tensor:
        if not enabled:
            return torch.zeros(
                len(batch["station_id"]), 32, device=batch["station_id"].device
            )
        return self.projection(
            torch.cat(
                [
                    self.station(batch["station_id"]),
                    self.line(batch["line_id"]),
                    self.direction(batch["direction_id"]),
                ],
                dim=-1,
            )
        )


class ForecastWeatherEncoder(nn.Module):
    """Small target-hour encoder; past observed weather is intentionally absent."""

    def __init__(self):
        super().__init__()
        self.precip_type = nn.Embedding(8, 3)
        self.sky_state = nn.Embedding(8, 3)
        self.network = nn.Sequential(
            nn.Linear(18, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.GELU(),
        )

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor, enabled: bool
    ) -> torch.Tensor:
        batch, horizon, _ = values.shape
        if not enabled:
            return torch.zeros(batch, horizon, 16, device=values.device)
        numeric = torch.cat([values[..., :6], mask[..., :6]], dim=-1)
        precip = values[..., 6].round().long().clamp(0, 6) + 1
        sky = values[..., 7].round().long().clamp(0, 6) + 1
        precip = torch.where(mask[..., 6].bool(), precip, torch.zeros_like(precip))
        sky = torch.where(mask[..., 7].bool(), sky, torch.zeros_like(sky))
        return self.network(
            torch.cat(
                [numeric, self.precip_type(precip), self.sky_state(sky)], dim=-1
            )
        )


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=2, dilation=dilation
        )
        self.norm = nn.GroupNorm(4, channels)
        self.dropout = nn.Dropout(0.08)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        causal = F.pad(values, (self.dilation, 0))
        update = F.gelu(self.norm(self.conv(causal)))
        return values + self.dropout(update)


class PerHourCausalEncoder(nn.Module):
    """Shared strictly causal TCN applied independently to each service hour."""

    def __init__(self, channels: int = 16):
        super().__init__()
        self.input_projection = nn.Conv1d(1, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                CausalResidualBlock(channels, dilation)
                for dilation in (1, 2, 4, 8, 16, 32)
            ]
        )
        self.output_norm = nn.GroupNorm(4, channels)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch, days, hours = history.shape
        values = history.transpose(1, 2).reshape(batch * hours, 1, days)
        values = self.input_projection(values)
        for block in self.blocks:
            values = block(values)
        encoded = self.output_norm(values)[:, :, -1]
        return encoded.reshape(batch, hours, -1)


class CrossHourBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.08)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = values.transpose(1, 2)
        update = self.conv2(F.gelu(self.conv1(update))).transpose(1, 2)
        return self.norm(values + self.dropout(update))


class GatedResidualForecaster(nn.Module):
    def __init__(self, data: ExperimentData, config: ModelConfig):
        super().__init__()
        self.config = config
        self.static = StaticContext(data)
        self.weather = ForecastWeatherEncoder()
        self.temporal = PerHourCausalEncoder(16)
        self.hour = nn.Embedding(data.n_hours, 6)
        self.weekday = nn.Embedding(7, 5)
        self.calendar_projection = nn.Sequential(
            nn.Linear(4, 8), nn.GELU(), nn.LayerNorm(8)
        )
        self.register_buffer("hour_ids", torch.arange(data.n_hours))
        # temporal 16 + explicit lags 15 + hour 6 + static 32 +
        # weekday 5 + day-of-year 2 + calendar 8 + weather 16
        self.token_projection = nn.Sequential(
            nn.Linear(100, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.08),
        )
        self.cross_hour = nn.ModuleList([CrossHourBlock(), CrossHourBlock()])
        self.general_head = nn.Linear(64, 2)
        self.event_head = nn.Linear(64 + 4, 2)
        self._initialize_safe_heads()

    def _initialize_safe_heads(self) -> None:
        nn.init.zeros_(self.general_head.weight)
        nn.init.zeros_(self.general_head.bias)
        self.general_head.bias.data[1] = -2.2
        nn.init.zeros_(self.event_head.weight)
        nn.init.zeros_(self.event_head.bias)
        self.event_head.bias.data[1] = -1.0

    @staticmethod
    def explicit_lags(
        history_raw: torch.Tensor,
        scale: torch.Tensor,
        anchor_raw: torch.Tensor,
    ) -> torch.Tensor:
        lag_values = torch.stack(
            [history_raw[:, -lag, :] for lag in ALL_LAGS], dim=-1
        )
        lag_log = torch.log1p(lag_values / scale.unsqueeze(-1))
        weekly = torch.stack(
            [history_raw[:, -lag, :] for lag in WEEKLY_LAGS], dim=-1
        )
        weekly_mad = torch.median(
            torch.abs(weekly - anchor_raw.unsqueeze(-1)), dim=-1
        ).values / scale
        weekly_std = torch.std(weekly, dim=-1, unbiased=False) / scale
        weekly_trend = (weekly[..., 0] - weekly[..., -1]) / scale
        recent_anchor = torch.median(weekly[..., 1:], dim=-1).values
        recent_residual = (weekly[..., 0] - recent_anchor) / scale
        prior_anchor = torch.median(
            torch.stack(
                [
                    history_raw[:, -lag, :]
                    for lag in (21, 28, 35, 42)
                ],
                dim=-1,
            ),
            dim=-1,
        ).values
        prior_residual = (history_raw[:, -14, :] - prior_anchor) / scale
        anchor_log = torch.log1p(anchor_raw / scale)
        return torch.cat(
            [
                lag_log,
                anchor_log.unsqueeze(-1),
                weekly_mad.unsqueeze(-1),
                weekly_std.unsqueeze(-1),
                weekly_trend.unsqueeze(-1),
                recent_residual.unsqueeze(-1),
                prior_residual.unsqueeze(-1),
            ],
            dim=-1,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scale = batch["scale"]
        history_z = torch.log1p(batch["hist_raw"] / scale[:, None, :])
        batch_size, _, horizon = history_z.shape
        if self.config.temporal:
            temporal = self.temporal(history_z)
        else:
            temporal = torch.zeros(
                batch_size, horizon, 16, device=history_z.device
            )
        lags = self.explicit_lags(
            batch["hist_raw"], scale, batch["anchor_raw"]
        )
        static = self.static(batch, self.config.station_context)
        static = static[:, None, :].expand(-1, horizon, -1)
        hour = self.hour(self.hour_ids)[None].expand(batch_size, -1, -1)
        weekday = self.weekday(batch["target_weekday"])[:, None].expand(
            -1, horizon, -1
        )
        doy = batch["target_doy"][:, None].expand(-1, horizon, -1)
        if self.config.calendar:
            calendar = self.calendar_projection(batch["calendar"])
        else:
            calendar = torch.zeros(batch_size, 8, device=history_z.device)
        calendar = calendar[:, None].expand(-1, horizon, -1)
        weather = self.weather(
            batch["future"], batch["future_mask"], self.config.weather
        )
        token = self.token_projection(
            torch.cat(
                [
                    temporal,
                    lags,
                    hour,
                    static,
                    weekday,
                    doy,
                    calendar,
                    weather,
                ],
                dim=-1,
            )
        )
        if self.config.cross_hour:
            for block in self.cross_hour:
                token = block(token)

        general_raw = self.general_head(token)
        general_gate = (
            torch.sigmoid(general_raw[..., 1])
            if self.config.gate
            else torch.ones_like(general_raw[..., 1])
        )
        general_delta = 2.5 * torch.tanh(general_raw[..., 0])
        correction = general_gate * general_delta

        event_gate = torch.zeros_like(general_gate)
        event_delta = torch.zeros_like(general_delta)
        if self.config.calendar:
            calendar_expanded = batch["calendar"][:, None].expand(
                -1, horizon, -1
            )
            event_raw = self.event_head(
                torch.cat([token, calendar_expanded], dim=-1)
            )
            event_gate = (
                torch.sigmoid(event_raw[..., 1])
                if self.config.gate
                else torch.ones_like(event_raw[..., 1])
            )
            event_delta = 5.0 * torch.tanh(event_raw[..., 0])
            event_strength = torch.max(
                batch["calendar"][:, :3], dim=-1
            ).values[:, None]
            correction = correction + event_strength * event_gate * event_delta

        raw_prediction = batch["anchor_raw"] + scale * correction
        prediction = torch.clamp_min(raw_prediction, 0.0)
        return {
            "prediction": prediction,
            "correction": correction,
            "general_gate": general_gate,
            "event_gate": event_gate,
            "event_delta": event_delta,
        }


def loss_function(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    global_target_mean: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = output["prediction"]
    target = batch["target_raw"]
    raw_mae = torch.mean(torch.abs(prediction - target)) / global_target_mean
    target_residual = (target - batch["anchor_raw"]) / batch["scale"]
    predicted_residual = (prediction - batch["anchor_raw"]) / batch["scale"]
    normalized_huber = F.huber_loss(
        predicted_residual, target_residual, delta=0.5
    )
    correction_penalty = torch.mean(torch.abs(output["correction"]))
    gate_penalty = torch.mean(output["general_gate"])
    loss = (
        raw_mae
        + 0.12 * normalized_huber
        + 5e-4 * correction_penalty
        + 2e-4 * gate_penalty
    )
    parts = {
        "raw_mae_scaled": float(raw_mae.detach()),
        "normalized_huber": float(normalized_huber.detach()),
        "correction_penalty": float(correction_penalty.detach()),
        "gate_mean": float(gate_penalty.detach()),
    }
    return loss, parts


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - actual
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(absolute.sum() / max(float(actual.sum()), 1.0)),
        "smape": float(
            np.mean(
                2.0
                * absolute
                / np.maximum(np.abs(actual) + np.abs(prediction), 1.0)
            )
        ),
        "bias": float(error.mean()),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "daily_mae": float(
            np.mean(np.abs(prediction.sum(axis=1) - actual.sum(axis=1)))
        ),
        "hourly_mae": [float(value) for value in absolute.mean(axis=0)],
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: ExperimentData,
    target_days: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    model.eval()
    total = len(target_days) * data.n_series
    parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "actual",
            "prediction",
            "anchor",
            "correction",
            "general_gate",
            "event_gate",
            "event_delta",
            "day_index",
            "series_index",
        )
    }
    for start in range(0, total, batch_size):
        indices = np.arange(start, min(total, start + batch_size))
        batch = data.make_batch(indices, target_days, device)
        output = model(batch)
        parts["actual"].append(batch["target_raw"].cpu().numpy())
        parts["prediction"].append(output["prediction"].cpu().numpy())
        parts["anchor"].append(batch["anchor_raw"].cpu().numpy())
        for key in ("correction", "general_gate", "event_gate", "event_delta"):
            parts[key].append(output[key].cpu().numpy())
        parts["day_index"].append(batch["days"].cpu().numpy())
        parts["series_index"].append(batch["series"].cpu().numpy())
    artifacts = {
        key: np.concatenate(values).astype(
            np.int16 if key in {"day_index", "series_index"} else np.float32
        )
        for key, values in parts.items()
    }
    result = metrics(artifacts["actual"], artifacts["prediction"])
    result.update(
        {
            "mean_absolute_correction_normalized": float(
                np.mean(np.abs(artifacts["correction"]))
            ),
            "mean_general_gate": float(artifacts["general_gate"].mean()),
            "mean_event_gate": float(artifacts["event_gate"].mean()),
            "zero_correction_share": float(
                np.mean(np.abs(artifacts["correction"]) < 1e-4)
            ),
        }
    )
    return result, artifacts


def baseline_artifacts(
    data: ExperimentData, target_days: np.ndarray
) -> dict[str, np.ndarray]:
    days = np.repeat(target_days, data.n_series)
    series = np.tile(np.arange(data.n_series), len(target_days))
    actual = data.demand[days, series]
    weekly = np.stack(
        [data.demand[days - lag, series] for lag in WEEKLY_LAGS], axis=0
    )
    prediction = np.median(weekly, axis=0)
    return {
        "actual": actual.astype(np.float32),
        "prediction": prediction.astype(np.float32),
        "day_index": days.astype(np.int16),
        "series_index": series.astype(np.int16),
    }


def selection_score(
    regular: dict, event: dict, regular_base: dict, event_base: dict
) -> float:
    regular_ratio = regular["mae"] / regular_base["mae"]
    event_ratio = event["mae"] / event_base["mae"]
    return float(0.85 * regular_ratio + 0.15 * event_ratio)


def train_model(
    data: ExperimentData,
    config: ModelConfig,
    train_days: np.ndarray,
    regular_validation_days: np.ndarray,
    event_validation_days: np.ndarray,
    device: torch.device,
    epochs: int,
    patience: int,
    min_epochs: int,
    batch_size: int,
) -> tuple[nn.Module, list[dict], dict]:
    set_seed(config.seed)
    model = GatedResidualForecaster(data, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=6e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=6e-5
    )
    regular_baseline_artifacts = baseline_artifacts(
        data, regular_validation_days
    )
    event_baseline_artifacts = baseline_artifacts(data, event_validation_days)
    regular_baseline = metrics(
        regular_baseline_artifacts["actual"],
        regular_baseline_artifacts["prediction"],
    )
    event_baseline = metrics(
        event_baseline_artifacts["actual"],
        event_baseline_artifacts["prediction"],
    )

    initial_regular, _ = evaluate_model(
        model, data, regular_validation_days, device, batch_size
    )
    initial_event, _ = evaluate_model(
        model, data, event_validation_days, device, batch_size
    )
    best_score = selection_score(
        initial_regular, initial_event, regular_baseline, event_baseline
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = [
        {
            "epoch": 0,
            "train_loss": None,
            "regular_validation_mae": initial_regular["mae"],
            "event_validation_mae": initial_event["mae"],
            "selection_score": best_score,
            "lr": optimizer.param_groups[0]["lr"],
        }
    ]
    total = len(train_days) * data.n_series
    stale = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        generator = np.random.default_rng(config.seed + epoch)
        permutation = generator.permutation(total)
        running, seen = 0.0, 0
        part_sums: dict[str, float] = {}
        for start in range(0, total, batch_size):
            indices = permutation[start : start + batch_size]
            batch = data.make_batch(indices, train_days, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, loss_parts = loss_function(
                output, batch, data.global_target_mean
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(indices)
            seen += len(indices)
            for key, value in loss_parts.items():
                part_sums[key] = part_sums.get(key, 0.0) + value * len(indices)
        scheduler.step()
        regular, _ = evaluate_model(
            model, data, regular_validation_days, device, batch_size
        )
        event, _ = evaluate_model(
            model, data, event_validation_days, device, batch_size
        )
        score = selection_score(
            regular, event, regular_baseline, event_baseline
        )
        row = {
            "epoch": epoch,
            "train_loss": running / seen,
            "regular_validation_mae": regular["mae"],
            "event_validation_mae": event["mae"],
            "selection_score": score,
            "lr": optimizer.param_groups[0]["lr"],
            **{key: value / seen for key, value in part_sums.items()},
        }
        history.append(row)
        print(
            f"{config.name} epoch={epoch:02d} loss={row['train_loss']:.5f} "
            f"regular={regular['mae']:.2f} event={event['mae']:.2f} "
            f"score={score:.4f}",
            flush=True,
        )
        if score < best_score - 1e-4:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break
    model.load_state_dict(best_state)
    regular, _ = evaluate_model(
        model, data, regular_validation_days, device, batch_size
    )
    event, _ = evaluate_model(
        model, data, event_validation_days, device, batch_size
    )
    validation = {
        "best_epoch": int(best_epoch),
        "selection_score": float(best_score),
        "regular": regular,
        "event": event,
        "regular_baseline": regular_baseline,
        "event_baseline": event_baseline,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "elapsed_seconds": float(time.time() - started),
    }
    return model, history, validation


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def core_configs() -> list[ModelConfig]:
    return [
        ModelConfig(
            "residual_mlp_calendar",
            temporal=False,
            weather=False,
            calendar=True,
            cross_hour=False,
        ),
        ModelConfig(
            "causal_residual_base",
            weather=False,
            calendar=False,
        ),
        ModelConfig(
            "causal_residual_weather",
            weather=True,
            calendar=False,
        ),
        ModelConfig(
            "causal_residual_calendar",
            weather=False,
            calendar=True,
        ),
        ModelConfig(
            "causal_residual_full",
            weather=True,
            calendar=True,
        ),
        ModelConfig(
            "causal_residual_full_no_gate",
            weather=True,
            calendar=True,
            gate=False,
        ),
        ModelConfig(
            "causal_residual_full_no_cross",
            weather=True,
            calendar=True,
            cross_hour=False,
        ),
    ]


def seed_configs() -> list[ModelConfig]:
    return [
        ModelConfig(
            f"causal_residual_full_no_gate_seed_{seed}",
            weather=True,
            calendar=True,
            gate=False,
            seed=seed,
        )
        for seed in (123, 2025)
    ]


def split_metrics(
    data: ExperimentData, artifacts: dict[str, np.ndarray]
) -> dict[str, dict]:
    day_indices = artifacts["day_index"]
    dates = data.dates[day_indices].astype("datetime64[D]")
    calendar = data.calendar[day_indices]
    masks = {
        "all": np.ones(len(day_indices), dtype=bool),
        "ordinary": np.max(calendar[:, :3], axis=1) == 0,
        "holiday": calendar[:, 0] == 1,
        "holiday_adjacent": np.max(calendar[:, 1:3], axis=1) == 1,
        "before_2025_12_20": dates < np.datetime64("2025-12-20"),
        "from_2025_12_20": dates >= np.datetime64("2025-12-20"),
        "christmas": dates == np.datetime64("2025-12-25"),
    }
    result = {}
    for name, mask in masks.items():
        if mask.any():
            result[name] = metrics(
                artifacts["actual"][mask], artifacts["prediction"][mask]
            )
    return result


def run_config(
    data: ExperimentData,
    config: ModelConfig,
    train_days: np.ndarray,
    regular_validation_days: np.ndarray,
    event_validation_days: np.ndarray,
    test_days: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    results: list[dict],
    output_artifacts: dict[str, np.ndarray],
    split_suffix: str = "",
) -> None:
    model, history, validation = train_model(
        data,
        config,
        train_days,
        regular_validation_days,
        event_validation_days,
        device,
        args.epochs,
        args.patience,
        args.min_epochs,
        args.batch_size,
    )
    results.append(
        {
            "name": config.name,
            "family": "deep_learning_v2",
            "split": f"validation{split_suffix}",
            "config": asdict(config),
            **validation,
        }
    )
    test_metrics, test_artifacts = evaluate_model(
        model, data, test_days, device, args.batch_size
    )
    results.append(
        {
            "name": config.name,
            "family": "deep_learning_v2",
            "split": f"test{split_suffix}",
            "config": asdict(config),
            **test_metrics,
            "slices": split_metrics(data, test_artifacts),
        }
    )
    save_json(
        args.output_dir / f"history__{config.name}{split_suffix}.json", history
    )
    if not split_suffix:
        for key, value in test_artifacts.items():
            output_artifacts[f"{config.name}__test_{key}"] = value
        regular_metrics, regular_artifacts = evaluate_model(
            model,
            data,
            regular_validation_days,
            device,
            args.batch_size,
        )
        for key, value in regular_artifacts.items():
            output_artifacts[f"{config.name}__validation_{key}"] = value
        output_artifacts[
            f"{config.name}__validation_mae"
        ] = np.asarray([regular_metrics["mae"]], dtype=np.float32)
    save_json(args.output_dir / "results.json", results)
    np.savez_compressed(
        args.output_dir / "predictions.npz", **output_artifacts
    )
    del model
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is required unless --allow-cpu is supplied")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"device={device} torch={torch.__version__}", flush=True)

    raw_dates = np.load(args.data)["dates"]
    full_train = target_range(raw_dates, "2025-02-26", "2025-10-31")
    regular_validation = target_range(
        raw_dates, "2025-11-01", "2025-11-14"
    )
    test_days = target_range(raw_dates, "2025-11-15", "2025-12-31")
    event_validation = np.asarray(
        [date_index(raw_dates, value) for value in EVENT_VALIDATION_DATES],
        dtype=np.int64,
    )
    train_days = full_train[
        ~np.isin(full_train, event_validation)
    ]
    data = ExperimentData(args.data, train_days)

    baseline_splits = {
        "validation_regular": regular_validation,
        "validation_event": event_validation,
        "test": test_days,
    }
    results: list[dict] = []
    output_artifacts: dict[str, np.ndarray] = {
        "dates": data.dates,
        "service_hours": data.hours.astype(np.int8),
        "series_station": data.series_names,
        "series_line": data.series_lines,
        "series_direction": data.series_directions,
    }
    for split, days in baseline_splits.items():
        artifacts = baseline_artifacts(data, days)
        results.append(
            {
                "name": "four_week_median",
                "family": "statistical",
                "split": split,
                **metrics(artifacts["actual"], artifacts["prediction"]),
                "slices": split_metrics(data, artifacts),
            }
        )
        for key, value in artifacts.items():
            output_artifacts[f"four_week_median__{split}_{key}"] = value

    configs = core_configs()
    if args.suite in {"seeds", "all"}:
        configs = seed_configs() if args.suite == "seeds" else configs + seed_configs()
    if args.suite == "smoke":
        configs = [core_configs()[-1]]
        args.epochs = 1
        args.patience = 1
        args.min_epochs = 1
        train_days = train_days[-7:]
        regular_validation = regular_validation[:2]
        event_validation = event_validation[:2]

    if args.suite != "folds":
        for config in configs:
            run_config(
                data,
                config,
                train_days,
                regular_validation,
                event_validation,
                test_days,
                device,
                args,
                results,
                output_artifacts,
            )

    if args.suite in {"folds", "all"}:
        fold_definitions = [
            ("fold_aug", "2025-02-26", "2025-07-31", "2025-08-01", "2025-08-14"),
            ("fold_sep", "2025-02-26", "2025-08-31", "2025-09-01", "2025-09-14"),
            ("fold_oct", "2025-02-26", "2025-09-30", "2025-10-01", "2025-10-14"),
        ]
        for fold_name, train_start, train_end, val_start, val_end in fold_definitions:
            fold_train = target_range(raw_dates, train_start, train_end)
            fold_validation = target_range(raw_dates, val_start, val_end)
            # Use earlier public holidays as a stress set when available.
            eligible_events = event_validation[event_validation <= fold_train.max()]
            if not len(eligible_events):
                eligible_events = fold_validation[:1]
            fold_train = fold_train[~np.isin(fold_train, eligible_events)]
            fold_data = ExperimentData(args.data, fold_train)
            for base_config in (
                ModelConfig(
                    "causal_residual_calendar",
                    weather=False,
                    calendar=True,
                ),
                ModelConfig(
                    "causal_residual_full",
                    weather=True,
                    calendar=True,
                ),
            ):
                config = ModelConfig(
                    name=f"{base_config.name}_{fold_name}",
                    weather=base_config.weather,
                    calendar=base_config.calendar,
                )
                run_config(
                    fold_data,
                    config,
                    fold_train,
                    fold_validation,
                    eligible_events,
                    fold_validation,
                    device,
                    args,
                    results,
                    output_artifacts,
                    split_suffix=f"__{fold_name}",
                )

    audit = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "contextDays": CONTEXT_DAYS,
        "causalDilations": [1, 2, 4, 8, 16, 32],
        "causalReceptiveFieldDays": 64,
        "anchorLagsDays": list(WEEKLY_LAGS),
        "series": data.n_series,
        "hours": data.hours.tolist(),
        "trainDates": [
            str(data.dates[train_days.min()]),
            str(data.dates[train_days.max()]),
        ],
        "trainExcludedEventDates": [
            str(data.dates[index]) for index in event_validation
        ],
        "regularValidationDates": [
            str(data.dates[regular_validation.min()]),
            str(data.dates[regular_validation.max()]),
        ],
        "testDates": [
            str(data.dates[test_days.min()]),
            str(data.dates[test_days.max()]),
        ],
        "selectionScore": "0.85*(regular MAE/baseline MAE)+0.15*(event MAE/baseline MAE)",
        "initialPrediction": "exact four-week same-hour median",
        "epochs": args.epochs,
        "patience": args.patience,
        "minimumEpochs": args.min_epochs,
        "batchSize": args.batch_size,
        "suite": args.suite,
    }
    save_json(args.output_dir / "run_audit.json", audit)
    save_json(args.output_dir / "results.json", results)
    np.savez_compressed(
        args.output_dir / "predictions.npz", **output_artifacts
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
