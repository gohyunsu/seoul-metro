#!/usr/bin/env python3
"""Leakage-safe hourly subway demand deep-learning experiments."""

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


CONTEXT_DAYS = 28
QUANTILES = (0.1, 0.5, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("hourly_experiment_data.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/hourly"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--architecture-folds", action="store_true")
    parser.add_argument("--tcn-ablations-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class ModelConfig:
    name: str
    architecture: str
    past_weather: bool = True
    future_weather: str = "forecast"
    station_context: bool = True
    weekly_skip: bool = True
    cross_attention: bool = True
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def date_index(dates: np.ndarray, value: str) -> int:
    matches = np.flatnonzero(dates.astype("datetime64[D]") == np.datetime64(value))
    if not len(matches):
        raise ValueError(f"Date not found: {value}")
    return int(matches[0])


def target_range(dates: np.ndarray, start: str, end: str) -> np.ndarray:
    return np.arange(
        max(CONTEXT_DAYS, date_index(dates, start)),
        date_index(dates, end) + 1,
        dtype=np.int64,
    )


def normalize_weather(
    values: np.ndarray,
    train_end: int,
    categorical: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    categorical = categorical or set()
    mask = np.isfinite(values)
    normalized = values.astype(np.float32).copy()
    center = np.zeros(values.shape[-1], dtype=np.float32)
    scale = np.ones(values.shape[-1], dtype=np.float32)
    for feature in range(values.shape[-1]):
        train_values = values[: train_end + 1, ..., feature]
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
    def __init__(self, path: Path, train_end: int):
        raw = np.load(path)
        self.dates = raw["dates"]
        self.demand_raw = raw["demand"].astype(np.float32)
        self.hours = raw["service_hours"].astype(np.int64)
        self.weekday = raw["weekday"].astype(np.int64)
        self.doy = np.stack([raw["doy_sin"], raw["doy_cos"]], axis=-1).astype(
            np.float32
        )
        self.station_id = raw["series_station_id"].astype(np.int64)
        self.line_id = raw["series_line_id"].astype(np.int64)
        self.direction_id = raw["series_direction_id"].astype(np.int64)
        self.series_names = raw["series_station"]
        self.series_lines = raw["series_line"]
        self.series_directions = raw["series_direction"]
        self.n_series = self.demand_raw.shape[1]
        self.n_hours = self.demand_raw.shape[2]
        self.n_stations = int(self.station_id.max() + 1)
        self.n_lines = int(self.line_id.max() + 1)
        self.n_directions = int(self.direction_id.max() + 1)

        self.series_scale = (
            np.median(self.demand_raw[: train_end + 1], axis=(0, 2)) + 1.0
        ).astype(np.float32)
        self.demand_z = np.log1p(
            self.demand_raw / self.series_scale[None, :, None]
        ).astype(np.float32)
        (
            self.observed,
            self.observed_mask,
            self.obs_center,
            self.obs_scale,
        ) = normalize_weather(raw["observed_weather"].astype(np.float32), train_end)
        self.forecast_raw = raw["forecast_weather"].astype(np.float32)
        (
            self.forecast,
            self.forecast_mask,
            self.fc_center,
            self.fc_scale,
        ) = normalize_weather(
            self.forecast_raw, train_end, categorical={6, 7}
        )
        self.oracle, self.oracle_mask = self._build_oracle(
            raw["observed_weather"].astype(np.float32)
        )

    def _build_oracle(self, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Forecast order: temperature, precipitation, snow, POP, humidity,
        # wind speed, precipitation type, sky state.
        out = np.full((*observed.shape[:2], 8), np.nan, dtype=np.float32)
        out[..., 0] = observed[..., 0]
        out[..., 1] = observed[..., 1]
        out[..., 2] = observed[..., 4]
        out[..., 3] = np.where(observed[..., 1] > 0, 100.0, 0.0)
        out[..., 4] = observed[..., 3]
        out[..., 5] = observed[..., 2]
        rain, snow = observed[..., 1] > 0, observed[..., 4] > 0
        out[..., 6] = np.select(
            [rain & snow, rain, snow], [2.0, 1.0, 3.0], default=0.0
        )
        cloud = observed[..., 5]
        out[..., 7] = np.select(
            [cloud <= 2, cloud <= 7, cloud > 7],
            [1.0, 3.0, 4.0],
            default=np.nan,
        )
        mask = np.isfinite(out).astype(np.float32)
        normalized = out.copy()
        for feature in range(8):
            fill = 0.0 if feature in {6, 7} else self.fc_center[feature]
            normalized[..., feature] = np.where(
                np.isfinite(normalized[..., feature]),
                normalized[..., feature],
                fill,
            )
            if feature not in {6, 7}:
                normalized[..., feature] = (
                    normalized[..., feature] - self.fc_center[feature]
                ) / self.fc_scale[feature]
        return normalized.astype(np.float32), mask

    def make_batch(
        self,
        sample_indices: np.ndarray,
        target_days: np.ndarray,
        config: ModelConfig,
        device: torch.device,
        weather_day_map: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        days = target_days[sample_indices // self.n_series]
        series = sample_indices % self.n_series
        history_days = days[:, None] - np.arange(
            CONTEXT_DAYS, 0, -1, dtype=np.int64
        )[None, :]
        weather_days = days if weather_day_map is None else weather_day_map[days]
        if config.future_weather == "oracle":
            future = self.oracle[weather_days]
            future_mask = self.oracle_mask[weather_days]
        else:
            future = self.forecast[weather_days]
            future_mask = self.forecast_mask[weather_days]
        arrays = {
            "hist_y": self.demand_z[history_days, series[:, None], :],
            "hist_obs": self.observed[history_days],
            "hist_obs_mask": self.observed_mask[history_days],
            "hist_weekday": self.weekday[history_days],
            "hist_doy": self.doy[history_days],
            "future": future,
            "future_mask": future_mask,
            "target_weekday": self.weekday[days],
            "target_doy": self.doy[days],
            "station_id": self.station_id[series],
            "line_id": self.line_id[series],
            "direction_id": self.direction_id[series],
            "target_z": self.demand_z[days, series, :],
            "target_raw": self.demand_raw[days, series, :],
            "series_scale": self.series_scale[series],
            "days": days,
            "series": series,
        }
        return {
            key: torch.from_numpy(value).to(device)
            for key, value in arrays.items()
        }


class StaticContext(nn.Module):
    def __init__(self, data: ExperimentData, dim: int = 32):
        super().__init__()
        self.station = nn.Embedding(data.n_stations, 20)
        self.line = nn.Embedding(data.n_lines, 6)
        self.direction = nn.Embedding(data.n_directions, 4)
        self.projection = nn.Sequential(
            nn.Linear(30, dim), nn.GELU(), nn.LayerNorm(dim)
        )

    def forward(
        self, batch: dict[str, torch.Tensor], enabled: bool
    ) -> torch.Tensor:
        if not enabled:
            return torch.zeros(
                batch["station_id"].shape[0],
                32,
                device=batch["station_id"].device,
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


class FutureWeatherVSN(nn.Module):
    """TFT-style variable selection over six numeric and two categorical inputs."""

    def __init__(self, context_dim: int = 32, variable_dim: int = 24):
        super().__init__()
        self.continuous = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, variable_dim),
                    nn.GELU(),
                    nn.LayerNorm(variable_dim),
                )
                for _ in range(6)
            ]
        )
        self.pty = nn.Embedding(8, variable_dim)
        self.sky = nn.Embedding(8, variable_dim)
        self.scores = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(variable_dim + context_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, 1),
                )
                for _ in range(8)
            ]
        )
        self.output = nn.Linear(variable_dim, context_dim)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor,
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, horizon, _ = values.shape
        if not enabled:
            return (
                torch.zeros(batch_size, horizon, 32, device=values.device),
                torch.full(
                    (batch_size, horizon, 8), 1 / 8, device=values.device
                ),
            )
        variables = [
            layer(torch.stack([values[..., i], mask[..., i]], dim=-1))
            for i, layer in enumerate(self.continuous)
        ]
        pty = values[..., 6].round().long().clamp(0, 6) + 1
        sky = values[..., 7].round().long().clamp(0, 6) + 1
        pty = torch.where(mask[..., 6].bool(), pty, torch.zeros_like(pty))
        sky = torch.where(mask[..., 7].bool(), sky, torch.zeros_like(sky))
        variables.extend([self.pty(pty), self.sky(sky)])
        context = static[:, None, :].expand(-1, horizon, -1)
        score = torch.cat(
            [
                scorer(torch.cat([variable, context], dim=-1))
                for scorer, variable in zip(self.scores, variables)
            ],
            dim=-1,
        )
        weights = torch.softmax(score, dim=-1)
        selected = torch.sum(
            weights.unsqueeze(-1) * torch.stack(variables, dim=-2), dim=-2
        )
        return self.output(selected), weights


def monotonic_quantiles(raw: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    median = anchor + raw[..., 0]
    lower = median - F.softplus(raw[..., 1])
    upper = median + F.softplus(raw[..., 2])
    return torch.stack([lower, median, upper], dim=-1)


class BaseForecastModel(nn.Module):
    def __init__(self, data: ExperimentData, config: ModelConfig):
        super().__init__()
        self.config = config
        self.static = StaticContext(data)
        self.hour = nn.Embedding(data.n_hours, 8)
        self.weekday = nn.Embedding(7, 5)
        self.register_buffer("hour_ids", torch.arange(data.n_hours))

    def static_context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.static(batch, self.config.station_context)

    def anchor(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.config.weekly_skip:
            return batch["hist_y"][:, -7, :]
        return torch.zeros_like(batch["hist_y"][:, -1, :])


class EmbeddingMLP(BaseForecastModel):
    def __init__(self, data: ExperimentData, config: ModelConfig):
        super().__init__(data, config)
        self.weather = FutureWeatherVSN()
        input_dim = 4 * data.n_hours + 32 + 32 * data.n_hours + 8 * data.n_hours + 7
        self.network = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, data.n_hours * 3),
        )

    def forward(self, batch: dict[str, torch.Tensor]):
        static = self.static_context(batch)
        weather, weights = self.weather(
            batch["future"],
            batch["future_mask"],
            static,
            self.config.future_weather != "none",
        )
        lags = torch.cat(
            [
                batch["hist_y"][:, -1, :],
                batch["hist_y"][:, -7, :],
                batch["hist_y"][:, -14, :],
                batch["hist_y"][:, -28, :],
            ],
            dim=-1,
        )
        hour = self.hour(self.hour_ids).reshape(1, -1).expand(lags.shape[0], -1)
        features = torch.cat(
            [
                lags,
                static,
                weather.reshape(lags.shape[0], -1),
                hour,
                self.weekday(batch["target_weekday"]),
                batch["target_doy"],
            ],
            dim=-1,
        )
        raw = self.network(features).reshape(-1, 18, 3)
        return monotonic_quantiles(raw, self.anchor(batch)), weights


class ContextDecoder(BaseForecastModel):
    def __init__(self, data: ExperimentData, config: ModelConfig):
        super().__init__(data, config)
        self.weather = FutureWeatherVSN()
        self.query = nn.Sequential(
            nn.Linear(32 + 32 + 8 + 5 + 2, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.output = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
        )

    def future_queries(
        self,
        batch: dict[str, torch.Tensor],
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weather, weights = self.weather(
            batch["future"],
            batch["future_mask"],
            static,
            self.config.future_weather != "none",
        )
        batch_size, horizon = static.shape[0], batch["future"].shape[1]
        hour = self.hour(self.hour_ids)[None].expand(batch_size, -1, -1)
        weekday = self.weekday(batch["target_weekday"])[:, None].expand(
            -1, horizon, -1
        )
        doy = batch["target_doy"][:, None].expand(-1, horizon, -1)
        static_expanded = static[:, None].expand(-1, horizon, -1)
        query = self.query(
            torch.cat([weather, static_expanded, hour, weekday, doy], dim=-1)
        )
        return query, weights


class GRUForecast(ContextDecoder):
    def __init__(self, data: ExperimentData, config: ModelConfig):
        super().__init__(data, config)
        input_dim = 18 + (12 * 18 if config.past_weather else 0)
        self.day_projection = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.encoder = nn.GRU(
            64, 64, num_layers=2, batch_first=True, dropout=0.1
        )

    def forward(self, batch: dict[str, torch.Tensor]):
        static = self.static_context(batch)
        features = [batch["hist_y"]]
        if self.config.past_weather:
            weather = torch.cat(
                [batch["hist_obs"], batch["hist_obs_mask"]], dim=-1
            )
            features.append(weather.flatten(start_dim=2))
        encoded, _ = self.encoder(self.day_projection(torch.cat(features, dim=-1)))
        query, weights = self.future_queries(batch, static)
        raw = self.output(query + encoded[:, -1:, :])
        return monotonic_quantiles(raw, self.anchor(batch)), weights


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm = nn.GroupNorm(4, channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.dropout(F.gelu(self.norm(self.conv(values))))


class TCNForecast(ContextDecoder):
    def __init__(self, data: ExperimentData, config: ModelConfig):
        super().__init__(data, config)
        input_dim = 18 + (12 * 18 if config.past_weather else 0)
        self.projection = nn.Linear(input_dim, 64)
        self.blocks = nn.ModuleList(
            [ResidualTCNBlock(64, dilation) for dilation in (1, 2, 4, 8)]
        )

    def forward(self, batch: dict[str, torch.Tensor]):
        static = self.static_context(batch)
        features = [batch["hist_y"]]
        if self.config.past_weather:
            weather = torch.cat(
                [batch["hist_obs"], batch["hist_obs_mask"]], dim=-1
            )
            features.append(weather.flatten(start_dim=2))
        encoded = self.projection(torch.cat(features, dim=-1)).transpose(1, 2)
        for block in self.blocks:
            encoded = block(encoded)
        query, weights = self.future_queries(batch, static)
        raw = self.output(query + encoded[:, :, -1].unsqueeze(1))
        return monotonic_quantiles(raw, self.anchor(batch)), weights


class FusionForecast(BaseForecastModel):
    """Hierarchical intra-day/inter-day encoder with weather-query attention."""

    def __init__(self, data: ExperimentData, config: ModelConfig):
        super().__init__(data, config)
        self.weather = FutureWeatherVSN()
        hist_dim = 1 + (12 if config.past_weather else 0)
        self.intra = nn.Sequential(
            nn.Conv1d(hist_dim, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.GELU(),
        )
        self.pool_scores = nn.Parameter(torch.zeros(data.n_hours))
        self.hist_weekday = nn.Embedding(7, 5)
        self.day_projection = nn.Linear(32 + 5 + 2, 64)
        self.inter = nn.GRU(
            64, 64, num_layers=2, batch_first=True, dropout=0.1
        )
        self.query = nn.Sequential(
            nn.Linear(32 + 32 + 8 + 5 + 2, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.film = nn.Linear(32, 128)
        self.cross = nn.MultiheadAttention(
            64, num_heads=4, dropout=0.1, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
        )
        self.output = nn.Linear(64, 3)

    def forward(self, batch: dict[str, torch.Tensor]):
        static = self.static_context(batch)
        hist = [batch["hist_y"].unsqueeze(-1)]
        if self.config.past_weather:
            hist.extend([batch["hist_obs"], batch["hist_obs_mask"]])
        hist_tensor = torch.cat(hist, dim=-1)
        batch_size, days, hours, features = hist_tensor.shape
        intra = self.intra(
            hist_tensor.reshape(batch_size * days, hours, features).transpose(1, 2)
        ).transpose(1, 2)
        pool = torch.softmax(self.pool_scores, dim=0)
        day_tokens = torch.sum(intra * pool[None, :, None], dim=1).reshape(
            batch_size, days, 32
        )
        day_tokens = self.day_projection(
            torch.cat(
                [
                    day_tokens,
                    self.hist_weekday(batch["hist_weekday"]),
                    batch["hist_doy"],
                ],
                dim=-1,
            )
        )
        memory, _ = self.inter(day_tokens)

        weather, weights = self.weather(
            batch["future"],
            batch["future_mask"],
            static,
            self.config.future_weather != "none",
        )
        hour = self.hour(self.hour_ids)[None].expand(batch_size, -1, -1)
        weekday = self.weekday(batch["target_weekday"])[:, None].expand(
            -1, hours, -1
        )
        doy = batch["target_doy"][:, None].expand(-1, hours, -1)
        query = self.query(
            torch.cat(
                [
                    weather,
                    static[:, None].expand(-1, hours, -1),
                    hour,
                    weekday,
                    doy,
                ],
                dim=-1,
            )
        )
        gamma, beta = self.film(static).chunk(2, dim=-1)
        query = (
            query * (1 + 0.1 * torch.tanh(gamma[:, None]))
            + beta[:, None]
        )
        if self.config.cross_attention:
            attended, attention = self.cross(
                query,
                memory,
                memory,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attended = memory[:, -1:, :].expand(-1, hours, -1)
            attention = torch.zeros(
                batch_size, 4, hours, days, device=query.device
            )
        hidden = query + attended
        hidden = hidden + self.fusion(hidden)
        raw = self.output(hidden)
        return (
            monotonic_quantiles(raw, self.anchor(batch)),
            weights,
            attention,
        )


def build_model(data: ExperimentData, config: ModelConfig) -> nn.Module:
    if config.architecture == "mlp":
        return EmbeddingMLP(data, config)
    if config.architecture == "gru":
        return GRUForecast(data, config)
    if config.architecture == "tcn":
        return TCNForecast(data, config)
    if config.architecture == "fusion":
        return FusionForecast(data, config)
    raise ValueError(config.architecture)


def quantile_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = target.unsqueeze(-1) - prediction
    quantiles = torch.tensor(QUANTILES, device=prediction.device)
    pinball = torch.maximum(
        quantiles * error, (quantiles - 1) * error
    ).mean()
    return pinball + 0.2 * F.huber_loss(
        prediction[..., 1], target, delta=0.25
    )


def inverse_prediction(
    prediction_z: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    return np.maximum(0.0, scales[:, None] * np.expm1(prediction_z))


def metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict:
    error = prediction - actual
    abs_error = np.abs(error)
    result = {
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(abs_error.sum() / max(float(actual.sum()), 1.0)),
        "smape": float(
            np.mean(
                2
                * abs_error
                / np.maximum(np.abs(actual) + np.abs(prediction), 1.0)
            )
        ),
        "bias": float(error.mean()),
        "daily_mae": float(
            np.mean(np.abs(prediction.sum(axis=1) - actual.sum(axis=1)))
        ),
        "median_absolute_error": float(np.median(abs_error)),
        "p90_absolute_error": float(np.quantile(abs_error, 0.9)),
        "negative_predictions": int((prediction < 0).sum()),
        "hourly_mae": [float(value) for value in abs_error.mean(axis=0)],
    }
    if lower is not None and upper is not None:
        result["interval_coverage_80"] = float(
            np.mean((actual >= lower) & (actual <= upper))
        )
        result["interval_width_80"] = float(np.mean(upper - lower))
    return result


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: ExperimentData,
    target_days: np.ndarray,
    config: ModelConfig,
    device: torch.device,
    batch_size: int,
    weather_day_map: np.ndarray | None,
) -> tuple[dict, dict[str, np.ndarray]]:
    model.eval()
    total = len(target_days) * data.n_series
    actual_parts, quantile_parts = [], []
    day_parts, series_parts, weather_parts = [], [], []
    attention_parts = []
    for start in range(0, total, batch_size):
        indices = np.arange(start, min(total, start + batch_size))
        batch = data.make_batch(
            indices, target_days, config, device, weather_day_map
        )
        output = model(batch)
        quantile_parts.append(output[0].cpu().numpy())
        actual_parts.append(batch["target_raw"].cpu().numpy())
        day_parts.append(batch["days"].cpu().numpy())
        series_parts.append(batch["series"].cpu().numpy())
        weather_parts.append(output[1].cpu().numpy())
        if len(output) > 2 and len(attention_parts) < 4:
            attention_parts.append(output[2].cpu().numpy())
    quantiles_z = np.concatenate(quantile_parts)
    actual = np.concatenate(actual_parts)
    series = np.concatenate(series_parts)
    scales = data.series_scale[series]
    lower = inverse_prediction(quantiles_z[..., 0], scales)
    median = inverse_prediction(quantiles_z[..., 1], scales)
    upper = inverse_prediction(quantiles_z[..., 2], scales)
    result = metrics(actual, median, lower, upper)
    result["series_macro_mae"] = float(
        np.mean(
            [
                np.abs(median[series == index] - actual[series == index]).mean()
                for index in range(data.n_series)
            ]
        )
    )
    artifacts = {
        "actual": actual.astype(np.float32),
        "prediction": median.astype(np.float32),
        "lower": lower.astype(np.float32),
        "upper": upper.astype(np.float32),
        "day_index": np.concatenate(day_parts).astype(np.int16),
        "series_index": series.astype(np.int16),
        "weather_weights": np.concatenate(weather_parts).astype(np.float32),
    }
    if attention_parts:
        artifacts["attention_sample"] = np.concatenate(attention_parts).astype(
            np.float32
        )
    return result, artifacts


def build_weather_map(
    data: ExperimentData,
    config: ModelConfig,
    train_days: np.ndarray,
    validation_days: np.ndarray,
) -> np.ndarray | None:
    if config.future_weather != "shuffled":
        return None
    mapping = np.arange(len(data.dates), dtype=np.int64)
    generator = np.random.default_rng(config.seed)
    for values in (train_days, validation_days):
        shuffled = values.copy()
        generator.shuffle(shuffled)
        mapping[values] = shuffled
    return mapping


def train_model(
    data: ExperimentData,
    config: ModelConfig,
    train_days: np.ndarray,
    validation_days: np.ndarray,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[nn.Module, list[dict], dict]:
    set_seed(config.seed)
    model = build_model(data, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    weather_map = build_weather_map(
        data, config, train_days, validation_days
    )
    total = len(train_days) * data.n_series
    best_state, best_mae = None, math.inf
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = np.random.default_rng(
            config.seed + epoch
        ).permutation(total)
        running_loss, seen = 0.0, 0
        for start in range(0, total, batch_size):
            indices = permutation[start : start + batch_size]
            batch = data.make_batch(
                indices, train_days, config, device, weather_map
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)[0]
            loss = quantile_loss(prediction, batch["target_z"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(indices)
            seen += len(indices)
        scheduler.step()
        validation_metrics, _ = evaluate_model(
            model,
            data,
            validation_days,
            config,
            device,
            batch_size,
            weather_map,
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "validation_mae": validation_metrics["mae"],
            "validation_wape": validation_metrics["wape"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"{config.name} epoch={epoch:02d} "
            f"loss={row['train_loss']:.5f} "
            f"val_mae={row['validation_mae']:.2f}",
            flush=True,
        )
        if validation_metrics["mae"] < best_mae - 0.05:
            best_mae = validation_metrics["mae"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No checkpoint was captured")
    model.load_state_dict(best_state)
    validation_metrics, _ = evaluate_model(
        model,
        data,
        validation_days,
        config,
        device,
        batch_size,
        weather_map,
    )
    validation_metrics["best_epoch"] = int(
        min(history, key=lambda row: row["validation_mae"])["epoch"]
    )
    validation_metrics["parameters"] = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    validation_metrics["elapsed_seconds"] = float(time.time() - started)
    return model, history, validation_metrics


def baseline_predictions(
    data: ExperimentData,
    target_days: np.ndarray,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    days = np.repeat(target_days, data.n_series)
    series = np.tile(np.arange(data.n_series), len(target_days))
    actual = data.demand_raw[days, series]
    if kind == "previous_day":
        prediction = data.demand_raw[days - 1, series]
    elif kind == "seasonal_naive":
        prediction = data.demand_raw[days - 7, series]
    elif kind == "four_week_median":
        prediction = np.median(
            np.stack(
                [
                    data.demand_raw[days - lag, series]
                    for lag in (7, 14, 21, 28)
                ]
            ),
            axis=0,
        )
    else:
        raise ValueError(kind)
    return actual, prediction, days, series


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def unique_configs(configs: list[ModelConfig]) -> list[ModelConfig]:
    result, names = [], set()
    for config in configs:
        if config.name not in names:
            names.add(config.name)
            result.append(config)
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}", flush=True)
    if device.type != "cuda":
        raise RuntimeError("GPU is required for the requested experiment suite")

    raw_dates = np.load(args.data)["dates"]
    fold_definitions = [
        ("fold_aug", "2025-01-29", "2025-07-31", "2025-08-01", "2025-08-14"),
        ("fold_sep", "2025-01-29", "2025-08-31", "2025-09-01", "2025-09-14"),
        ("fold_oct", "2025-01-29", "2025-09-30", "2025-10-01", "2025-10-14"),
        ("fold_nov", "2025-01-29", "2025-10-31", "2025-11-01", "2025-11-14"),
    ]
    final_train = target_range(raw_dates, "2025-01-29", "2025-10-31")
    final_validation = target_range(
        raw_dates, "2025-11-01", "2025-11-14"
    )
    test_days = target_range(raw_dates, "2025-11-15", "2025-12-31")
    data = ExperimentData(args.data, train_end=int(final_train.max()))

    results: list[dict] = []
    artifacts: dict[str, np.ndarray] = {
        "dates": data.dates,
        "service_hours": data.hours,
        "series_station": data.series_names,
        "series_line": data.series_lines,
        "series_direction": data.series_directions,
    }
    for split_name, split_days in (
        ("validation", final_validation),
        ("test", test_days),
    ):
        for baseline in (
            "previous_day",
            "seasonal_naive",
            "four_week_median",
        ):
            actual, prediction, day_ids, series_ids = baseline_predictions(
                data, split_days, baseline
            )
            results.append(
                {
                    "name": baseline,
                    "family": "statistical",
                    "split": split_name,
                    **metrics(actual, prediction),
                }
            )
            if split_name == "test":
                artifacts[f"{baseline}__prediction"] = prediction.astype(
                    np.float32
                )
                artifacts["test_actual"] = actual.astype(np.float32)
                artifacts["test_day_index"] = day_ids.astype(np.int16)
                artifacts["test_series_index"] = series_ids.astype(np.int16)

    architectures = [
        ModelConfig("embedding_mlp", "mlp"),
        ModelConfig("gru_seq2seq", "gru"),
        ModelConfig("dilated_tcn", "tcn"),
        ModelConfig("hierarchical_fusion", "fusion"),
    ]
    ablations = [
        ModelConfig(
            "fusion_no_weather",
            "fusion",
            past_weather=False,
            future_weather="none",
        ),
        ModelConfig(
            "fusion_past_weather_only",
            "fusion",
            past_weather=True,
            future_weather="none",
        ),
        ModelConfig(
            "fusion_future_weather_only",
            "fusion",
            past_weather=False,
            future_weather="forecast",
        ),
        ModelConfig(
            "fusion_no_station", "fusion", station_context=False
        ),
        ModelConfig(
            "fusion_no_cross_attention",
            "fusion",
            cross_attention=False,
        ),
        ModelConfig(
            "fusion_no_weekly_skip", "fusion", weekly_skip=False
        ),
        ModelConfig(
            "fusion_shuffled_forecast",
            "fusion",
            future_weather="shuffled",
        ),
        ModelConfig(
            "fusion_oracle_weather", "fusion", future_weather="oracle"
        ),
    ]
    if args.tcn_ablations_only:
        architectures = []
        ablations = [
            ModelConfig(
                "tcn_no_weather",
                "tcn",
                past_weather=False,
                future_weather="none",
            ),
            ModelConfig(
                "tcn_past_weather_only",
                "tcn",
                past_weather=True,
                future_weather="none",
            ),
            ModelConfig(
                "tcn_future_weather_only",
                "tcn",
                past_weather=False,
                future_weather="forecast",
            ),
            ModelConfig(
                "tcn_shuffled_forecast",
                "tcn",
                future_weather="shuffled",
            ),
            ModelConfig(
                "tcn_oracle_weather",
                "tcn",
                future_weather="oracle",
            ),
            ModelConfig(
                "tcn_no_station",
                "tcn",
                station_context=False,
            ),
            ModelConfig(
                "tcn_no_weekly_skip",
                "tcn",
                weekly_skip=False,
            ),
            ModelConfig("tcn_seed_123", "tcn", seed=123),
            ModelConfig("tcn_seed_2025", "tcn", seed=2025),
        ]
    if args.quick:
        architectures = architectures[-2:]
        ablations = ablations[:3]
        args.epochs = min(args.epochs, 5)
        args.patience = min(args.patience, 2)

    if args.architecture_folds and not args.quick:
        for fold_name, train_start, train_end, val_start, val_end in fold_definitions:
            fold_train = target_range(data.dates, train_start, train_end)
            fold_validation = target_range(data.dates, val_start, val_end)
            fold_data = ExperimentData(
                args.data, train_end=int(fold_train.max())
            )
            for config in architectures:
                model, history, validation_metrics = train_model(
                    fold_data,
                    config,
                    fold_train,
                    fold_validation,
                    device,
                    args.epochs,
                    args.patience,
                    args.batch_size,
                )
                results.append(
                    {
                        "name": config.name,
                        "family": "deep_learning",
                        "split": fold_name,
                        "config": asdict(config),
                        **validation_metrics,
                    }
                )
                save_json(
                    args.output_dir
                    / f"history__{config.name}__{fold_name}.json",
                    history,
                )
                del model
                torch.cuda.empty_cache()

    for config in unique_configs(architectures + ablations):
        model, history, validation_metrics = train_model(
            data,
            config,
            final_train,
            final_validation,
            device,
            args.epochs,
            args.patience,
            args.batch_size,
        )
        results.append(
            {
                "name": config.name,
                "family": "deep_learning",
                "split": "validation",
                "config": asdict(config),
                **validation_metrics,
            }
        )
        weather_map = build_weather_map(
            data,
            config,
            final_train,
            np.concatenate([final_validation, test_days]),
        )
        test_metrics, test_artifacts = evaluate_model(
            model,
            data,
            test_days,
            config,
            device,
            args.batch_size,
            weather_map,
        )
        results.append(
            {
                "name": config.name,
                "family": "deep_learning",
                "split": "test",
                "config": asdict(config),
                **test_metrics,
            }
        )
        save_json(
            args.output_dir / f"history__{config.name}.json", history
        )
        for key, value in test_artifacts.items():
            if key in {"actual", "day_index", "series_index"}:
                continue
            if (
                key in {"weather_weights", "attention_sample"}
                and config.name != "hierarchical_fusion"
            ):
                continue
            artifacts[f"{config.name}__{key}"] = value
        if config.name == "hierarchical_fusion":
            artifacts[
                "hierarchical_fusion__station_embedding"
            ] = model.static.station.weight.detach().cpu().numpy().astype(
                np.float32
            )
            artifacts[
                "hierarchical_fusion__line_embedding"
            ] = model.static.line.weight.detach().cpu().numpy().astype(
                np.float32
            )
        save_json(args.output_dir / "results.json", results)
        np.savez_compressed(
            args.output_dir / "predictions.npz", **artifacts
        )
        del model
        torch.cuda.empty_cache()

    audit = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "contextDays": CONTEXT_DAYS,
        "targetHours": data.hours.tolist(),
        "series": data.n_series,
        "finalTrainDates": [
            str(data.dates[final_train.min()]),
            str(data.dates[final_train.max()]),
        ],
        "validationDates": [
            str(data.dates[final_validation.min()]),
            str(data.dates[final_validation.max()]),
        ],
        "testDates": [
            str(data.dates[test_days.min()]),
            str(data.dates[test_days.max()]),
        ],
        "epochs": args.epochs,
        "patience": args.patience,
        "batchSize": args.batch_size,
    }
    save_json(args.output_dir / "run_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
