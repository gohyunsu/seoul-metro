#!/usr/bin/env python3
"""Build the private hourly ridership/weather tensor bundle used by the report.

The source weather archive contains repeated full-year blocks.  Forecast issue
timestamps are UTC, while the ASOS and ridership timestamps are interpreted in
KST.  For every target service hour, only forecasts issued no later than the
05:10 KST operational cutoff are retained.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RIDERSHIP_HOURS = [f"{hour:02d}-{hour + 1:02d}시간대" for hour in range(6, 24)]
SERVICE_HOURS = np.arange(6, 24, dtype=np.int8)
KST_OFFSET = pd.Timedelta(hours=9)
FORECAST_CUTOFF = pd.Timedelta(hours=5, minutes=10)

FORECAST_FILES = {
    "temperature": "forecast/1시간기온/교남동_1시간기온_20250101_20251231.csv",
    "precipitation": "forecast/1시간강수량/교남동_1시간강수량_20250101_20251231.csv",
    "snow": "forecast/1시간적설/교남동_1시간적설_20250101_20251231.csv",
    "precip_probability": "forecast/강수확률/교남동_강수확률_20250101_20251231.csv",
    "humidity": "forecast/습도/교남동_습도_20250101_20251231.csv",
    "wind_speed": "forecast/풍속/교남동_풍속_20250101_20251231.csv",
    "precip_type": "forecast/강수형태/교남동_강수형태_20250101_20251231.csv",
    "sky": "forecast/하늘상태/교남동_하늘상태_20250101_20251231.csv",
}

ASOS_FEATURES = {
    "temperature": "기온(°C)",
    "precipitation": "강수량(mm)",
    "wind_speed": "풍속(m/s)",
    "humidity": "습도(%)",
    "snow": "적설(cm)",
    "cloud": "전운량(10분위)",
}


@dataclass
class ForecastAudit:
    raw_rows: int
    unique_rows: int
    duplicate_rows: int
    inferred_year_copies: int
    sentinel_missing: int
    selected_rows: int
    selected_coverage: float
    median_lead_hours: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ridership",
        type=Path,
        default=Path("data/raw/seoul_metro_ridership_2025.csv"),
    )
    parser.add_argument(
        "--weather-zip",
        type=Path,
        default=Path("data/raw/drive-download-20260725T073510Z-1-001.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission/hourly_experiment_data.npz"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("submission/hourly_data_audit.json"),
    )
    return parser.parse_args()


def read_ridership(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    frame = pd.read_csv(path, encoding="cp949")
    date_column = "수송일자" if "수송일자" in frame.columns else "사용일자"
    line_column = "호선" if "호선" in frame.columns else "호선명"
    required = {date_column, line_column, "역번호", "역명", "승하차구분", *RIDERSHIP_HOURS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Ridership columns missing: {missing}")

    frame = frame.rename(
        columns={
            date_column: "date",
            line_column: "line",
            "역번호": "station_code",
            "역명": "station",
            "승하차구분": "direction",
        }
    )
    blank_source_rows = int(
        frame[["date", "line", "station_code", "station", "direction"]]
        .isna()
        .all(axis=1)
        .sum()
    )
    frame = frame.dropna(
        subset=["date", "line", "station_code", "station", "direction"]
    ).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["station_code"] = (
        frame["station_code"].astype("string").str.replace(r"\.0$", "", regex=True)
    )
    series_columns = ["line", "station_code", "direction"]
    # Station name is display metadata: a rename must not create a new series.
    meta = (
        frame.sort_values("date", kind="stable")
        .groupby(series_columns, as_index=False, sort=False)
        .agg(station=("station", "last"))
        .sort_values(series_columns, kind="stable")
        .reset_index(drop=True)
    )
    meta["series_id"] = np.arange(len(meta), dtype=np.int32)
    frame = frame.merge(
        meta[series_columns + ["series_id"]],
        on=series_columns,
        how="left",
        validate="many_to_one",
    )

    duplicates = int(frame.duplicated(["date", "series_id"]).sum())
    if duplicates:
        raise ValueError(f"Unexpected duplicate ridership date-series rows: {duplicates}")

    dates = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
    full_index = pd.MultiIndex.from_product(
        [dates, meta["series_id"]], names=["date", "series_id"]
    )
    matrix = (
        frame.set_index(["date", "series_id"])[RIDERSHIP_HOURS]
        .apply(pd.to_numeric, errors="coerce")
        .reindex(full_index)
    )
    demand = matrix.to_numpy(dtype=np.float32).reshape(len(dates), len(meta), len(SERVICE_HOURS))

    line_categories = sorted(meta["line"].unique())
    direction_categories = sorted(meta["direction"].unique())
    station_categories = sorted(
        meta[["line", "station_code"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    line_lookup = {value: index for index, value in enumerate(line_categories)}
    direction_lookup = {value: index for index, value in enumerate(direction_categories)}
    station_lookup = {value: index for index, value in enumerate(station_categories)}

    series_meta = {
        "series_line_id": meta["line"].map(line_lookup).to_numpy(dtype=np.int16),
        "series_station_id": np.asarray(
            [station_lookup[(row.line, row.station_code)] for row in meta.itertuples()],
            dtype=np.int16,
        ),
        "series_direction_id": meta["direction"].map(direction_lookup).to_numpy(dtype=np.int8),
        "series_line": meta["line"].astype(str).to_numpy(dtype="U16"),
        "series_station_code": meta["station_code"].astype(str).to_numpy(dtype="U16"),
        "series_station": meta["station"].astype(str).to_numpy(dtype="U64"),
        "series_direction": meta["direction"].astype(str).to_numpy(dtype="U8"),
    }
    audit = {
        "sourceRows": int(len(frame)),
        "blankSourceRowsRemoved": blank_source_rows,
        "dateStart": dates.min().strftime("%Y-%m-%d"),
        "dateEnd": dates.max().strftime("%Y-%m-%d"),
        "dates": int(len(dates)),
        "series": int(len(meta)),
        "stations": int(len(station_categories)),
        "lines": int(len(line_categories)),
        "directions": direction_categories,
        "serviceHours": SERVICE_HOURS.tolist(),
        "regularTargetCells": int(np.prod(demand.shape)),
        "missingTargetCells": int(np.isnan(demand).sum()),
        "excludedColumns": ["06시이전", "24시이후"],
    }
    return dates.to_numpy(dtype="datetime64[D]"), demand, series_meta, audit


def parse_forecast_member(blob: bytes) -> tuple[pd.DataFrame, int, int, int]:
    text = blob.decode("ascii", errors="replace")
    start_pattern = re.compile(r"Start\s*:\s*(\d{8})")
    data_pattern = re.compile(
        r"^\s*(\d+),(\d{4}),\+?(\d+),\s*([-+]?\d+(?:\.\d+)?)\s*$"
    )
    current_start: pd.Timestamp | None = None
    records: list[tuple[pd.Timestamp, int, str, int, float]] = []
    header_dates: list[pd.Timestamp] = []
    for line in text.splitlines():
        start_match = start_pattern.search(line)
        if start_match:
            current_start = pd.to_datetime(start_match.group(1), format="%Y%m%d")
            header_dates.append(current_start)
        data_match = data_pattern.match(line)
        if data_match:
            if current_start is None:
                raise ValueError("Forecast row appeared before a Start date header")
            records.append(
                (
                    current_start,
                    int(data_match.group(1)),
                    data_match.group(2),
                    int(data_match.group(3)),
                    float(data_match.group(4)),
                )
            )
    if not records:
        raise ValueError("No forecast rows parsed")
    frame = pd.DataFrame(
        records,
        columns=["start_date", "day", "hour", "lead_hours", "value"],
    )
    header_series = pd.Series(header_dates)
    header_counts = header_series.value_counts()
    copies = int(header_counts.max())
    if int(frame["day"].ne(frame["start_date"].dt.day).sum()):
        raise ValueError("Forecast row day does not agree with its Start date header")
    sentinel_missing = int(np.isclose(frame["value"], -999.9, equal_nan=False).sum())
    frame.loc[np.isclose(frame["value"], -999.9, equal_nan=False), "value"] = np.nan
    hour_text = frame["hour"].astype(str).str.zfill(4)
    frame["issue_utc"] = (
        frame["start_date"]
        + pd.to_timedelta(hour_text.str.slice(0, 2).astype(int), unit="h")
        + pd.to_timedelta(hour_text.str.slice(2, 4).astype(int), unit="m")
    )
    frame["issue_kst"] = frame["issue_utc"] + KST_OFFSET
    frame["valid_kst"] = (
        frame["issue_utc"]
        + pd.to_timedelta(frame["lead_hours"], unit="h")
        + KST_OFFSET
    )
    frame["valid_hour_kst"] = frame["valid_kst"].dt.floor("h")

    raw_rows = len(frame)
    frame = (
        frame.sort_values(["issue_utc", "lead_hours"], kind="stable")
        .drop_duplicates(["issue_utc", "lead_hours"], keep="first")
        .reset_index(drop=True)
    )
    return frame, copies, sentinel_missing, raw_rows


def select_operational_forecasts(
    frame: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    targets = pd.DataFrame({"valid_hour_kst": target_index})
    targets["target_date"] = targets["valid_hour_kst"].dt.normalize()
    targets["cutoff_kst"] = targets["target_date"] + FORECAST_CUTOFF
    eligible = frame.merge(
        targets[["valid_hour_kst", "cutoff_kst"]],
        on="valid_hour_kst",
        how="inner",
        validate="many_to_many",
    )
    eligible = eligible[eligible["issue_kst"] <= eligible["cutoff_kst"]]
    selected = (
        eligible.sort_values(["valid_hour_kst", "issue_kst"], kind="stable")
        .drop_duplicates("valid_hour_kst", keep="last")
        .set_index("valid_hour_kst")
        .reindex(target_index)
    )
    selected["selected_lead_hours"] = (
        selected.index.to_series().to_numpy() - selected["issue_kst"]
    ) / pd.Timedelta(hours=1)
    return selected


def read_weather(
    archive_path: Path,
    dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    date_index = pd.DatetimeIndex(dates.astype("datetime64[ns]"))
    target_index = pd.DatetimeIndex(
        [
            date + pd.Timedelta(hours=int(hour))
            for date in date_index
            for hour in SERVICE_HOURS
        ]
    )
    forecast_values = np.full(
        (len(date_index), len(SERVICE_HOURS), len(FORECAST_FILES)),
        np.nan,
        dtype=np.float32,
    )
    forecast_leads = np.full_like(forecast_values, np.nan)
    forecast_audit: dict[str, dict] = {}

    with zipfile.ZipFile(archive_path) as archive:
        for feature_index, (feature, member) in enumerate(FORECAST_FILES.items()):
            parsed, copies, sentinel_missing, raw_rows = parse_forecast_member(
                archive.read(member)
            )
            selected = select_operational_forecasts(parsed, target_index)
            forecast_values[:, :, feature_index] = selected["value"].to_numpy(
                dtype=np.float32
            ).reshape(len(date_index), len(SERVICE_HOURS))
            forecast_leads[:, :, feature_index] = selected[
                "selected_lead_hours"
            ].to_numpy(dtype=np.float32).reshape(len(date_index), len(SERVICE_HOURS))
            coverage = float(selected["value"].notna().mean())
            median_lead = (
                float(selected["selected_lead_hours"].median())
                if selected["selected_lead_hours"].notna().any()
                else None
            )
            forecast_audit[feature] = ForecastAudit(
                raw_rows=raw_rows,
                unique_rows=int(len(parsed)),
                duplicate_rows=int(raw_rows - len(parsed)),
                inferred_year_copies=copies,
                sentinel_missing=sentinel_missing,
                selected_rows=int(selected["value"].notna().sum()),
                selected_coverage=coverage,
                median_lead_hours=median_lead,
            ).__dict__

        asos = pd.read_csv(
            io.BytesIO(archive.read("data_ASOS.csv")),
            encoding="cp949",
        )
    asos["일시"] = pd.to_datetime(asos["일시"])
    asos = asos.set_index("일시").sort_index()
    observed = np.full(
        (len(date_index), len(SERVICE_HOURS), len(ASOS_FEATURES)),
        np.nan,
        dtype=np.float32,
    )
    for feature_index, (feature, column) in enumerate(ASOS_FEATURES.items()):
        values = pd.to_numeric(asos[column], errors="coerce").reindex(target_index)
        if feature in {"precipitation", "snow"}:
            values = values.fillna(0.0)
        observed[:, :, feature_index] = values.to_numpy(dtype=np.float32).reshape(
            len(date_index), len(SERVICE_HOURS)
        )

    audit = {
        "forecastTimezone": "issue UTC; valid time converted to KST (UTC+09:00)",
        "operationalCutoffKST": "05:10",
        "forecastFeatureOrder": list(FORECAST_FILES),
        "observedFeatureOrder": list(ASOS_FEATURES),
        "forecast": forecast_audit,
        "forecastMissingCells": int(np.isnan(forecast_values).sum()),
        "observedMissingCells": int(np.isnan(observed).sum()),
    }
    return observed, forecast_values, forecast_leads, audit


def build_calendar(dates: np.ndarray) -> dict[str, np.ndarray]:
    index = pd.DatetimeIndex(dates.astype("datetime64[ns]"))
    day_of_year = index.dayofyear.to_numpy()
    return {
        "weekday": index.dayofweek.to_numpy(dtype=np.int8),
        "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32),
        "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32),
        # The following are analysis strata, not model-selection inputs.
        "analysis_weekend": np.asarray(index.dayofweek >= 5, dtype=np.int8),
        "analysis_month": index.month.to_numpy(dtype=np.int8),
    }


def main() -> None:
    args = parse_args()
    dates, demand, series_meta, ridership_audit = read_ridership(args.ridership)
    observed, forecast, forecast_leads, weather_audit = read_weather(
        args.weather_zip, dates
    )
    calendar = build_calendar(dates)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        dates=dates,
        demand=demand,
        service_hours=SERVICE_HOURS,
        observed_weather=observed,
        forecast_weather=forecast,
        forecast_lead_hours=forecast_leads,
        forecast_feature_names=np.asarray(list(FORECAST_FILES), dtype="U32"),
        observed_feature_names=np.asarray(list(ASOS_FEATURES), dtype="U32"),
        **series_meta,
        **calendar,
    )
    audit = {
        "ridership": ridership_audit,
        "weather": weather_audit,
        "output": str(args.output),
        "outputBytes": args.output.stat().st_size,
    }
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
