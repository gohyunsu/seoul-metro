# Methodology

## Research question

Can recent station-level patterns and calendar context estimate the next calendar day's passenger volume for each station, line, and boarding direction?

## Target and unit of analysis

The unit is a station-line-direction-day series. The target is `daily_total`, the sum of passenger counts across the 20 time bands on the next calendar date. Time-band behavior remains central to the descriptive analysis, while the daily target keeps the forecasting experiment efficient and directly interpretable.

## Features

- Calendar: weekday, weekend, month, week-of-year, day-of-month, day-of-year.
- Identity: line, station, station number, and boarding direction.
- History: lag 1, lag 7, lag 14, lag 28 and rolling mean features within each station-line-direction series.
- Optional later extension: holiday, event, disruption, and transfer-volume features.

The source-wide table is retained for modeling and `daily_total` is added. For EDA, the 20 time-band columns are reshaped into a long view. Calendar and historical features are generated only after sorting each station-line-direction series by date. Rows without the required history are excluded from model training, never imputed from future observations.

## Split

Use the earliest 80% of dates for training, the next 10% for validation, and the final 10% for the untouched test period. Hyperparameters are selected using the validation period. The final test score is reported once for the selected configuration.

## Model ladder

1. Seasonal-naive: use the same series' value seven days earlier.
2. Ridge regression: a stable, interpretable linear reference after one-hot encoding identities and scaling numeric features.
3. HistGradientBoostingRegressor: a non-linear model able to capture interactions between calendar signals, station identity, and recent demand.

The goal is not to maximize a leaderboard score. It is to show whether additional model complexity produces a meaningful and stable improvement over a defensible baseline.

## Metrics

- MAE: average absolute error in passengers, easy to interpret operationally.
- RMSE: penalizes large misses more heavily.
- WAPE: total absolute error divided by total observed volume, useful across demand scales.
- sMAPE: symmetric percentage error with a zero-safe denominator.

Metrics are shown overall and by line, boarding direction, weekday/weekend, and demand quantile. A model that performs well only on large stations will not be presented as uniformly reliable.

## Visual analysis set

1. Daily network demand trend with weekday/weekend annotation.
2. Line share and station ranking by annual passenger volume.
3. Hour × weekday heatmap for network demand.
4. Boarding versus alighting profiles by time band.
5. Station profile small multiples for selected high-volume and contrasting stations.
6. Predicted-versus-observed and residual distribution on the test period.

## Ethical and practical boundaries

The forecast is a planning aid, not a promise of exact future demand. It should not be used to infer individual movement, because the source is aggregate. Any operational deployment would require monitoring drift, validating service changes, and adding external context that explains exceptional dates.
