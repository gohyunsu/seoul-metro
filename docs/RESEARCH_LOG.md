# Research Log

## 2026-07-25

- Rebuilt the target as 18 hourly outputs for 546 line-station-direction
  series, yielding 3,587,220 audited target cells with no missing targets.
- Aligned eight target-day forecast variables to an operational 05:10 KST
  cutoff and retained missingness masks after removing repeated forecast
  blocks.
- Diagnosed the first deep-learning suite: its lag-7 anchor was weaker than the
  four-week median, its nominal TCN did not directly cover all advertised
  weekly lags, past weather dominated the input dimension, and its quantile/log
  objective was misaligned with raw-passenger MAE.
- Added `scripts/run_hourly_v2_experiments.py`: an exact four-week anchor,
  per-series-hour scale, shared strictly causal 56-day encoder, explicit weekly
  lags, bounded residuals, optional gate, isolated special-day correction, and
  target-day forecast ablations.
- Four-week-median test performance: MAE 78.51, RMSE 228.32.
- Composite-validation selection chose the combined weather/special-day model
  without the sigmoid gate: test MAE 71.11, RMSE 167.67, WAPE 7.58%.
- Three selected-model seeds produced test MAE 71.11, 74.46, and 71.89
  (mean 72.49, sample standard deviation 1.75); all were below the anchor.
- Weather improved the causal base from 76.54 to 75.40 MAE, but degraded the
  special-day model from 70.65 to 72.52, so weather is not claimed to be
  uniformly beneficial.
- Kept the assignment DOCX, PDF preview, source weather archive, predictions,
  and report builder under the Git-ignored `submission/` directory.

## 2026-07-21

- Confirmed the source page and dataset identity OA-12921.
- Selected the latest available source file, dated through 2025-12-31.
- Downloaded the official CSV through the source page's file endpoint.
- Recorded a source snapshot of 25,029,452 bytes and SHA-256 `99591265daf498015499f03913cd3253a57c3ff976ae182f9c4af89fcca4e5ba`.
- Found CP949 encoding, 26 columns, 199,424 physical rows, and 134 completely blank rows.
- Found 199,290 valid rows, 365 dates, 243 stations, 8 lines, and 20 time bands.
- Found no negative values and no duplicate logical keys after blank-row removal.
- Confirmed project task: next-calendar-day daily demand forecasting for each station × line × boarding direction series, supported by a concrete EDA lab.
- EDA result: weekday network mean is 49.6% higher than weekend mean; the 18–19 time band is the annual peak; the top ten stations account for 15.4% of total volume.
- Model result: the seven-day seasonal-naive baseline remains selected on validation and reaches test MAE 1,008 passengers and WAPE 5.9% for 2025-11-28 through 2025-12-31.

## Decision register

| Decision | Reason | Revisit when |
| --- | --- | --- |
| Use the latest annual file as the primary snapshot | It is the newest file available from the requested dataset | A newer annual file is published |
| Remove only fully blank rows | No evidence supports altering valid zero counts | A source revision changes the blank-row pattern |
| Use chronological evaluation | Random splits leak future patterns into training | Never for the final time-dependent result |
| Keep the raw file unchanged | Enables provenance and exact rebuilds | Only if the provider republishes the same snapshot |
| Preserve the four-week median as the V2 anchor | It is stronger and more stable than the lag-7 neural anchor | A multi-year learned baseline beats it |
| Isolate special-day correction | Holiday residuals should not shift ordinary-day predictions | A larger event dataset supports a shared head |
| Select by ordinary and stress validation | Ordinary validation contains no public holiday | Replace with nested multi-year rolling validation |
