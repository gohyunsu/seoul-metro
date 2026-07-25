# Methodology

## Research question

Can a deep model improve next-day hourly passenger forecasts without discarding
the strong weekly seasonal pattern already captured by a four-week median?

## Target and unit of analysis

The observation unit is `(target date, line-station-direction series, service
hour)`. Each sample outputs the 18 regular one-hour bands from 06:00 through
23:00 for one of 546 line-station-direction series. The irregular `before
06:00` and `after 24:00` aggregates are excluded.

## Operational information set

- Demand history: the strictly previous 56 days by 18 service hours.
- Explicit same-hour lags: 1, 7, 14, 21, 28, 35, 42, 49, and 56 days.
- Identity: station, line, and boarding direction as categorical embeddings.
- Time: service-hour embedding, weekday embedding, and cyclic day-of-year.
- Optional forecast weather: eight target-day KMA fields and missingness masks.
- Optional special-day context: holiday, day before, day after, and consecutive
  holiday length.

Station numbers are lookup categories, not ordered numeric measurements.
Addresses are not model inputs. Target-day weather is limited to the latest
forecast issued no later than 05:10 KST; realized target-day observations are
never used by the deployable models.

## Anchor and residual target

The fixed anchor is the median of the same hour from 7, 14, 21, and 28 days
earlier. A scale is estimated separately for each series and service hour from
training dates only. Every V2 neural model starts with exactly zero correction,
so its epoch-zero output is numerically identical to the anchor.

The causal encoder applies a shared six-block TCN independently to each service
hour. Kernel size 2 and dilations `1, 2, 4, 8, 16, 32` give a 64-day receptive
field, covering the complete 56-day input. A shallow convolution over the 18
hour tokens is tested separately. Output corrections are bounded with `tanh`;
the candidate gate is tested by ablation. Special-day correction is isolated
from ordinary-day correction.

## Objective and split

The primary loss is raw-passenger MAE divided by the training target mean.
A smaller Huber term on scaled residuals prevents large stations from
completely dominating optimization. Correction and gate penalties discourage
unnecessary departures from the anchor.

- Parameter-estimation dates: 2025-02-26 through 2025-10-31.
- Ordinary validation: 2025-11-01 through 2025-11-14.
- Special-day stress validation: 2025-08-15 and 2025-10-03 through 2025-10-09,
  removed from parameter estimation.
- Final test: 2025-11-15 through 2025-12-31.

Checkpoint selection uses
`0.85 × ordinary-MAE/ordinary-anchor-MAE + 0.15 ×
event-MAE/event-anchor-MAE`. The stress set is not a pure terminal
rolling-origin split; this selection-bias limitation is explicit.

## Comparison ladder

1. Previous day, previous week, and four-week same-hour median.
2. Ridge and HistGradientBoosting references.
3. V1 Embedding MLP, GRU, TCN, and hierarchical fusion models.
4. V2 causal residual base.
5. Weather-only, special-day-only, and combined V2 inputs.
6. Gate and cross-hour ablations.
7. Three random-seed repetitions of the validation-selected configuration.

## Metrics and interpretation

MAE is primary. RMSE, WAPE, sMAPE, bias, absolute-error quantiles, hourly MAE,
calendar slices, weather-condition slices, and correction diagnostics are
reported. Weather ablations diagnose predictive contribution, not causal
effect. A single-year result is not evidence of deployment stability; a
multi-year outer holdout remains required.
