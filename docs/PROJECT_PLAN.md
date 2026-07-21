# Project Plan

## Working title

Seoul Metro Demand: reading a year of movement

## Purpose

Build a complete, inspectable data project around station-level daily and time-band ridership. The output must make the reasoning visible: why the question matters, how the data was obtained, what the data can and cannot say, how features were formed, why a model was selected, and how performance was evaluated.

## Confirmed task

Estimate the next calendar day's total boardings or alightings for every station × line × direction series in the Seoul Metro-operated 1–8 line coverage. The prediction target is the sum of the 20 time-band counts in a source row. The result is intended to support a daily watchlist: which station-direction series may require attention tomorrow?

The task is deliberately narrower than claiming to forecast the entire city's future mobility. It has a fixed unit, a directly observed target, and a time-aware test period. The first success criterion is to beat the seven-day seasonal-naive baseline on validation MAE by at least 5%. If no learned model clears that bar, the baseline remains the selected reference and the result is reported as a meaningful finding.

## Evaluation coverage

| Evaluation area | Project evidence | Planned artifact |
| --- | --- | --- |
| Problem definition | Next-calendar-day station × line × direction demand watchlist | `docs/METHODOLOGY.md`, site task lock |
| Data collection | Official source, download date, file identity, encoding, schema, coverage, license | `docs/DATASET_CONTEXT.md`, raw file, provenance table |
| Data understanding and visualization | Quality audit, daily distribution, weekday/weekend comparison, time-band ranking, line share, station concentration, boarding-vs-alighting, anomaly dates, hour-by-day heatmap | `reports/`, site EDA lab |
| Analysis and preprocessing | Blank-row removal, type conversion, key validation, daily target construction, chronological split, lag and rolling features | `scripts/`, `docs/METHODOLOGY.md` |
| Algorithms | Transparent baseline plus regularized and tree-based regressors | model comparison table and experiment log |
| Evaluation | Time-aware holdout, MAE, RMSE, WAPE, sMAPE, residual review, segment-level performance | `reports/model_metrics.csv`, site model section |
| Future improvement | External calendar, weather, events, service disruptions, longer historical window, probabilistic forecasts | limitations and next-steps section |

## Research workflow

### 1. Frame the decision

Define the prediction unit as one station × line × boarding type × date. The target is the next observed date's total passenger count across the 20 time bands. Time bands remain available for descriptive insights while the daily target stays operationally interpretable.

### 2. Establish the data contract

Record source URL, file name, download date, file size, checksum, encoding, column meanings, date range, station coverage, and known ownership boundaries. Keep the raw source immutable; all derived data must be rebuildable from it.

### 3. Audit before interpretation

Measure row counts, blank rows, missing cells, duplicate rows, duplicate logical keys, negative values, date continuity, line counts, station counts, and boarding-type balance. Every removal or transformation must be reported with before-and-after counts.

### 4. Create the analysis table

Keep the source grain for the forecasting table and add `daily_total = sum(20 time bands)`. Add calendar features such as weekday, weekend flag, month, week-of-year, and day-of-month. Add historical lag and rolling features within each station-line-direction series only after sorting by date. Use the 20 time bands in a separate long-form view for EDA figures and interpretation.

### 5. Explore before modeling

Use questions rather than decoration: which locations dominate total volume, which time bands distinguish weekdays from weekends, where are boarding and alighting asymmetric, which dates behave unusually, and how concentrated is the network in its largest stations. Every figure needs a title, legend where appropriate, axis labels, units, source note, and a short interpretation. The EDA checklist is maintained in `docs/EDA_SPEC.md`.

### 6. Compare models honestly

Use a chronological holdout rather than a random split. Compare a seasonal-naive baseline with a linear model and a tree-based model. Fit preprocessing on the training period only. Report the same metrics for every model and segment the final errors by line, boarding type, weekday/weekend, and demand scale.

### 7. Communicate the result

The public site follows the order: signal → evidence → method → model → limits. It should be readable without a notebook, while linking detailed methodology and provenance for readers who want to audit the work.

## Definition of done

- Raw source and checksum are recorded.
- All seven required reasoning areas have a corresponding document section.
- At least five publication-quality figures include interpretations.
- The model split is chronological and leakage-safe.
- Baseline and at least two model families are compared.
- Metrics include an error measure in passenger units and a scale-aware percentage measure.
- The website works on a static host without a backend.
- Every public claim is traceable to a generated table, figure, or source note.
- The confirmed task, target, split, success criterion, and fallback decision are visible on the public site.
