# Project Plan

## Working title

Seoul Metro Demand: reading a year of movement

## Purpose

Build a complete, inspectable data project around station-level daily and time-band ridership. The output must make the reasoning visible: why the question matters, how the data was obtained, what the data can and cannot say, how features were formed, why a model was selected, and how performance was evaluated.

## Evaluation coverage

| Evaluation area | Project evidence | Planned artifact |
| --- | --- | --- |
| Problem definition | A concrete decision question: anticipate station-time demand for operational attention | `docs/METHODOLOGY.md`, site hero and question cards |
| Data collection | Official source, download date, file identity, encoding, schema, coverage, license | `docs/DATASET_CONTEXT.md`, raw file, provenance table |
| Data understanding and visualization | Distribution, calendar pattern, station ranking, line comparison, boarding-vs-alighting, hour-by-day heatmap | `reports/`, site insight sections |
| Analysis and preprocessing | Blank-row removal, type conversion, key validation, long-form transformation, chronological split, feature construction | `scripts/`, `docs/METHODOLOGY.md` |
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

Convert the 20 wide time-band columns into a long table. Add calendar features such as weekday, weekend flag, month, week-of-year, and day-of-month. Add historical lag and rolling features within each station-line-direction-hour series only after sorting by date.

### 5. Explore before modeling

Use questions rather than decoration: which locations dominate total volume, which time bands distinguish weekdays from weekends, where are boarding and alighting asymmetric, and which dates behave unusually. Every figure needs a title, legend where appropriate, axis labels, units, source note, and a short interpretation.

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
