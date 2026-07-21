# Reference Patterns

The project borrows presentation and experiment patterns from public Kaggle transportation datasets. These are reference patterns, not additional training data.

## Seoul Metro Usage

The Seoul Metro Usage data card is a useful precedent for documenting how inconsistent source formats and encodings are normalized, and for separating station metadata from daily logs. We follow that spirit by preserving the official CP949 source, documenting the reshaping step, and treating station identity as a first-class key.

Source: https://www.kaggle.com/datasets/kimjmin/seoul-metro-usage

## MTA ridership datasets

MTA ridership datasets illustrate the value of combining daily and hourly views and of explaining collection boundaries before making comparisons. We apply the same discipline: descriptive charts use the 20 source time bands, while the model uses an explicitly stated daily target and reports segment-level errors.

Source: https://www.kaggle.com/datasets/princehobby/metropolitan-transportation-authority-mta-datasets

## Seoul bike sharing dataset

The Seoul bike sharing dataset demonstrates a clear time-series framing with calendar and environmental features. We adopt the calendar-feature and forecasting framing, but keep the first version honest about the absence of weather, event, and service-disruption data. Those variables are future extensions rather than silently invented inputs.

Source: https://www.kaggle.com/datasets/lnoahl/seoul-bike-sharing-dataset

## Design principles carried forward

- Write a data card before modeling.
- Separate exploratory patterns from predictive inputs.
- Prefer chronological evaluation for demand data.
- Compare against a seasonal-naive baseline.
- Report errors in both passenger units and scale-aware percentages.
- State coverage boundaries and missing context next to the result.
