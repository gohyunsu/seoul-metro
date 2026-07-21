# Research Log

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
