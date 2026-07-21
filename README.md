# seoul-metro

An evidence-led view of how Seoul Metro demand changes across stations, lines, days, and time bands.

The project turns Seoul Open Data Plaza's station-level ridership records into a reproducible analysis and a public-facing narrative. It combines descriptive analysis with a leakage-safe next-day demand forecasting experiment, so the result is useful both as a city-mobility story and as a transparent machine-learning workflow.

## Project questions

1. When and where does demand concentrate across Seoul Metro Lines 1–8?
2. How differently do boarding and alighting patterns behave by station and time band?
3. Can calendar, station, direction, and historical demand features estimate the next day's station demand?
4. Which stations and time bands deserve attention when expected demand is unusually high?

## Repository map

- `data/raw/`: source files downloaded from Seoul Open Data Plaza.
- `docs/`: data context, methodology, project decisions, and research log.
- `scripts/`: reproducible preparation, analysis, and site-data generation scripts.
- `site/`: static website published with GitHub Pages.
- `reports/`: generated figures and report-ready tables.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_analysis.py
python scripts/build_site_data.py
python -m http.server 8000 --directory site
```

Open `http://localhost:8000` after the site-data build completes.

## Data provenance

The primary source is [Seoul Open Data Plaza dataset OA-12921](https://data.seoul.go.kr/dataList/OA-12921/F/1/datasetView.do), provided by Seoul Metro. The current source snapshot is the file `서울교통공사_역별 일별 시간대별 승하차인원_20251231.csv`, downloaded on 2026-07-21. The source is CP949-encoded and is retained unchanged in `data/raw/`.

## License and attribution

Source data is published under Seoul Open Data Plaza's KOGL Type 3 terms shown on the source page. When redistributing derived charts or tables, retain attribution to Seoul Metro and Seoul Open Data Plaza and follow the source license's no-derivatives condition for the original work.
