# Dataset Context

## Source snapshot

| Field | Value |
| --- | --- |
| Provider | Seoul Metro |
| Portal | Seoul Open Data Plaza |
| Dataset ID | OA-12921 |
| Source URL | https://data.seoul.go.kr/dataList/OA-12921/F/1/datasetView.do |
| File | 서울교통공사_역별 일별 시간대별 승하차인원_20251231.csv |
| Download date | 2026-07-21 |
| Encoding | CP949 |
| Coverage | 2025-01-01 through 2025-12-31 |
| Scope | Seoul Metro-operated sections of Lines 1–8 |
| Update cycle | Annual |
| Source license | KOGL Type 3: attribution and no derivatives, as displayed on source page |

## Grain

The source is wide. One valid row represents a date, line, station, and boarding type (`승차` or `하차`), followed by 20 passenger-count fields from before 06:00 through after 24:00. The data is therefore naturally reshaped into one observation per date × line × station × boarding type × time band.

## Observed snapshot audit

The downloaded file contains 199,424 physical rows and 26 source columns. After excluding 134 completely blank trailing rows, 199,290 valid rows remain. The valid snapshot covers 365 dates, 242 station names, eight Seoul Metro lines, and 20 time bands. There are no negative passenger counts and no duplicate logical keys in the nonblank data.

The source includes line-specific operating boundaries. It does not represent every metropolitan railway service: for example, the page identifies the Seoul Metro-managed portions of Lines 1, 4, 7, and 8. Conclusions must therefore be phrased as conclusions about this provider's coverage, not about every rail service in the Seoul metropolitan area.

## Field dictionary

| Field | Meaning | Treatment |
| --- | --- | --- |
| 연번 | Source row number | Retain for audit, not a model feature |
| 수송일자 | Service date | Parse as date and derive calendar features |
| 호선 | Line | Categorical feature |
| 역번호 | Station identifier | Categorical key; preserve as string |
| 역명 | Station name | Display label and categorical feature |
| 승하차구분 | Boarding or alighting | Categorical feature |
| 06시이전 … 24시이후 | Passenger count in each time band | Numeric target/value; reshape to `time_band` and `passengers` |

## Interpretation cautions

- A count is a recorded boarding or alighting volume, not a unique-person count.
- A high station total may reflect transfer behavior, surrounding land use, or the provider's station boundary; the dataset alone cannot identify causality.
- The 2025-only snapshot supports within-year seasonality and short-horizon forecasting, but not robust long-run trend claims.
- Public holidays, events, service disruptions, school calendars, and station changes are not included in the source file. Their absence belongs in the limitations section.
