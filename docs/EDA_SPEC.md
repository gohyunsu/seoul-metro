# EDA Specification

## Purpose

The EDA must answer concrete questions before any model result is shown. The site renders these checks from `site/generated/site_data.json`; the source of truth is `scripts/build_site_data.py`.

## Required checks

| Question | Calculation | Interpretation |
| --- | --- | --- |
| Is the source structurally usable? | Physical rows, blank rows, source columns, missing cells, negative cells, duplicate rows, duplicate logical keys | Explain every removal and confirm whether valid zero counts were preserved |
| How large is a typical day? | Mean, median, standard deviation, minimum, maximum, high/low dates | Distinguish typical demand from exceptional dates |
| Does the calendar matter? | Mean daily total for Monday–Sunday, weekday-to-weekend lift | Establish whether a seasonal baseline is plausible |
| Which hours carry the network? | Total and daily mean for all 20 time bands, share of annual volume | Identify peak periods without hiding the late-night tail |
| Which lines carry scale? | Annual total, share, daily mean, station count by line | Compare network contribution with line size |
| Is demand concentrated? | Top station ranking and top-10 share of network volume | Avoid treating a large-station result as universal |
| Do boarding and alighting differ? | Annual totals and time-band profiles by direction | Interpret station role and directional asymmetry |
| Which dates deserve review? | Top five and bottom five daily totals with weekday labels | Separate ordinary seasonality from unusual observations |

## Visual contract

Every chart or table must state its unit, period, aggregation, and interpretation. The public page includes:

1. Daily trend and distribution statistics.
2. Weekday × time-band heatmap.
3. Time-band ranking table.
4. Line contribution table.
5. Station concentration ranking.
6. Boarding versus alighting profile.
7. High/low date watchlist.
8. Data-quality audit card.

## Modeling handoff

EDA is complete only when the task can be stated without ambiguity: predict the next calendar day's `daily_total` for each station × line × direction series. The baseline is the same series' value seven days earlier. EDA findings must be used to justify the baseline and to explain why errors may differ by scale, line, direction, and weekday/weekend segment.
