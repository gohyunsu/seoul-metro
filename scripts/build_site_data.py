from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / 'data/raw/seoul_metro_ridership_2025.csv'
SITE_DATA_PATH = ROOT / 'site/generated/site_data.json'
REPORT_DIR = ROOT / 'reports'
FIGURE_DIR = REPORT_DIR / 'figures'

# The core model deliberately uses a small, predeclared input contract.  Each
# field has a distinct role and is available by the prediction cutoff: fixed
# series identity (3), target-date calendar (2), and past demand history (5).
# Month/day/day-of-year/weekend were omitted because they duplicate weekday or
# ISO-week information without adding a separately defensible input source.
BASE_FEATURE_COLUMNS = [
    'line', 'station_code', 'direction',
    'weekday', 'week_of_year',
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'rolling_7',
]
BASE_CATEGORICAL_COLUMNS = ['line', 'station_code', 'direction']

# Representative centroids for the ten ranked station names. These were read from
# the Korean national subway-station spatial feature service (2026.04 reference)
# and are used only to position the summary overlay; ridership still comes solely
# from RAW_PATH. Transfer complexes are represented by one point per station name.
STATION_COORDINATES = {
    '잠실(송파구청)': {'lat': 37.514339, 'lng': 127.103366},
    '서울역': {'lat': 37.555341, 'lng': 126.971876},
    '홍대입구': {'lat': 37.557271, 'lng': 126.925318},
    '강남': {'lat': 37.498051, 'lng': 127.027974},
    '사당': {'lat': 37.475909, 'lng': 126.981412},
    '고속터미널': {'lat': 37.504454, 'lng': 127.004575},
    '구로디지털단지': {'lat': 37.485369, 'lng': 126.901376},
    '신림': {'lat': 37.484198, 'lng': 126.929592},
    '종로3가': {'lat': 37.571312, 'lng': 126.991510},
    '삼성(무역센터)': {'lat': 37.508910, 'lng': 127.063123},
}
SPATIAL_SOURCE_URL = 'https://portal.esrikr.com/arcgis/rest/services/Hosted/MOIS_KR_Subway/FeatureServer/1'


def metric_row(name, actual, predicted):
    actual = np.asarray(actual, dtype='float64')
    predicted = np.asarray(predicted, dtype='float64')
    absolute = np.abs(actual - predicted)
    denominator = np.abs(actual) + np.abs(predicted)
    return {
        'name': name,
        'mae': float(mean_absolute_error(actual, predicted)),
        'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
        'wape': float(absolute.sum() / max(np.abs(actual).sum(), 1)),
        'smape': float(np.mean(np.divide(2 * absolute, denominator, out=np.zeros_like(absolute), where=denominator != 0))),
    }


def load_source():
    raw = pd.read_csv(RAW_PATH, encoding='cp949')
    physical_rows = len(raw)
    blank_mask = raw.replace(r'^\s*$', np.nan, regex=True).isna().all(axis=1)
    blank_rows = int(blank_mask.sum())
    source = raw.loc[~blank_mask].copy()
    time_columns = list(source.columns[6:])
    numeric_values = source[time_columns].apply(pd.to_numeric, errors='coerce')
    missing_cells = int(numeric_values.isna().sum().sum())
    negative_cells = int((numeric_values < 0).sum().sum())
    zero_value_cells = int((numeric_values == 0).sum().sum())
    duplicate_rows = int(source.duplicated().sum())
    duplicate_keys = int(source.duplicated(subset=['수송일자', '호선', '역번호', '역명', '승하차구분']).sum())
    source['date'] = pd.to_datetime(source['수송일자'])
    source['line'] = source['호선'].astype('string')
    source['station_code'] = source['역번호'].astype('string').str.replace(r'\.0$', '', regex=True)
    source['station'] = source['역명'].astype('string')
    source['direction'] = source['승하차구분'].astype('string')
    source[time_columns] = numeric_values.fillna(0).astype('int32')
    audit = {
        'physicalRows': physical_rows,
        'blankRowsRemoved': blank_rows,
        'validRows': len(source),
        'sourceColumns': len(time_columns) + 6,
        'timeBandCount': len(time_columns),
        'missingCells': missing_cells,
        'negativeCells': negative_cells,
        'numericCellsChecked': int(numeric_values.size),
        'zeroValueCells': zero_value_cells,
        'duplicateRows': duplicate_rows,
        'duplicateKeys': duplicate_keys,
        'dateStart': source['date'].min().strftime('%Y-%m-%d'),
        'dateEnd': source['date'].max().strftime('%Y-%m-%d'),
    }
    return source, time_columns, audit


def build_model_frame(source, time_columns):
    identifiers = ['date', 'line', 'station_code', 'station', 'direction']
    model = source[identifiers].copy()
    model['passengers'] = source[time_columns].sum(axis=1).astype('float32')
    model = model.sort_values(['line', 'station_code', 'direction', 'date']).reset_index(drop=True)
    group_columns = ['line', 'station_code', 'direction']
    grouped = model.groupby(group_columns, sort=False)['passengers']
    model['lag_1'] = grouped.shift(1)
    model['lag_7'] = grouped.shift(7)
    model['lag_14'] = grouped.shift(14)
    model['lag_28'] = grouped.shift(28)
    model['rolling_7'] = grouped.transform(lambda values: values.shift(1).rolling(7, min_periods=7).mean())
    model['weekday'] = model['date'].dt.dayofweek.astype('int8')
    model['week_of_year'] = model['date'].dt.isocalendar().week.astype('int16')
    return model


INPUT_LABELS = {
    'line': '노선',
    'station_code': '역번호',
    'direction': '승하차 방향',
    'weekday': '요일',
    'week_of_year': 'ISO 주차',
    'lag_1': '하루 전 수요',
    'lag_7': '일주일 전 수요',
    'lag_14': '2주 전 수요',
    'lag_28': '4주 전 수요',
    'rolling_7': '직전 7일 평균',
}
WEEKDAY_LABELS = ['월', '화', '수', '목', '금', '토', '일']


def quantile_profile(values):
    series = pd.Series(values).dropna().astype('float64')
    points = [('P05', .05), ('P25', .25), ('P50', .50), ('P75', .75), ('P95', .95)]
    return {
        'count': int(len(series)),
        'mean': float(series.mean()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        'quantiles': [{'label': label, 'value': float(series.quantile(point))} for label, point in points],
    }


def compact_bin_value(value):
    value = float(value)
    if value >= 1_000_000:
        return f'{value / 1_000_000:.1f}m'
    if value >= 1_000:
        return f'{value / 1_000:.1f}k'
    return f'{value:.0f}'


def shared_log_histogram(series_rows, bins=16, upper_quantile=.995, lower_bound=0.0):
    """Create comparable percentage histograms on shared log-spaced raw-value bins."""
    cleaned = [pd.Series(values).dropna().clip(lower=0).astype('float64') for _, _, values in series_rows]
    combined = pd.concat(cleaned, ignore_index=True)
    cap = max(float(combined.quantile(upper_quantile)), 1.0)
    lower_bound = max(0.0, min(float(lower_bound), cap))
    log_edges = np.linspace(np.log1p(lower_bound), np.log1p(cap), bins + 1)
    raw_edges = np.expm1(log_edges)
    labels = [
        f'{compact_bin_value(raw_edges[index])}–{compact_bin_value(raw_edges[index + 1])}'
        for index in range(bins)
    ]
    labels[-1] = f'≥{compact_bin_value(raw_edges[-2])}'
    rows = []
    for (key, label, _), values in zip(series_rows, cleaned):
        clipped = values.clip(lower=lower_bound, upper=cap)
        counts, _ = np.histogram(np.log1p(clipped), bins=log_edges)
        rows.append({
            'key': key,
            'label': label,
            'counts': [int(value) for value in counts],
            'shares': [float(value / max(len(values), 1)) for value in counts],
            'zeroShare': float((values == 0).mean()),
            'p995': float(values.quantile(upper_quantile)),
            'maximum': float(values.max()),
        })
    return {
        'labels': labels,
        'cap': cap,
        'upperQuantile': upper_quantile,
        'lowerBound': lower_bound,
        'binning': 'raw values clipped to the declared bounds, then log1p-spaced into equal-width bins',
        'rows': rows,
    }


def binned_target_profile(frame, feature, bins=10):
    """Return a post-hoc target profile over quantile bins of one input."""
    values = frame[feature]
    try:
        buckets = pd.qcut(values, q=bins, duplicates='drop')
    except ValueError:
        return []
    profiled = frame.assign(_bucket=buckets).groupby('_bucket', observed=True).agg(
        inputMedian=(feature, 'median'), targetMean=('passengers', 'mean'), rows=('passengers', 'size')
    )
    return [
        {
            'label': f'Q{index:02d}',
            'inputMedian': float(row['inputMedian']),
            'targetMean': float(row['targetMean']),
            'rows': int(row['rows']),
        }
        for index, (_, row) in enumerate(profiled.iterrows(), start=1)
    ]


def split_distribution(frame):
    target = quantile_profile(frame['passengers'])
    return {
        'rows': int(len(frame)),
        'target': target,
        'lag7Median': float(frame['lag_7'].median()),
        'rolling7Median': float(frame['rolling_7'].median()),
    }


def build_input_profile(model_frame, train, validation, test, ridge, ridge_validation_prediction, baseline_test_prediction):
    """Build public, reproducible input diagnostics after the warm-up rule."""
    profile = model_frame[BASE_FEATURE_COLUMNS + ['passengers']].copy()
    category_rows = lambda column: [
        {'label': str(label), 'rows': int(count), 'share': float(count / len(profile))}
        for label, count in profile[column].value_counts().sort_index().items()
    ]
    station_series = (
        model_frame[['line', 'station_code', 'direction']].drop_duplicates()
        .groupby('station_code').size().sort_index()
    )
    multiplicity = station_series.value_counts().sort_index()
    history_columns = ['lag_1', 'lag_7', 'lag_14', 'lag_28', 'rolling_7']
    history_summary = [
        {'key': column, 'label': INPUT_LABELS[column], **quantile_profile(profile[column])}
        for column in history_columns
    ]
    history_correlations = [
        {
            'key': column,
            'label': INPUT_LABELS[column],
            'pearson': float(profile[[column, 'passengers']].corr().iloc[0, 1]),
        }
        for column in history_columns
    ]
    numeric_columns = ['weekday', 'week_of_year', *history_columns, 'passengers']
    numeric_labels = {**INPUT_LABELS, 'passengers': '목표 일일 수요'}
    numeric_distribution = shared_log_histogram([
        (column, numeric_labels[column], profile[column])
        for column in [*history_columns, 'passengers']
    ])
    numeric_correlation = profile[numeric_columns].corr()
    scatter = profile[['lag_7', 'passengers']].sample(n=min(2600, len(profile)), random_state=42)
    scatter_x_cap = float(profile['lag_7'].quantile(.995))
    scatter_y_cap = float(profile['passengers'].quantile(.995))
    station_target = model_frame.groupby('station_code')['passengers'].agg(['size', 'mean', 'median'])
    weekday = (
        profile.groupby('weekday')['passengers'].agg(['size', 'mean', 'median'])
        .reindex(range(7))
    )
    iso_week = profile.groupby('week_of_year')['passengers'].agg(['size', 'mean', 'median']).sort_index()
    rng = np.random.default_rng(42)
    validation_features = validation[BASE_FEATURE_COLUMNS].copy()
    baseline_mae = float(mean_absolute_error(validation['passengers'], ridge_validation_prediction))
    importance = []
    for column in BASE_FEATURE_COLUMNS:
        shuffled = validation_features.copy()
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
        shuffled_mae = float(mean_absolute_error(validation['passengers'], ridge.predict(shuffled)))
        importance.append({
            'key': column,
            'label': INPUT_LABELS[column],
            'maeIncrease': shuffled_mae - baseline_mae,
            'relativeIncrease': (shuffled_mae - baseline_mae) / baseline_mae if baseline_mae else 0.0,
        })
    importance.sort(key=lambda row: row['maeIncrease'], reverse=True)
    seasonal_errors = test.assign(_abs_error=np.abs(test['passengers'].to_numpy() - baseline_test_prediction))
    error_weekday = seasonal_errors.groupby('weekday').agg(rows=('passengers', 'size'), mae=('_abs_error', 'mean'), actualMean=('passengers', 'mean')).reindex(range(7))
    demand_bucket = pd.qcut(seasonal_errors['passengers'], 4, labels=['하위 25%', '25–50%', '50–75%', '상위 25%'], duplicates='drop')
    error_demand = seasonal_errors.assign(_bucket=demand_bucket).groupby('_bucket', observed=True).agg(
        rows=('passengers', 'size'), mae=('_abs_error', 'mean'), actualMean=('passengers', 'mean')
    )
    return {
        'contract': {
            'rawInputCount': len(BASE_FEATURE_COLUMNS),
            'categoricalInputCount': len(BASE_CATEGORICAL_COLUMNS),
            'numericInputCount': len(BASE_FEATURE_COLUMNS) - len(BASE_CATEGORICAL_COLUMNS),
            'featureReadyRows': int(len(profile)),
            'missingAfterWarmup': int(profile[BASE_FEATURE_COLUMNS].isna().sum().sum()),
            'encodedColumns': int(sum(train[column].nunique() for column in BASE_CATEGORICAL_COLUMNS) + 7),
        },
        'categorical': {
            'line': category_rows('line'),
            'direction': category_rows('direction'),
            'stationCode': {
                'levels': int(station_series.size),
                'minSeries': int(station_series.min()),
                'medianSeries': float(station_series.median()),
                'maxSeries': int(station_series.max()),
                'multiplicity': [{'seriesPerCode': int(level), 'stationCodeCount': int(count)} for level, count in multiplicity.items()],
                'rowsPerCode': quantile_profile(station_target['size']),
                'medianTargetAcrossCodes': quantile_profile(station_target['median']),
            },
        },
        'calendar': {
            'weekday': [
                {'label': WEEKDAY_LABELS[index], 'rows': int(row['size']), 'targetMean': float(row['mean']), 'targetMedian': float(row['median'])}
                for index, row in weekday.iterrows()
            ],
            'isoWeek': [
                {'week': int(week), 'rows': int(row['size']), 'targetMean': float(row['mean']), 'targetMedian': float(row['median'])}
                for week, row in iso_week.iterrows()
            ],
        },
        'history': {
            'quantiles': history_summary,
            'correlation': history_correlations,
            'targetByLag7Decile': binned_target_profile(profile, 'lag_7'),
            'distribution': numeric_distribution,
            'lag7Scatter': {
                'sampleMethod': 'deterministic simple random sample, random_state=42',
                'rows': int(len(scatter)),
                'xCap': scatter_x_cap,
                'yCap': scatter_y_cap,
                'points': [
                    {
                        'x': float(min(row['lag_7'], scatter_x_cap)),
                        'y': float(min(row['passengers'], scatter_y_cap)),
                    }
                    for _, row in scatter.iterrows()
                ],
            },
        },
        'numericCorrelation': {
            'keys': numeric_columns,
            'labels': [numeric_labels[column] for column in numeric_columns],
            'values': [
                [float(numeric_correlation.loc[row, column]) for column in numeric_columns]
                for row in numeric_columns
            ],
            'method': 'Pearson correlation over every feature-ready series-day',
        },
        'split': {
            'train': split_distribution(train),
            'validation': split_distribution(validation),
            'test': split_distribution(test),
        },
        'ridgePermutationImportance': {
            'baselineValidationMae': baseline_mae,
            'rows': importance,
            'method': 'validation rows에서 한 입력 열만 무작위로 섞은 뒤 MAE 증가를 측정',
        },
        'seasonalErrorSlices': {
            'model': 'Seasonal naive',
            'weekday': [
                {'label': WEEKDAY_LABELS[index], 'rows': int(row['rows']), 'mae': float(row['mae']), 'actualMean': float(row['actualMean'])}
                for index, row in error_weekday.iterrows()
            ],
            'demand': [
                {'label': str(label), 'rows': int(row['rows']), 'mae': float(row['mae']), 'actualMean': float(row['actualMean'])}
                for label, row in error_demand.iterrows()
            ],
        },
    }


def build_summary(source, time_columns):
    band_totals = source[time_columns].sum().sort_index()
    daily = source.groupby('date')[time_columns].sum().sum(axis=1).sort_index()
    daily_band = source.groupby('date')[time_columns].sum().sort_index()
    line_totals = source.groupby('line')[time_columns].sum().sum(axis=1).sort_values(ascending=False)
    station_totals = source.groupby('station')[time_columns].sum().sum(axis=1).sort_values(ascending=False)
    direction_totals = source.groupby('direction')[time_columns].sum().sum(axis=1)
    typical_by_weekday = daily_band.copy()
    typical_by_weekday['weekday'] = typical_by_weekday.index.dayofweek
    heatmap = typical_by_weekday.groupby('weekday')[time_columns].mean().reindex(range(7))
    top_line = line_totals.index[0]
    top_station = station_totals.index[0]
    peak_time = band_totals.idxmax()
    daily_frame = pd.DataFrame({'total': daily})
    daily_frame['weekday'] = daily_frame.index.dayofweek
    weekday_mean = daily_frame.groupby('weekday')['total'].mean()
    weekday_labels = ['월', '화', '수', '목', '금', '토', '일']
    weekday_table = [
        {'label': weekday_labels[index], 'mean': int(value), 'isWeekend': index >= 5}
        for index, value in weekday_mean.reindex(range(7)).items()
    ]
    band_table = [
        {'label': label, 'total': int(value), 'share': float(value / band_totals.sum()), 'dailyMean': int(value / len(daily))}
        for label, value in band_totals.items()
    ]
    line_station_counts = source.groupby('line')['station'].nunique()
    line_table = [
        {'label': label, 'total': int(value), 'share': float(value / line_totals.sum()), 'dailyMean': int(value / len(daily)), 'stationCount': int(line_station_counts.get(label, 0))}
        for label, value in line_totals.items()
    ]
    direction_table = [
        {'label': label, 'total': int(value), 'share': float(value / direction_totals.sum())}
        for label, value in direction_totals.items()
    ]
    high_dates = daily.sort_values(ascending=False).head(5)
    low_dates = daily.sort_values().head(5)
    date_records = lambda values: [
        {'date': date.strftime('%m.%d'), 'weekday': weekday_labels[date.dayofweek], 'total': int(value)}
        for date, value in values.items()
    ]
    top10_share = float(station_totals.head(10).sum() / station_totals.sum())
    monthly = daily.groupby(daily.index.month).agg(['size', 'mean', 'median', 'min', 'max'])
    monthly_quantiles = daily.groupby(daily.index.month).quantile([.1, .25, .75, .9]).unstack()
    daily_change = daily.pct_change().dropna() * 100
    change_cap = max(float(daily_change.abs().quantile(.99)), 1.0)
    change_edges = np.linspace(-change_cap, change_cap, 19)
    change_counts, _ = np.histogram(daily_change.clip(-change_cap, change_cap), bins=change_edges)
    change_labels = [
        f'{change_edges[index]:+.0f}–{change_edges[index + 1]:+.0f}%'
        for index in range(len(change_edges) - 1)
    ]
    line_time = source.groupby('line')[time_columns].sum().sort_index() / len(daily)
    line_direction = source.groupby(['line', 'direction'])[time_columns].sum().sum(axis=1).unstack(fill_value=0).sort_index()
    cumulative_station_share = station_totals.cumsum() / station_totals.sum()
    station_distribution = shared_log_histogram([
        ('annual_station_total', '역별 연간 승하차량', station_totals)
    ], bins=18, upper_quantile=1.0, lower_bound=float(station_totals.min()))
    monthly_labels = [f'{int(month)}월' for month in monthly.index]
    return {
        'totalPassengers': int(source[time_columns].to_numpy().sum()),
        'stationCount': int(source['station'].nunique()),
        'lineCount': int(source['line'].nunique()),
        'validRows': int(len(source)),
        'dateCount': int(source['date'].nunique()),
        'peakTime': peak_time,
        'peakTimeDetail': f'{int(band_totals.max()):,}명 누적',
        'topStation': top_station,
        'topStationDetail': f'{int(station_totals.iloc[0]):,}명 · {top_line} 포함',
        'dailyInsight': f'일평균 {int(daily.mean()):,}명, 최대 {daily.idxmax():%m월 %d일}에 {int(daily.max()):,}명이 기록됐습니다.',
        'lineInsight': f'{top_line}이 전체의 {line_totals.iloc[0] / line_totals.sum() * 100:.1f}%로 가장 큰 비중을 차지합니다.',
        'directionInsight': f'승차 {int(direction_totals.get("승차", 0)):,}명, 하차 {int(direction_totals.get("하차", 0)):,}명으로 집계됩니다.',
        'placeInsight': f'{top_station}이 연간 {int(station_totals.iloc[0]):,}명으로 가장 많습니다. 상위 10개 역이 전체의 {station_totals.head(10).sum() / station_totals.sum() * 100:.1f}%를 차지합니다.',
        'forecastInsight': '최근 7일 리듬을 기준선으로 삼아, 달력과 역의 조합이 만드는 추가 신호를 비교합니다.',
        'splitText': '80 / 10 / 10',
        'weekdayMean': int(daily_frame[daily_frame['weekday'] < 5]['total'].mean()),
        'weekendMean': int(daily_frame[daily_frame['weekday'] >= 5]['total'].mean()),
        'weekdayLift': f'{(daily_frame[daily_frame["weekday"] < 5]["total"].mean() / daily_frame[daily_frame["weekday"] >= 5]["total"].mean() - 1) * 100:.1f}%',
        'dailyMedian': int(daily.median()),
        'dailyStd': int(daily.std()),
        'top10Share': f'{top10_share * 100:.1f}%',
        'edaLead': f'평일 평균은 주말보다 {(daily_frame[daily_frame["weekday"] < 5]["total"].mean() / daily_frame[daily_frame["weekday"] >= 5]["total"].mean() - 1) * 100:.1f}% 높고, 18–19시가 전체 시간대 중 가장 큽니다.',
    }, daily, band_totals, line_totals, station_totals, heatmap, {
        'weekday': weekday_table,
        'bands': band_table,
        'lines': line_table,
        'directions': direction_table,
        'highDates': date_records(high_dates),
        'lowDates': date_records(low_dates),
        'daily': {'mean': int(daily.mean()), 'median': int(daily.median()), 'std': int(daily.std()), 'min': int(daily.min()), 'max': int(daily.max()), 'minDate': daily.idxmin().strftime('%m.%d'), 'maxDate': daily.idxmax().strftime('%m.%d')},
        'monthlyDistribution': {
            'labels': monthly_labels,
            'rows': [
                {
                    'label': f'{int(month)}월',
                    'days': int(row['size']),
                    'mean': float(row['mean']),
                    'median': float(row['median']),
                    'min': float(row['min']),
                    'p10': float(monthly_quantiles.loc[month, .1]),
                    'p25': float(monthly_quantiles.loc[month, .25]),
                    'p75': float(monthly_quantiles.loc[month, .75]),
                    'p90': float(monthly_quantiles.loc[month, .9]),
                    'max': float(row['max']),
                }
                for month, row in monthly.iterrows()
            ],
        },
        'dailyChange': {
            'labels': change_labels,
            'counts': [int(value) for value in change_counts],
            'cap': change_cap,
            'median': float(daily_change.median()),
            'p10': float(daily_change.quantile(.1)),
            'p90': float(daily_change.quantile(.9)),
            'largestRise': {'date': daily_change.idxmax().strftime('%m.%d'), 'value': float(daily_change.max())},
            'largestDrop': {'date': daily_change.idxmin().strftime('%m.%d'), 'value': float(daily_change.min())},
        },
        'lineTimeHeatmap': {
            'rows': [str(value) for value in line_time.index],
            'columns': time_columns,
            'values': [[int(value) for value in line_time.loc[line].values] for line in line_time.index],
            'min': int(line_time.to_numpy().min()),
            'max': int(line_time.to_numpy().max()),
            'unit': 'mean passengers per calendar day',
        },
        'lineDirectionBalance': [
            {
                'label': str(line),
                'boarding': int(row.get('승차', 0)),
                'alighting': int(row.get('하차', 0)),
                'boardingShare': float(row.get('승차', 0) / max(row.sum(), 1)),
                'balancePct': float((row.get('승차', 0) - row.get('하차', 0)) / max(row.sum(), 1) * 100),
            }
            for line, row in line_direction.iterrows()
        ],
        'stationConcentration': {
            'stationCount': int(len(station_totals)),
            'points': [
                {'rank': int(rank), 'share': float(share)}
                for rank, share in enumerate(cumulative_station_share.values, start=1)
            ],
            'top1Share': float(cumulative_station_share.iloc[0]),
            'top10Share': float(cumulative_station_share.iloc[min(9, len(cumulative_station_share) - 1)]),
            'top25Share': float(cumulative_station_share.iloc[min(24, len(cumulative_station_share) - 1)]),
            'top50Share': float(cumulative_station_share.iloc[min(49, len(cumulative_station_share) - 1)]),
            'distribution': station_distribution,
        },
    }


def make_figures(daily, band_totals, line_totals, station_totals, heatmap, time_columns):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 10, 'figure.dpi': 130})
    figure, axis = plt.subplots(figsize=(11, 4.4))
    axis.plot(daily.index, daily.values, color='#1c5b43', linewidth=1.3)
    axis.fill_between(daily.index, daily.values, color='#b9e3cc', alpha=.45)
    axis.set(title='Seoul Metro Daily Ridership, 2025', xlabel='Date', ylabel='Passengers')
    axis.grid(axis='y', color='#d8e1db', linewidth=.7)
    figure.tight_layout(); figure.savefig(FIGURE_DIR / 'daily_network_demand.png', bbox_inches='tight'); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 5))
    matrix = heatmap.to_numpy() / 1_000_000
    image = axis.imshow(matrix, aspect='auto', cmap='YlGn')
    axis.set(title='Typical Ridership by Weekday and Time Band', xlabel='Time band', ylabel='Weekday')
    axis.set_xticks(range(len(time_columns))); axis.set_xticklabels(time_columns, rotation=75, ha='right', fontsize=7)
    axis.set_yticks(range(7)); axis.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    figure.colorbar(image, ax=axis, label='Million passengers')
    figure.tight_layout(); figure.savefig(FIGURE_DIR / 'weekday_time_heatmap.png', bbox_inches='tight'); plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    top = station_totals.head(15).sort_values()
    axis.barh(top.index, top.values / 1_000_000, color='#1c5b43')
    axis.set(title='Top Stations by Annual Ridership', xlabel='Million passengers', ylabel='Station')
    axis.grid(axis='x', color='#d8e1db', linewidth=.7)
    figure.tight_layout(); figure.savefig(FIGURE_DIR / 'station_ranking.png', bbox_inches='tight'); plt.close(figure)


def run_models(model_frame):
    feature_columns = BASE_FEATURE_COLUMNS
    categorical = BASE_CATEGORICAL_COLUMNS
    numeric = [column for column in feature_columns if column not in categorical]
    source_model_rows = len(model_frame)
    series_count = int(model_frame[['line', 'station_code', 'direction']].drop_duplicates().shape[0])
    model_frame = model_frame.dropna(subset=['lag_1', 'lag_7', 'lag_14', 'lag_28', 'rolling_7']).copy()
    history_warmup_rows = source_model_rows - len(model_frame)
    dates = sorted(model_frame['date'].unique())
    train_end = dates[int(len(dates) * .8) - 1]
    validation_end = dates[int(len(dates) * .9) - 1]
    train = model_frame[model_frame['date'] <= train_end].copy()
    validation = model_frame[(model_frame['date'] > train_end) & (model_frame['date'] <= validation_end)].copy()
    test = model_frame[model_frame['date'] > validation_end].copy().reset_index(drop=True)
    sample = train.sample(n=min(250_000, len(train)), random_state=42)
    categorical_levels = {column: int(train[column].nunique()) for column in categorical}
    results = []
    baseline_validation_prediction = validation['lag_7'].to_numpy()
    baseline_test_prediction = test['lag_7'].to_numpy()
    baseline_validation = metric_row('Seasonal naive', validation['passengers'], baseline_validation_prediction)
    baseline_test = metric_row('Seasonal naive', test['passengers'], baseline_test_prediction)
    baseline_validation['split'] = 'validation'; baseline_test['split'] = 'test'
    results.extend([baseline_validation, baseline_test])

    ridge_preprocessor = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('numeric', StandardScaler(), numeric),
    ])
    ridge = Pipeline([('preprocess', ridge_preprocessor), ('model', Ridge(alpha=10.0))])
    ridge.fit(sample[feature_columns], sample['passengers'])
    ridge_validation_prediction = ridge.predict(validation[feature_columns])
    ridge_test_prediction = ridge.predict(test[feature_columns])
    ridge_validation = metric_row('Ridge', validation['passengers'], ridge_validation_prediction)
    ridge_test = metric_row('Ridge', test['passengers'], ridge_test_prediction)
    ridge_validation['split'] = 'validation'; ridge_test['split'] = 'test'
    results.extend([ridge_validation, ridge_test])

    tree_preprocessor = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
        ('numeric', 'passthrough', numeric),
    ])
    tree = Pipeline([('preprocess', tree_preprocessor), ('model', HistGradientBoostingRegressor(max_iter=120, learning_rate=.08, max_leaf_nodes=31, l2_regularization=10, random_state=42))])
    tree.fit(sample[feature_columns], sample['passengers'])
    tree_validation_prediction = tree.predict(validation[feature_columns])
    tree_test_prediction = tree.predict(test[feature_columns])
    tree_validation = metric_row('HistGradientBoosting', validation['passengers'], tree_validation_prediction)
    tree_test = metric_row('HistGradientBoosting', test['passengers'], tree_test_prediction)
    tree_validation['split'] = 'validation'; tree_test['split'] = 'test'
    results.extend([tree_validation, tree_test])

    validation_results = [row for row in results if row['split'] == 'validation']
    best_name = min(validation_results, key=lambda row: row['mae'])['name']
    public_models = []
    for name in ['Seasonal naive', 'Ridge', 'HistGradientBoosting']:
        test_row = next(item for item in results if item['name'] == name and item['split'] == 'test').copy()
        validation_row = next(item for item in results if item['name'] == name and item['split'] == 'validation')
        test_row['best'] = name == best_name
        test_row['validation'] = {
            key: validation_row[key]
            for key in ['mae', 'rmse', 'wape', 'smape']
        }
        public_models.append(test_row)
    input_profile = build_input_profile(
        model_frame, train, validation, test, ridge,
        ridge_validation_prediction, baseline_test_prediction,
    )
    pd.DataFrame(results).to_csv(REPORT_DIR / 'model_metrics.csv', index=False)
    preferred_example = test[(test['line'] == '2호선') & (test['station'] == '강남') & (test['direction'] == '승차')]
    example = (preferred_example if not preferred_example.empty else test).sort_values('date').iloc[0]
    example_position = int(example.name)
    example_actual = int(example['passengers'])
    test_example = {
        'date': example['date'].strftime('%Y-%m-%d'),
        'series': {
            'line': str(example['line']),
            'station': str(example['station']),
            'stationCode': str(example['station_code']),
            'direction': str(example['direction']),
        },
        'calendar': {
            key: int(example[key])
            for key in ['weekday', 'week_of_year']
        },
        'history': {
            key: int(round(example[key]))
            for key in ['lag_1', 'lag_7', 'lag_14', 'lag_28', 'rolling_7']
        },
        'actual': example_actual,
        'predictions': {
            'Seasonal naive': int(round(baseline_test_prediction[example_position])),
            'Ridge': int(round(ridge_test_prediction[example_position])),
            'HistGradientBoosting': int(round(tree_test_prediction[example_position])),
        },
        'absoluteErrors': {
            'Seasonal naive': int(round(abs(example_actual - baseline_test_prediction[example_position]))),
            'Ridge': int(round(abs(example_actual - ridge_test_prediction[example_position]))),
            'HistGradientBoosting': int(round(abs(example_actual - tree_test_prediction[example_position]))),
        },
    }
    audit = {
        'trainStart': str(train['date'].min())[:10],
        'trainEnd': str(train_end)[:10],
        'validationStart': str(validation['date'].min())[:10],
        'validationEnd': str(validation_end)[:10],
        'testStart': str(test['date'].min())[:10],
        'testEnd': str(test['date'].max())[:10],
        'trainRows': len(train),
        'validationRows': len(validation),
        'testRows': len(test),
        'trainingSampleRows': len(sample),
        'modelSourceRows': source_model_rows,
        'featureReadyRows': len(model_frame),
        'historyWarmupRows': history_warmup_rows,
        'seriesCount': series_count,
        'modelDateCount': len(dates),
        'featureCount': len(feature_columns),
        'categoricalFeatureCount': len(categorical),
        'numericFeatureCount': len(numeric),
        'categoricalLevels': categorical_levels,
        'lineLevels': categorical_levels['line'],
        'stationLevels': categorical_levels['station_code'],
        'directionLevels': categorical_levels['direction'],
        'ridgeDesignColumns': sum(categorical_levels.values()) + len(numeric),
        'treeInputColumns': sum(categorical_levels.values()) + len(numeric),
        'calendarFeatureCount': 2,
        'historyFeatureCount': 5,
        'predictionRows': len(test),
        'selectedByValidation': best_name,
        'testExample': test_example,
    }
    return public_models, audit, input_profile


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    source, time_columns, source_audit = load_source()
    model_frame = build_model_frame(source, time_columns)
    summary, daily, band_totals, line_totals, station_totals, heatmap, eda = build_summary(source, time_columns)
    make_figures(daily, band_totals, line_totals, station_totals, heatmap, time_columns)
    models, model_audit, input_profile = run_models(model_frame)
    summary['selectedModel'] = model_audit['selectedByValidation']
    summary['testWindow'] = f"{model_audit['testStart']}–{model_audit['testEnd']}"
    selected_test = next(model for model in models if model['best'])
    summary['selectedTestMae'] = int(round(selected_test['mae']))
    summary['selectedTestWape'] = f"{selected_test['wape'] * 100:.1f}%"
    direction = source.groupby('direction')[time_columns].sum().reindex(['승차', '하차']).fillna(0)
    top_stations = station_totals.head(10)
    spatial_stations = [
        {
            'name': str(name),
            'value': int(value),
            'rank': rank,
            **STATION_COORDINATES[str(name)],
        }
        for rank, (name, value) in enumerate(top_stations.items(), start=1)
        if str(name) in STATION_COORDINATES
    ]
    site_data = {
        'summary': summary,
        'daily': {'labels': [date.strftime('%m-%d') for date in daily.index], 'values': [int(value) for value in daily.values]},
        'lines': {'labels': [str(value) for value in line_totals.index], 'values': [int(value) for value in line_totals.values]},
        'direction': {'labels': time_columns, 'boarding': [int(value) for value in direction.loc['승차'].values], 'alighting': [int(value) for value in direction.loc['하차'].values]},
        'heatmap': {'labels': time_columns, 'values': [[int(value) for value in heatmap[time_column].values] for time_column in time_columns], 'min': int(heatmap.to_numpy().min()), 'max': int(heatmap.to_numpy().max())},
        'stations': {'values': [{'name': str(name), 'value': int(value)} for name, value in top_stations.items()]},
        'spatial': {
            'stations': spatial_stations,
            'scope': '2025 annual ridership · top 10 ranked station names only',
            'coordinateMethod': 'representative station-complex centroid',
            'sourceUrl': SPATIAL_SOURCE_URL,
            'rankedStationCount': int(len(top_stations)),
            'coordinatesMatched': int(len(spatial_stations)),
            'coordinatesMissing': int(len(top_stations) - len(spatial_stations)),
        },
        'models': models,
        'inputProfile': input_profile,
        'eda': eda,
        'audit': {
            **source_audit,
            'stationCount': int(source['station'].nunique()),
            'lineCount': int(source['line'].nunique()),
            'model': model_audit,
        },
    }
    SITE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SITE_DATA_PATH.write_text(json.dumps(site_data, ensure_ascii=False, indent=2), encoding='utf-8')
    (REPORT_DIR / 'data_audit.json').write_text(json.dumps(site_data['audit'], ensure_ascii=False, indent=2), encoding='utf-8')
    (REPORT_DIR / 'input_profile.json').write_text(
        json.dumps(input_profile, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps({'site_data': str(SITE_DATA_PATH), 'summary': summary, 'models': models, 'model_audit': model_audit}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
