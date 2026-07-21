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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / 'data/raw/seoul_metro_ridership_2025.csv'
SITE_DATA_PATH = ROOT / 'site/generated/site_data.json'
REPORT_DIR = ROOT / 'reports'
FIGURE_DIR = REPORT_DIR / 'figures'

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
    source['station_code'] = source['역번호'].astype('string')
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
    model['month'] = model['date'].dt.month.astype('int8')
    model['week_of_year'] = model['date'].dt.isocalendar().week.astype('int16')
    model['day_of_month'] = model['date'].dt.day.astype('int8')
    model['day_of_year'] = model['date'].dt.dayofyear.astype('int16')
    model['is_weekend'] = (model['weekday'] >= 5).astype('int8')
    return model


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
    feature_columns = ['line', 'station', 'direction', 'weekday', 'month', 'week_of_year', 'day_of_month', 'day_of_year', 'is_weekend', 'lag_1', 'lag_7', 'lag_14', 'lag_28', 'rolling_7']
    categorical = ['line', 'station', 'direction']
    numeric = [column for column in feature_columns if column not in categorical]
    source_model_rows = len(model_frame)
    series_count = int(model_frame[['line', 'station_code', 'direction']].drop_duplicates().shape[0])
    model_frame = model_frame.dropna(subset=['lag_1', 'lag_7', 'lag_14', 'lag_28', 'rolling_7']).copy()
    history_warmup_rows = source_model_rows - len(model_frame)
    dates = sorted(model_frame['date'].unique())
    train_end = dates[int(len(dates) * .8) - 1]
    validation_end = dates[int(len(dates) * .9) - 1]
    train = model_frame[model_frame['date'] <= train_end]
    validation = model_frame[(model_frame['date'] > train_end) & (model_frame['date'] <= validation_end)]
    test = model_frame[model_frame['date'] > validation_end]
    sample = train.sample(n=min(250_000, len(train)), random_state=42)
    results = []
    baseline_validation = metric_row('Seasonal naive', validation['passengers'], validation['lag_7'])
    baseline_test = metric_row('Seasonal naive', test['passengers'], test['lag_7'])
    baseline_validation['split'] = 'validation'; baseline_test['split'] = 'test'
    results.extend([baseline_validation, baseline_test])

    ridge_preprocessor = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('numeric', StandardScaler(), numeric),
    ])
    ridge = Pipeline([('preprocess', ridge_preprocessor), ('model', Ridge(alpha=10.0))])
    ridge.fit(sample[feature_columns], sample['passengers'])
    ridge_validation = metric_row('Ridge', validation['passengers'], ridge.predict(validation[feature_columns]))
    ridge_test = metric_row('Ridge', test['passengers'], ridge.predict(test[feature_columns]))
    ridge_validation['split'] = 'validation'; ridge_test['split'] = 'test'
    results.extend([ridge_validation, ridge_test])

    tree_preprocessor = ColumnTransformer([
        ('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical),
        ('numeric', 'passthrough', numeric),
    ])
    tree = Pipeline([('preprocess', tree_preprocessor), ('model', HistGradientBoostingRegressor(max_iter=120, learning_rate=.08, max_leaf_nodes=31, l2_regularization=10, random_state=42))])
    tree.fit(sample[feature_columns], sample['passengers'])
    tree_validation = metric_row('HistGradientBoosting', validation['passengers'], tree.predict(validation[feature_columns]))
    tree_test = metric_row('HistGradientBoosting', test['passengers'], tree.predict(test[feature_columns]))
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
    pd.DataFrame(results).to_csv(REPORT_DIR / 'model_metrics.csv', index=False)
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
        'calendarFeatureCount': 6,
        'historyFeatureCount': 5,
        'predictionRows': len(test),
        'selectedByValidation': best_name,
    }
    return public_models, audit


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    source, time_columns, source_audit = load_source()
    model_frame = build_model_frame(source, time_columns)
    summary, daily, band_totals, line_totals, station_totals, heatmap, eda = build_summary(source, time_columns)
    make_figures(daily, band_totals, line_totals, station_totals, heatmap, time_columns)
    models, model_audit = run_models(model_frame)
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
        },
        'models': models,
        'eda': eda,
        'audit': {**source_audit, 'stationCount': int(source['station'].nunique()), 'lineCount': int(source['line'].nunique()), 'model': model_audit},
    }
    SITE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SITE_DATA_PATH.write_text(json.dumps(site_data, ensure_ascii=False, indent=2), encoding='utf-8')
    (REPORT_DIR / 'data_audit.json').write_text(json.dumps(site_data['audit'], ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'site_data': str(SITE_DATA_PATH), 'summary': summary, 'models': models, 'model_audit': model_audit}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
