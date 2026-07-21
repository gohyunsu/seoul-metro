#!/usr/bin/env python3
"""Refresh the weather-only ablation without rebuilding unchanged EDA assets.

The full site builder remains the canonical end-to-end build. This focused
entry point is useful while iterating on the CPU-bound weather experiments:
it reuses the fixed baseline metrics already published in site_data.json and
rewrites only the weather analysis and its report.
"""

import json

from build_site_data import (
    REPORT_DIR,
    SITE_DATA_PATH,
    build_model_frame,
    load_source,
    load_weather,
    run_weather_ablation,
)


def main():
    site_data = json.loads(SITE_DATA_PATH.read_text(encoding='utf-8'))
    source, time_columns, _ = load_source()
    model_frame = build_model_frame(source, time_columns)
    weather_frame, weather_audit = load_weather()
    weather_model_frame = model_frame.merge(weather_frame, on='date', how='left', validate='many_to_one')
    weather_analysis = run_weather_ablation(weather_model_frame, site_data['models'])

    site_data['weatherAnalysis'] = {'weather': weather_audit, **weather_analysis}
    site_data['audit']['externalInputs'] = {'weatherRows': weather_audit['rowCount']}
    SITE_DATA_PATH.write_text(json.dumps(site_data, ensure_ascii=False, indent=2), encoding='utf-8')
    (REPORT_DIR / 'weather_ablation.json').write_text(
        json.dumps(site_data['weatherAnalysis'], ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps({
        'experiments': [
            {'id': row['id'], 'validationMae': row['validation']['mae'], 'testMae': row['test']['mae']}
            for row in weather_analysis['experiments']
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
