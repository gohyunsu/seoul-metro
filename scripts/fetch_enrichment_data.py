#!/usr/bin/env python3
"""Fetch the public weather input used by the weather ablation.

The files are deliberately cached in ``data/raw`` so a later build does not
silently depend on a live API response.
"""

from pathlib import Path
import gzip
import json
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
WEATHER_URL = 'https://data.meteostat.net/daily/2025/47108.csv.gz'


def download(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'seoul-metro-data-build/1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    weather_bytes = download(WEATHER_URL)
    weather_path = RAW_DIR / 'seoul_weather_2025_meteostat.csv'
    weather_path.write_bytes(gzip.decompress(weather_bytes))

    print(json.dumps({
        'weather': str(weather_path),
        'weatherBytes': len(weather_bytes),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
