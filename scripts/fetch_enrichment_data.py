#!/usr/bin/env python3
"""Fetch the two public external inputs used by the enrichment experiment.

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
KRIC_URL = (
    'https://portal.esrikr.com/arcgis/rest/services/Hosted/KRIC_SubwayStation/'
    'FeatureServer/0/query?where=1%3D1&outFields='
    'stationname,originallinename,linename,linenumber,stationnumber,'
    'stationaddress,stationlatitude,stationlongitude&returnGeometry=false&'
    'resultRecordCount=2000&f=json'
)


def download(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'seoul-metro-data-build/1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    weather_bytes = download(WEATHER_URL)
    weather_path = RAW_DIR / 'seoul_weather_2025_meteostat.csv'
    weather_path.write_bytes(gzip.decompress(weather_bytes))

    stations = json.loads(download(KRIC_URL).decode('utf-8'))
    if stations.get('error') or not stations.get('features'):
        raise RuntimeError(f'KRIC station response is invalid: {stations.get("error")}')
    station_path = RAW_DIR / 'kric_subway_stations.json'
    station_path.write_text(json.dumps(stations, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'weather': str(weather_path),
        'weatherBytes': len(weather_bytes),
        'stations': str(station_path),
        'stationFeatures': len(stations['features']),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
