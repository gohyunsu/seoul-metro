# seoul-metro

2025년 서울교통공사 1–8호선 승하차 기록을 검증하고, 시간·노선·방향·역별 패턴을 이해한 뒤, 다음 날의 `노선 × 역번호 × 승하차 방향`별 일일 수요를 예측하는 재현 가능한 데이터 프로젝트입니다.

- 공개 사이트: <https://gohyunsu.github.io/seoul-metro/>
- 원자료: [서울 열린데이터광장 OA-12921](https://data.seoul.go.kr/dataList/OA-12921/F/1/datasetView.do)

## 연구 흐름

1. **문제 정의** — 목표 날짜 `t`의 시리즈별 일일 승객 수를 예측 대상으로 고정합니다.
2. **데이터 수집·감사** — 원본 행, 완전 공백, 결측, 음수, 0, 중복 키와 날짜 범위를 검사합니다.
3. **데이터 이해·시각화** — 일별 총량, 요일×시간대, 노선, 승하차 방향, 역 순위와 공간 분포를 비교합니다.
4. **전처리·입력 설계** — 일일 목표와 10개 입력을 만들고, `shift` 우선 규칙과 28일 warm-up으로 미래 정보 유입을 막습니다.
5. **예측 알고리즘** — Seasonal naive, Ridge, HistGradientBoosting을 같은 입력과 시간 분할에서 비교합니다.
6. **평가·결과 분석** — validation MAE로 모델을 선택하고 마지막 test에서 MAE·RMSE·WAPE·sMAPE와 세그먼트 오차를 보고합니다.
7. **한계·향후 연구** — 단일 연도, 집계 자료, 단일 holdout, 점 예측이라는 경계를 명시합니다.

현재 validation MAE가 가장 낮은 모델은 7일 전 같은 시리즈의 값을 쓰는 **Seasonal naive**입니다. 테스트 MAE는 약 **1,008명**, WAPE는 **5.9%**입니다. 이는 복잡한 후보보다 주간 반복이 더 안정적인 신호였다는 결과이며, 수요 변화의 원인이나 실제 혼잡도를 뜻하지 않습니다.

## 저장소 구조

- `data/raw/` — 서울 열린데이터광장에서 내려받은 원본 스냅샷
- `scripts/build_site_data.py` — 감사, 집계, 특징 생성, 모델 평가, 사이트 데이터 생성
- `reports/` — 감사 JSON, 입력 프로파일, 모델 지표 CSV, 정적 그림
- `site/generated/site_data.json` — 브라우저가 읽는 생성 데이터 계약
- `site/` — GitHub Pages에 게시되는 메인·상세 페이지
- `docs/` — 데이터 범위, 방법론, 분석 결정과 연구 기록

## 로컬 재현

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_site_data.py
python -m http.server 8000 --directory site
```

`http://localhost:8000`에서 생성된 사이트를 확인할 수 있습니다. 재실행 후에는 `reports/data_audit.json`, `reports/model_metrics.csv`, `reports/input_profile.json`과 `site/generated/site_data.json`을 함께 비교해야 합니다.

## 데이터 출처와 이용 범위

승하차 원본은 서울교통공사가 제공하고 서울 열린데이터광장이 공개한 2025년 파일입니다. CP949 인코딩 원본을 `data/raw/`에 보존하며 빌드 스크립트는 이를 읽기만 합니다. 원자료 또는 파생 결과를 이용·배포할 때에는 [공식 데이터 페이지](https://data.seoul.go.kr/dataList/OA-12921/F/1/datasetView.do)의 최신 출처 표기와 이용 조건을 확인해야 합니다.
