# seoul-metro

2025년 서울교통공사 1–8호선 승하차 기록을 검증하고, 다음 날의
`노선 × 역번호 × 승하차 방향 × 18개 시간대` 수요를 예측하는 재현 가능한
데이터 프로젝트입니다. 통계·ML 기준선과 함께 56일 인과적 시계열 인코더,
역·노선·승하차 임베딩, 기상예보·특수일 제거 실험을 비교합니다.

- 공개 사이트: <https://gohyunsu.github.io/seoul-metro/>
- 원자료: [서울 열린데이터광장 OA-12921](https://data.seoul.go.kr/dataList/OA-12921/F/1/datasetView.do)

## 연구 흐름

1. **문제 정의** — 목표 날짜 `t`의 시리즈별 06–23시 승객 수 18개를 예측 대상으로 고정합니다.
2. **데이터 수집·감사** — 원본 행, 완전 공백, 결측, 음수, 0, 중복 키와 날짜 범위를 검사합니다.
3. **데이터 이해·시각화** — 일별 총량, 요일×시간대, 노선, 승하차 방향, 역 순위와 공간 분포를 비교합니다.
4. **전처리·입력 설계** — 56일 문맥, 주간 시차, 시계열×시간대 척도와 운영시점 이전 예보만 사용해 미래 정보 유입을 막습니다.
5. **예측 알고리즘** — 4주 중앙값을 정확한 기준점으로 보존하고, 인과적 TCN이 상한이 제한된 잔차만 학습합니다.
6. **평가·결과 분석** — validation MAE로 모델을 선택하고 마지막 test에서 MAE·RMSE·WAPE·sMAPE와 세그먼트 오차를 보고합니다.
7. **한계·향후 연구** — 단일 연도, 집계 자료, 단일 holdout, 점 예측이라는 경계를 명시합니다.

시간대별 4주 중앙값의 테스트 MAE는 **78.51명**입니다. 일반일과 특수일의
상대 MAE를 함께 사용한 검증 규칙은 예보·특수일 제한 잔차 모델의 게이트 제거
구성을 선택했고, 테스트 MAE **71.11명**, RMSE **167.67명**, WAPE
**7.58%**를 기록했습니다. 세 난수 초기화의 테스트 MAE 평균은 **72.49명**,
표본표준편차는 **1.75명**이었습니다. 날씨는 인과적 기본 모델에는 도움이
되었지만 특수일 입력과 함께 사용할 때 악화되어, 일관된 개선 요인으로
주장하지 않습니다.

## 저장소 구조

- `data/raw/` — 서울 열린데이터광장에서 내려받은 원본 스냅샷
- `scripts/build_site_data.py` — 감사, 집계, 특징 생성, 모델 평가, 사이트 데이터 생성
- `scripts/build_hourly_experiment_data.py` — 시간대 수요·예보 텐서와 감사 산출물 생성
- `scripts/run_hourly_v2_experiments.py` — 기준선 보존형 인과적 잔차 실험과 어블레이션
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
