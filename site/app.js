const numberFormat = new Intl.NumberFormat('ko-KR');
const compactFormat = new Intl.NumberFormat('ko-KR', { notation: 'compact', maximumFractionDigits: 1 });

async function loadData() {
  const response = await fetch('generated/site_data.json');
  if (!response.ok) throw new Error('사이트 데이터가 없습니다. build_site_data.py를 먼저 실행하세요.');
  return response.json();
}

function setText(data) {
  const values = {
    ...data.summary,
    ...data.audit,
    ...data.audit.model,
    ...data.eda.daily,
    ...data.spatial,
    ...(data.weatherAnalysis?.summary || {}),
  };
  document.querySelectorAll('[data-value]').forEach((element) => {
    const key = element.dataset.value;
    if (values[key] !== undefined) element.textContent = typeof values[key] === 'number' ? numberFormat.format(values[key]) : values[key];
  });
  const totalPassengers = document.querySelector('[data-value="totalPassengers"]');
  if (totalPassengers) totalPassengers.textContent = compactFormat.format(values.totalPassengers);
}

function chartDefaults() {
  return { responsive: true, maintainAspectRatio: false, interaction: { intersect: false, mode: 'index' }, plugins: { legend: { display: false }, tooltip: { backgroundColor: '#18241e', padding: 12, titleFont: { family: 'DM Mono' }, bodyFont: { family: 'Manrope' }, displayColors: false } }, scales: { x: { grid: { display: false }, ticks: { color: '#708078', font: { family: 'DM Mono', size: 9 }, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } }, y: { grid: { color: '#d8e1db' }, ticks: { color: '#708078', font: { family: 'DM Mono', size: 9 }, callback: (value) => compactFormat.format(value) } } } };
}

function drawCharts(data) {
  const colors = ['#1c5b43', '#ff8b4a', '#d5b73b', '#719d86', '#9aaea1', '#c4d0c8', '#708078', '#b3c4ba'];
  const daily = data.daily;
  const dailyChart = document.getElementById('dailyChart');
  if (dailyChart) new Chart(dailyChart, { type: 'line', data: { labels: daily.labels, datasets: [{ data: daily.values, borderColor: '#1c5b43', backgroundColor: 'rgba(185,227,204,.35)', fill: true, borderWidth: 2, pointRadius: 0, tension: .25 }] }, options: chartDefaults() });
  const lineChart = document.getElementById('lineChart');
  if (lineChart) new Chart(lineChart, { type: 'doughnut', data: { labels: data.lines.labels, datasets: [{ data: data.lines.values, backgroundColor: colors, borderWidth: 0, hoverOffset: 8 }] }, options: { responsive: true, maintainAspectRatio: false, cutout: '68%', plugins: { legend: { display: true, position: 'right', labels: { color: '#526159', boxWidth: 10, boxHeight: 10, padding: 13, font: { family: 'DM Mono', size: 9 } } }, tooltip: { backgroundColor: '#18241e', padding: 12, displayColors: false } } } });
  const direction = data.direction;
  const directionChart = document.getElementById('directionChart');
  if (directionChart) new Chart(directionChart, { type: 'line', data: { labels: direction.labels, datasets: [{ label: '승차', data: direction.boarding, borderColor: '#ff8b4a', borderWidth: 2, pointRadius: 2, tension: .3 }, { label: '하차', data: direction.alighting, borderColor: '#1c5b43', borderWidth: 2, pointRadius: 2, tension: .3 }] }, options: { ...chartDefaults(), plugins: { ...chartDefaults().plugins, legend: { display: true, labels: { color: '#526159', boxWidth: 10, font: { family: 'DM Mono', size: 9 } } } } } });
}

function drawHeatmap(data) {
  const container = document.getElementById('heatmap');
  if (!container) return;
  const weekdays = ['월', '화', '수', '목', '금', '토', '일'];
  const heat = data.heatmap;
  container.appendChild(Object.assign(document.createElement('span'), { className: 'heat-label', textContent: '' }));
  weekdays.forEach((day) => container.appendChild(Object.assign(document.createElement('span'), { className: 'heat-label', textContent: day })));
  heat.labels.forEach((label, rowIndex) => {
    const displayLabel = container.dataset.compactLabels === 'true'
      ? label.replace('시간대', '').replace('시이전', '시 이전').replace('시이후', '시 이후')
      : label;
    container.appendChild(Object.assign(document.createElement('span'), { className: 'heat-label', textContent: displayLabel }));
    heat.values[rowIndex].forEach((value, columnIndex) => {
      const cell = document.createElement('span');
      cell.className = 'heat-cell';
      const ratio = (value - heat.min) / (heat.max - heat.min || 1);
      cell.style.backgroundColor = `rgba(28,91,67,${(0.08 + ratio * .9).toFixed(2)})`;
      cell.title = `${label} · ${weekdays[columnIndex]}요일 · ${numberFormat.format(value)}명`;
      container.appendChild(cell);
    });
  });
}

function drawRanking(data) {
  const container = document.getElementById('stationRanking');
  if (!container) return;
  const max = data.stations.values[0].value;
  data.stations.values.forEach((station, index) => {
    const row = document.createElement('div'); row.className = 'ranking-row';
    row.innerHTML = `<span class="rank-index">0${index + 1}</span><div><span class="rank-name">${station.name}</span><div class="rank-bar-wrap"><div class="rank-bar" style="width:${Math.max(4, station.value / max * 100)}%"></div></div></div><span class="rank-number">${(station.value / 1000000).toFixed(1)}m</span>`;
    container.appendChild(row);
  });
}

function drawStationMap(data) {
  const container = document.getElementById('stationMap');
  const stations = data.spatial?.stations;
  if (!container || !stations?.length || !window.L) return;

  const map = window.L.map(container, {
    attributionControl: true,
    scrollWheelZoom: false,
    zoomControl: true,
  });
  window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors',
  }).addTo(map);

  const max = Math.max(...stations.map((station) => station.value));
  const bounds = stations.map((station) => [station.lat, station.lng]);
  map.fitBounds(bounds, { padding: [34, 34], maxZoom: 12 });

  if (typeof window.L.heatLayer === 'function') {
    const heatPoints = stations.map((station) => [
      station.lat,
      station.lng,
      Math.sqrt(station.value / max),
    ]);
    window.L.heatLayer(heatPoints, {
      radius: 42,
      blur: 34,
      minOpacity: .28,
      gradient: { .2: '#b9e3cc', .55: '#d5b73b', .78: '#ff8b4a', 1: '#c75238' },
    }).addTo(map);
  }

  stations.forEach((station) => {
    const ratio = station.value / max;
    const color = ratio > .8 ? '#c75238' : ratio > .58 ? '#ff8b4a' : ratio > .38 ? '#d5b73b' : '#1c5b43';
    const radius = 8 + Math.sqrt(ratio) * 17;
    const marker = window.L.circleMarker([station.lat, station.lng], {
      radius,
      color: '#ffffff',
      weight: 2,
      fillColor: color,
      fillOpacity: .9,
    }).addTo(map);
    marker.bindTooltip(`<b>${station.rank}. ${station.name}</b><br>${numberFormat.format(station.value)}명 / 2025년`, {
      direction: 'top',
      offset: [0, -radius],
      opacity: .96,
    });
  });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
  }[character]));
}

function drawPredictionExample(data) {
  const container = document.getElementById('predictionExample');
  const example = data.audit?.model?.testExample;
  if (!container || !example) return;
  const weekdayLabels = ['월', '화', '수', '목', '금', '토', '일'];
  const calendarRows = [
    ['weekday', `요일 ${weekdayLabels[example.calendar.weekday] ?? example.calendar.weekday}`],
    ['month', `${example.calendar.month}월`],
    ['week_of_year', `ISO ${example.calendar.week_of_year}주`],
    ['day_of_month', `${example.calendar.day_of_month}일`],
    ['day_of_year', `${example.calendar.day_of_year}일째`],
    ['is_weekend', example.calendar.is_weekend ? '주말 = 1' : '평일 = 0'],
  ];
  const historyRows = Object.entries(example.history).map(([key, value]) => `<div><span>${escapeHtml(key)}</span><strong>${numberFormat.format(value)}명</strong></div>`).join('');
  const predictionRows = ['Seasonal naive', 'Ridge', 'HistGradientBoosting'].map((name) => `
    <div class="example-prediction-row">
      <strong>${escapeHtml(name)}</strong>
      <span>${numberFormat.format(example.predictions[name])}명</span>
      <span>절대오차 ${numberFormat.format(example.absoluteErrors[name])}명</span>
    </div>`).join('');
  container.innerHTML = `
    <div class="prediction-example-head">
      <div><span>TEST DATE</span><strong>${escapeHtml(example.date)}</strong></div>
      <div><span>SERIES KEY</span><strong>${escapeHtml(example.series.line)} · ${escapeHtml(example.series.station)} · ${escapeHtml(example.series.direction)}</strong></div>
      <div><span>STATION CODE</span><strong>${escapeHtml(example.series.stationCode)}</strong></div>
      <div><span>ACTUAL TARGET</span><strong>${numberFormat.format(example.actual)}명</strong></div>
    </div>
    <div class="prediction-example-grid">
      <article><p>01 / IDENTITY INPUTS</p><h3>시리즈가 무엇인가</h3><dl><div><dt>line</dt><dd>${escapeHtml(example.series.line)}</dd></div><div><dt>station</dt><dd>${escapeHtml(example.series.station)}</dd></div><div><dt>direction</dt><dd>${escapeHtml(example.series.direction)}</dd></div></dl></article>
      <article><p>02 / CALENDAR INPUTS</p><h3>목표일 t에서 이미 아는 값</h3><dl>${calendarRows.map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`).join('')}</dl></article>
      <article><p>03 / HISTORY INPUTS</p><h3>t 이전의 같은 시리즈</h3><div class="example-history">${historyRows}</div></article>
    </div>
    <div class="example-predictions"><p>04 / THREE MODEL OUTPUTS</p>${predictionRows}</div>`;
}

function drawModels(data) {
  const container = document.getElementById('modelTable');
  if (!container) return;
  container.innerHTML = '<div class="model-row header"><span>MODEL</span><span>MAE</span><span>RMSE</span><span>WAPE</span></div>';
  data.models.forEach((model) => {
    const row = document.createElement('div'); row.className = `model-row ${model.best ? 'best' : ''}`;
    row.innerHTML = `<strong>${model.name}</strong><span>${numberFormat.format(Math.round(model.mae))}</span><span>${numberFormat.format(Math.round(model.rmse))}</span><span>${(model.wape * 100).toFixed(1)}%</span>`;
    container.appendChild(row);
  });
}

function formatMetric(value) {
  return numberFormat.format(Math.round(value));
}

function formatChange(value) {
  const sign = value > 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(1)}%`;
}

function drawWeatherAnalysis(data) {
  const analysis = data.weatherAnalysis;
  if (!analysis) return;
  const { weather, experiments, baselines, summary } = analysis;
  const decision = document.getElementById('weatherDecision');
  if (decision) decision.textContent = summary.weatherDecision;

  const sourceFacts = document.getElementById('weatherSources');
  if (sourceFacts) {
    sourceFacts.innerHTML = `
      <article><span>WEATHER COVERAGE</span><strong>${weather.dateStart}–${weather.dateEnd}</strong><p>${numberFormat.format(weather.rowCount)} daily rows · Seoul WMO ${weather.stationId}</p></article>
      <article><span>USABLE AT t</span><strong>t−1 only</strong><p>8 lagged weather variables are eligible for an operational candidate</p></article>
      <article><span>ABLATION STEPS</span><strong>0 → 4 → 8</strong><p>baseline, thermal-humidity subset, then full weather feature set</p></article>`;
  }

  const weatherLedger = document.getElementById('weatherLedger');
  if (weatherLedger) {
    weatherLedger.innerHTML = weather.fields.map((field) => `<tr>
      <td><code>${escapeHtml(field.name)}</code></td><td>${escapeHtml(field.label)}</td><td>${escapeHtml(field.unit)}</td>
      <td>${numberFormat.format(field.missingBeforePolicy)}</td><td><code>${escapeHtml(field.strictFeature)}</code></td>
    </tr>`).join('');
  }

  const resultTable = document.getElementById('weatherResults');
  const names = {
    lagged_thermal_humidity: '전날 열·습도 4개',
    lagged_weather_full: '전날 전체 날씨 8개',
    target_weather_oracle: '목표일 사후 날씨 8개',
  };
  if (resultTable) {
    const baseRows = [
      `<div class="weather-result-row baseline"><strong>Seasonal naive</strong><span>운영 기준선</span><b>${formatMetric(baselines.seasonal.validation.mae)}</b><b>${formatMetric(baselines.seasonal.test.mae)}</b><i>최종 비교 기준</i></div>`,
      `<div class="weather-result-row baseline"><strong>HGB / 날씨 없음</strong><span>14개 기본 변수</span><b>${formatMetric(baselines.hgb.validation.mae)}</b><b>${formatMetric(baselines.hgb.test.mae)}</b><i>날씨 ablation 기준</i></div>`,
    ];
    const rows = experiments.map((experiment) => `<div class="weather-result-row ${experiment.eligibility.includes('불가') ? 'oracle' : ''}">
      <strong>${names[experiment.id] || escapeHtml(experiment.id)}</strong><span>${escapeHtml(experiment.eligibility)} · ${experiment.featureCount} inputs</span>
      <b>${formatMetric(experiment.validation.mae)}</b><b>${formatMetric(experiment.test.mae)}</b>
      <i>HGB 대비 ${formatChange(experiment.comparison.testMaeChangeVsHgb)}</i></div>`);
    resultTable.innerHTML = `<div class="weather-result-row weather-result-head"><span>VARIANT</span><span>AVAILABILITY / INPUTS</span><span>VALID MAE</span><span>TEST MAE</span><span>TEST Δ vs HGB</span></div>${baseRows.join('')}${rows.join('')}`;
  }

  if (!window.Chart) return;
  const dailyChart = document.getElementById('weatherDailyChart');
  if (dailyChart) new Chart(dailyChart, {
    data: {
      labels: weather.visual.daily.labels,
      datasets: [
        { type: 'line', label: '평균 기온 (°C)', data: weather.visual.daily.temp, borderColor: '#ff8b4a', backgroundColor: 'rgba(255,139,74,.12)', fill: true, pointRadius: 0, borderWidth: 1.7, tension: .25, yAxisID: 'temperature' },
        { type: 'bar', label: '강수량 (mm)', data: weather.visual.daily.prcp, backgroundColor: 'rgba(28,91,67,.45)', borderWidth: 0, yAxisID: 'precipitation' },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, interaction: { intersect: false, mode: 'index' }, plugins: { legend: { display: true, labels: { color: '#526159', boxWidth: 10, font: { family: 'DM Mono', size: 9 } } }, tooltip: { backgroundColor: '#18241e', displayColors: false } }, scales: { x: { grid: { display: false }, ticks: { color: '#708078', maxTicksLimit: 8, font: { family: 'DM Mono', size: 9 } } }, temperature: { position: 'left', grid: { color: '#d8e1db' }, ticks: { color: '#ff8b4a' } }, precipitation: { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#1c5b43' }, beginAtZero: true } } },
  });
  const monthlyChart = document.getElementById('weatherMonthlyChart');
  if (monthlyChart) new Chart(monthlyChart, {
    data: { labels: weather.visual.monthly.labels, datasets: [
      { type: 'bar', label: '월 강수량 (mm)', data: weather.visual.monthly.prcp, backgroundColor: 'rgba(28,91,67,.64)', borderWidth: 0, yAxisID: 'precipitation' },
      { type: 'line', label: '월 평균 기온 (°C)', data: weather.visual.monthly.temp, borderColor: '#ff8b4a', pointBackgroundColor: '#ff8b4a', pointRadius: 3, borderWidth: 2, tension: .28, yAxisID: 'temperature' },
      { type: 'line', label: '월 평균 습도 (%)', data: weather.visual.monthly.rhum, borderColor: '#d5b73b', pointRadius: 2, borderWidth: 1.7, tension: .28, yAxisID: 'humidity' },
    ] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: true, labels: { color: '#526159', boxWidth: 10, font: { family: 'DM Mono', size: 9 } } }, tooltip: { backgroundColor: '#18241e', displayColors: false } }, scales: { x: { grid: { display: false }, ticks: { color: '#708078', font: { family: 'DM Mono', size: 9 } } }, precipitation: { position: 'left', beginAtZero: true, grid: { color: '#d8e1db' }, ticks: { color: '#1c5b43' } }, temperature: { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#ff8b4a' } }, humidity: { display: false, min: 0, max: 100 } } },
  });
  const ablationChart = document.getElementById('weatherAblationChart');
  if (ablationChart) new Chart(ablationChart, {
    type: 'bar',
    data: { labels: ['Seasonal', 'HGB / no weather', 't−1 thermal + humidity', 't−1 full weather', 't oracle'], datasets: [{ label: 'Test MAE (lower is better)', data: [baselines.seasonal.test.mae, baselines.hgb.test.mae, ...experiments.map((item) => item.test.mae)], backgroundColor: ['#1c5b43', '#9aaea1', '#d5b73b', '#ff8b4a', '#c75238'], borderWidth: 0 }] },
    options: { ...chartDefaults(), plugins: { ...chartDefaults().plugins, legend: { display: false }, tooltip: { backgroundColor: '#18241e', displayColors: false } }, scales: { x: { ...chartDefaults().scales.x, ticks: { color: '#526159', font: { family: 'DM Mono', size: 8 }, maxRotation: 0, autoSkip: false } }, y: { ...chartDefaults().scales.y, beginAtZero: false } } },
  });
}

function drawModelComparison(data) {
  const container = document.getElementById('modelComparison');
  if (!container) return;
  const metric = (value) => numberFormat.format(Math.round(value));
  container.innerHTML = [
    '<div class="comparison-row comparison-header"><span>MODEL</span><span>VALIDATION<br>MAE</span><span>TEST<br>MAE</span><span>TEST<br>WAPE</span></div>',
    ...data.models.map((model) => `<div class="comparison-row ${model.best ? 'comparison-best' : ''}"><strong>${model.name}</strong><span>${metric(model.validation?.mae ?? model.mae)}</span><span>${metric(model.mae)}</span><span>${(model.wape * 100).toFixed(1)}%</span></div>`),
  ].join('');
}

function drawBandBars(data) {
  const container = document.getElementById('bandBars');
  if (!container) return;
  const values = data.heatmap.values.map((row) => row.reduce((sum, value) => sum + value, 0));
  const max = Math.max(...values);
  container.innerHTML = data.heatmap.labels.map((label, index) => {
    const compactLabel = label.replace('시간대', '').replace('시이전', '시 이전').replace('시이후', '시 이후');
    const height = Math.max(5, values[index] / max * 100);
    return `<span class="pulse-bar" title="${label} · 평균 ${numberFormat.format(values[index])}명"><i style="height:${height.toFixed(2)}%"></i><b>${compactLabel}</b></span>`;
  }).join('');
}

function renderTableRow(cells, className = '') {
  return `<div class="data-row ${className}">${cells.map((cell) => `<span>${cell}</span>`).join('')}</div>`;
}

function drawEdaTables(data) {
  const weekdayTable = document.getElementById('weekdayTable');
  const bandTable = document.getElementById('bandTable');
  const lineTable = document.getElementById('lineTable');
  const anomalyTable = document.getElementById('anomalyTable');
  const weekdayRows = data.eda.weekday.map((row) => renderTableRow([row.label, numberFormat.format(row.mean), row.isWeekend ? 'weekend' : 'weekday'], row.isWeekend ? 'is-weekend' : ''));
  if (weekdayTable) weekdayTable.innerHTML = renderTableRow(['DAY', 'MEAN', 'TYPE'], 'data-header') + weekdayRows.join('');

  const maxBand = Math.max(...data.eda.bands.map((row) => row.total));
  const bandRows = [...data.eda.bands].sort((left, right) => right.total - left.total).map((row) => renderTableRow([row.label, `<div class="table-bar"><i style="width:${row.total / maxBand * 100}%"></i></div>`, `${(row.share * 100).toFixed(1)}%`], ''));
  if (bandTable) bandTable.innerHTML = renderTableRow(['TIME', 'SCALE', 'SHARE'], 'data-header') + bandRows.join('');

  const maxLine = Math.max(...data.eda.lines.map((row) => row.total));
  const lineRows = data.eda.lines.map((row) => renderTableRow([row.label, `<div class="table-bar"><i style="width:${row.total / maxLine * 100}%"></i></div>`, `${(row.share * 100).toFixed(1)}%`], ''));
  if (lineTable) lineTable.innerHTML = renderTableRow(['LINE', 'SCALE', 'SHARE'], 'data-header') + lineRows.join('');

  const highRows = data.eda.highDates.map((row) => renderTableRow([`<b class="high-mark">HIGH</b> ${row.date} ${row.weekday}`, numberFormat.format(row.total)]));
  const lowRows = data.eda.lowDates.map((row) => renderTableRow([`<b class="low-mark">LOW</b> ${row.date} ${row.weekday}`, numberFormat.format(row.total)]));
  if (anomalyTable) anomalyTable.innerHTML = highRows.join('') + lowRows.join('');
}

loadData().then((data) => { setText(data); drawCharts(data); drawHeatmap(data); drawRanking(data); drawStationMap(data); drawPredictionExample(data); drawModels(data); drawModelComparison(data); drawBandBars(data); drawEdaTables(data); drawWeatherAnalysis(data); }).catch((error) => { console.error(error); });
