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
    ...(data.inputProfile?.contract || {}),
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
    ['week_of_year', `ISO ${example.calendar.week_of_year}주`],
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
      <div><span>SERIES KEY</span><strong>${escapeHtml(example.series.line)} · ${escapeHtml(example.series.stationCode)} · ${escapeHtml(example.series.direction)}</strong></div>
      <div><span>DISPLAY NAME</span><strong>${escapeHtml(example.series.station)}</strong></div>
      <div><span>ACTUAL TARGET</span><strong>${numberFormat.format(example.actual)}명</strong></div>
    </div>
    <div class="prediction-example-grid">
      <article><p>01 / IDENTITY INPUTS</p><h3>시리즈가 무엇인가</h3><dl><div><dt>line</dt><dd>${escapeHtml(example.series.line)}</dd></div><div><dt>station_code</dt><dd>${escapeHtml(example.series.stationCode)} · ${escapeHtml(example.series.station)}</dd></div><div><dt>direction</dt><dd>${escapeHtml(example.series.direction)}</dd></div></dl></article>
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

function drawModelComparison(data) {
  const container = document.getElementById('modelComparison');
  if (!container) return;
  const metric = (value) => numberFormat.format(Math.round(value));
  container.innerHTML = [
    '<div class="comparison-row comparison-header"><span>MODEL</span><span>VALIDATION<br>MAE</span><span>TEST<br>MAE</span><span>TEST<br>WAPE</span></div>',
    ...data.models.map((model) => `<div class="comparison-row ${model.best ? 'comparison-best' : ''}"><strong>${model.name}</strong><span>${metric(model.validation?.mae ?? model.mae)}</span><span>${metric(model.mae)}</span><span>${(model.wape * 100).toFixed(1)}%</span></div>`),
  ].join('');
}

function profileChartOptions({ indexAxis = 'x', tickLimit = 8 } = {}) {
  const horizontal = indexAxis === 'y';
  return {
    ...chartDefaults(), indexAxis,
    plugins: { ...chartDefaults().plugins, legend: { display: true, labels: { color: '#526159', boxWidth: 10, font: { family: 'DM Mono', size: 9 } } } },
    scales: {
      x: horizontal
        ? { grid: { color: '#d8e1db' }, ticks: { color: '#708078', font: { family: 'DM Mono', size: 9 }, maxTicksLimit: tickLimit, callback: (value) => numberFormat.format(Math.round(value)) } }
        : { ...chartDefaults().scales.x, ticks: { ...chartDefaults().scales.x.ticks, maxTicksLimit: tickLimit } },
      y: horizontal
        ? { grid: { display: false }, ticks: { color: '#526159', font: { family: 'Noto Sans KR', size: 10 } } }
        : { ...chartDefaults().scales.y, ticks: { ...chartDefaults().scales.y.ticks, callback: (value) => numberFormat.format(Math.round(value)) } },
    },
  };
}

function drawInputProfileOverview(data) {
  const profile = data.inputProfile;
  if (!profile) return;
  const insight = document.getElementById('inputProfileInsight');
  if (insight) {
    const top = profile.ridgePermutationImportance.rows[0];
    insight.innerHTML = `<strong>${escapeHtml(top.label)}</strong><span>Ridge validation에서 이 열을 섞으면 MAE가 ${numberFormat.format(Math.round(top.maeIncrease))}명 늘었습니다.</span>`;
  }
  const chart = document.getElementById('inputOverviewImportanceChart');
  if (!chart || !window.Chart) return;
  const rows = profile.ridgePermutationImportance.rows.slice(0, 5).reverse();
  new Chart(chart, {
    type: 'bar',
    data: { labels: rows.map((row) => row.label), datasets: [{ label: 'validation MAE 증가 (명)', data: rows.map((row) => row.maeIncrease), backgroundColor: '#1c5b43', borderWidth: 0, borderRadius: 3 }] },
    options: profileChartOptions({ indexAxis: 'y', tickLimit: 6 }),
  });
}

function drawInputProfile(data) {
  const profile = data.inputProfile;
  if (!profile) return;
  const { categorical, calendar, history, split, ridgePermutationImportance: importance, seasonalErrorSlices: errors } = profile;
  const categoryChart = document.getElementById('categoryCoverageChart');
  if (categoryChart && window.Chart) {
    new Chart(categoryChart, {
      type: 'bar',
      data: {
        labels: [...categorical.line.map((row) => row.label), ...categorical.direction.map((row) => row.label)],
        datasets: [{
          label: 'feature-ready series-days',
          data: [...categorical.line.map((row) => row.rows), ...categorical.direction.map((row) => row.rows)],
          backgroundColor: [...categorical.line.map(() => '#1c5b43'), ...categorical.direction.map(() => '#ff8b4a')], borderWidth: 0, borderRadius: 3,
        }],
      },
      options: profileChartOptions({ tickLimit: 10 }),
    });
  }
  const stationFacts = document.getElementById('stationCodeFacts');
  if (stationFacts) {
    const station = categorical.stationCode;
    stationFacts.innerHTML = `<div><span>CATEGORIES</span><strong>${numberFormat.format(station.levels)}</strong></div><div><span>MEDIAN SERIES / CODE</span><strong>${station.medianSeries.toFixed(1)}</strong></div><div><span>RANGE</span><strong>${station.minSeries}–${station.maxSeries}</strong></div>`;
  }
  const stationMultiplicity = document.getElementById('stationMultiplicity');
  if (stationMultiplicity) {
    const max = Math.max(...categorical.stationCode.multiplicity.map((row) => row.stationCodeCount));
    stationMultiplicity.innerHTML = categorical.stationCode.multiplicity.map((row) => `<div><span>${row.seriesPerCode} series</span><i><b style="width:${(row.stationCodeCount / max * 100).toFixed(1)}%"></b></i><strong>${numberFormat.format(row.stationCodeCount)} codes</strong></div>`).join('');
  }
  const weekdayChart = document.getElementById('weekdayProfileChart');
  if (weekdayChart && window.Chart) {
    new Chart(weekdayChart, {
      type: 'bar',
      data: { labels: calendar.weekday.map((row) => row.label), datasets: [{ label: '평균 target', data: calendar.weekday.map((row) => row.targetMean), backgroundColor: '#1c5b43', borderWidth: 0, borderRadius: 3 }, { label: '중앙 target', data: calendar.weekday.map((row) => row.targetMedian), backgroundColor: '#b9e3cc', borderWidth: 0, borderRadius: 3 }] },
      options: profileChartOptions({ tickLimit: 7 }),
    });
  }
  const isoWeekChart = document.getElementById('isoWeekProfileChart');
  if (isoWeekChart && window.Chart) {
    new Chart(isoWeekChart, {
      type: 'line',
      data: { labels: calendar.isoWeek.map((row) => `W${String(row.week).padStart(2, '0')}`), datasets: [{ label: '중앙 target', data: calendar.isoWeek.map((row) => row.targetMedian), borderColor: '#ff8b4a', backgroundColor: 'rgba(255,139,74,.12)', fill: true, borderWidth: 2, pointRadius: 0, tension: .25 }] },
      options: profileChartOptions({ tickLimit: 9 }),
    });
  }
  const calendarTable = document.getElementById('calendarProfileTable');
  if (calendarTable) calendarTable.innerHTML = `<div class="profile-table-head"><span>WEEKDAY</span><span>ROWS</span><span>TARGET MEAN</span><span>TARGET MEDIAN</span></div>${calendar.weekday.map((row) => `<div><strong>${row.label}</strong><span>${numberFormat.format(row.rows)}</span><span>${numberFormat.format(Math.round(row.targetMean))}명</span><span>${numberFormat.format(Math.round(row.targetMedian))}명</span></div>`).join('')}`;
  const quantileChart = document.getElementById('historyQuantileChart');
  if (quantileChart && window.Chart) {
    const colors = ['#1c5b43', '#ff8b4a', '#d5b73b', '#719d86', '#708078'];
    new Chart(quantileChart, {
      type: 'line',
      data: { labels: history.quantiles[0].quantiles.map((row) => row.label), datasets: history.quantiles.map((row, index) => ({ label: row.label, data: row.quantiles.map((point) => point.value), borderColor: colors[index], pointBackgroundColor: colors[index], pointRadius: 2, borderWidth: 2, tension: .25 })) },
      options: profileChartOptions({ tickLimit: 5 }),
    });
  }
  const relationshipChart = document.getElementById('lagRelationshipChart');
  if (relationshipChart && window.Chart) {
    const bins = history.targetByLag7Decile;
    new Chart(relationshipChart, {
      type: 'line',
      data: { labels: bins.map((row) => row.label), datasets: [{ label: 'lag_7 중앙값', data: bins.map((row) => row.inputMedian), borderColor: '#719d86', borderWidth: 2, pointRadius: 2, tension: .24 }, { label: 'target 평균', data: bins.map((row) => row.targetMean), borderColor: '#ff8b4a', backgroundColor: 'rgba(255,139,74,.12)', fill: true, borderWidth: 2, pointRadius: 2, tension: .24 }] },
      options: profileChartOptions({ tickLimit: 10 }),
    });
  }
  const historyTable = document.getElementById('historyProfileTable');
  if (historyTable) {
    const correlationByKey = Object.fromEntries(history.correlation.map((row) => [row.key, row.pearson]));
    historyTable.innerHTML = `<div class="profile-table-head history-head"><span>INPUT</span><span>P05</span><span>P50</span><span>P95</span><span>MEAN</span><span>PEARSON r</span></div>${history.quantiles.map((row) => { const point = Object.fromEntries(row.quantiles.map((item) => [item.label, item.value])); return `<div class="history-head"><strong>${row.label}</strong><span>${numberFormat.format(Math.round(point.P05))}</span><span>${numberFormat.format(Math.round(point.P50))}</span><span>${numberFormat.format(Math.round(point.P95))}</span><span>${numberFormat.format(Math.round(row.mean))}</span><span>${correlationByKey[row.key].toFixed(3)}</span></div>`; }).join('')}`;
  }
  const splitTable = document.getElementById('splitProfileTable');
  if (splitTable) {
    const rows = [['TRAIN', split.train], ['VALIDATION', split.validation], ['TEST', split.test]];
    splitTable.innerHTML = `<div class="split-profile-head"><span>SPLIT</span><span>ROWS</span><span>TARGET P05</span><span>TARGET P50</span><span>TARGET P95</span><span>LAG_7 P50</span><span>ROLLING_7 P50</span></div>${rows.map(([label, row]) => { const points = Object.fromEntries(row.target.quantiles.map((item) => [item.label, item.value])); return `<div><strong>${label}</strong><span>${numberFormat.format(row.rows)}</span><span>${numberFormat.format(Math.round(points.P05))}</span><span>${numberFormat.format(Math.round(points.P50))}</span><span>${numberFormat.format(Math.round(points.P95))}</span><span>${numberFormat.format(Math.round(row.lag7Median))}</span><span>${numberFormat.format(Math.round(row.rolling7Median))}</span></div>`; }).join('')}`;
  }
  const importanceChart = document.getElementById('importanceChart');
  if (importanceChart && window.Chart) {
    const rows = [...importance.rows].reverse();
    new Chart(importanceChart, {
      type: 'bar',
      data: { labels: rows.map((row) => row.label), datasets: [{ label: 'validation MAE 증가 (명)', data: rows.map((row) => row.maeIncrease), backgroundColor: rows.map((row) => row.maeIncrease >= 0 ? '#1c5b43' : '#c75238'), borderWidth: 0, borderRadius: 3 }] },
      options: profileChartOptions({ indexAxis: 'y', tickLimit: 10 }),
    });
  }
  const importanceMethod = document.getElementById('importanceMethod');
  if (importanceMethod) importanceMethod.textContent = `기준 validation MAE ${numberFormat.format(Math.round(importance.baselineValidationMae))}명 · ${importance.method}`;
  const errorWeekdayChart = document.getElementById('errorWeekdayChart');
  if (errorWeekdayChart && window.Chart) {
    new Chart(errorWeekdayChart, { type: 'bar', data: { labels: errors.weekday.map((row) => row.label), datasets: [{ label: 'test MAE', data: errors.weekday.map((row) => row.mae), backgroundColor: '#1c5b43', borderWidth: 0, borderRadius: 3 }] }, options: profileChartOptions({ tickLimit: 7 }) });
  }
  const errorDemandChart = document.getElementById('errorDemandChart');
  if (errorDemandChart && window.Chart) {
    new Chart(errorDemandChart, { type: 'bar', data: { labels: errors.demand.map((row) => row.label), datasets: [{ label: 'test MAE', data: errors.demand.map((row) => row.mae), backgroundColor: '#ff8b4a', borderWidth: 0, borderRadius: 3 }] }, options: profileChartOptions({ tickLimit: 4 }) });
  }
  const errorTable = document.getElementById('errorSliceTable');
  if (errorTable) errorTable.innerHTML = `<div class="profile-table-head"><span>DEMAND SLICE</span><span>ROWS</span><span>ACTUAL MEAN</span><span>SEASONAL MAE</span></div>${errors.demand.map((row) => `<div><strong>${row.label}</strong><span>${numberFormat.format(row.rows)}</span><span>${numberFormat.format(Math.round(row.actualMean))}명</span><span>${numberFormat.format(Math.round(row.mae))}명</span></div>`).join('')}`;
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

loadData().then((data) => { setText(data); drawCharts(data); drawHeatmap(data); drawRanking(data); drawStationMap(data); drawPredictionExample(data); drawModels(data); drawModelComparison(data); drawInputProfileOverview(data); drawInputProfile(data); drawEdaTables(data); }).catch((error) => { console.error(error); });
