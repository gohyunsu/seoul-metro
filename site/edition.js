const fmt = new Intl.NumberFormat('en-US');
const serviceHours = Array.from({ length: 18 }, (_, index) => String(index + 6).padStart(2, '0'));

function weekdayCounts(dates, year) {
  const counts = Array(7).fill(0);
  dates.forEach((date) => {
    const day = new Date(year + '-' + date + 'T12:00:00Z').getUTCDay();
    if (!Number.isNaN(day)) counts[(day + 6) % 7] += 1;
  });
  return counts;
}

function dayProfiles(data) {
  const counts = weekdayCounts(data.daily.labels, data.audit.dateStart.slice(0, 4));
  const rows = data.heatmap.values.slice(1, 19);
  const weighted = (values, from, to) => {
    const count = counts.slice(from, to).reduce((sum, value) => sum + value, 0);
    return values.slice(from, to).reduce((sum, value, index) => sum + value * counts[index + from], 0) / count;
  };
  return {
    weekday: rows.map((row) => weighted(row, 0, 5)),
    weekend: rows.map((row) => weighted(row, 5, 7)),
  };
}

function renderHero(data) {
  const container = document.getElementById('heroChart');
  if (!container) return;
  const bandValues = new Map(data.eda.bands.map((band) => [band.label, band.dailyMean]));
  const values = data.heatmap.labels.slice(1, 19).map((label) => bandValues.get(label) || 0);
  const max = Math.max(...values);
  container.replaceChildren();
  values.forEach((value, index) => {
    const bar = document.createElement('span');
    bar.className = 'art-bar' + (index === 12 ? ' peak' : '');
    bar.style.setProperty('--bar-height', `${Math.max(5, (value / max) * 72)}%`);
    bar.title = `${serviceHours[index]}:00–${String(index + 7).padStart(2, '0')}:00 · daily average ${fmt.format(Math.round(value))} passengers`;
    container.appendChild(bar);
  });
  const points = values.map((value, index) => `${(index + .5) / 18 * 1000},${300 - (value / max) * 192}`);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 1000 300');
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
  path.setAttribute('points', points.join(' '));
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', '#e2ba77');
  path.setAttribute('stroke-width', '2');
  path.setAttribute('vector-effect', 'non-scaling-stroke');
  svg.appendChild(path);
  container.appendChild(svg);
}

function renderRhythm(profiles, type) {
  const plot = document.getElementById('rhythmPlot');
  const insight = document.getElementById('rhythmInsight');
  if (!plot || !insight) return;
  const values = profiles[type];
  const max = Math.max(...profiles.weekday, ...profiles.weekend);
  const peak = values.indexOf(Math.max(...values));
  plot.replaceChildren();
  values.forEach((value, index) => {
    const bar = document.createElement('div');
    bar.className = 'rhythm-bar' + (index === peak ? ' peak' : '');
    bar.style.setProperty('--bar-height', `${Math.max(3, (value / max) * 88)}%`);
    bar.title = `${serviceHours[index]}:00–${String(index + 7).padStart(2, '0')}:00 · ${fmt.format(Math.round(value))} passengers`;
    const column = document.createElement('i');
    bar.appendChild(column);
    if (index === peak) {
      const label = document.createElement('span');
      label.textContent = fmt.format(Math.round(value));
      bar.appendChild(label);
    }
    plot.appendChild(bar);
  });
  const label = type === 'weekday' ? 'Weekday' : 'Weekend';
  plot.setAttribute('aria-label', `${label} average network ridership by service hour, 06:00–23:00. Peak at ${serviceHours[peak]}:00 with ${fmt.format(Math.round(values[peak]))} passengers`);
  insight.textContent = `${label} peak: ${serviceHours[peak]}:00–${String(peak + 7).padStart(2, '0')}:00 · ${fmt.format(Math.round(values[peak]))} passengers on average`;
  document.querySelectorAll('[data-rhythm]').forEach((button) => {
    const active = button.dataset.rhythm === type;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}

async function start() {
  try {
    const response = await fetch('generated/site_data.json');
    if (!response.ok) throw new Error('data unavailable');
    const data = await response.json();
    renderHero(data);
    const profiles = dayProfiles(data);
    renderRhythm(profiles, 'weekday');
    document.querySelectorAll('[data-rhythm]').forEach((button) => button.addEventListener('click', () => renderRhythm(profiles, button.dataset.rhythm)));
  } catch {
    const plot = document.getElementById('rhythmPlot');
    if (plot) plot.textContent = 'Chart data could not be loaded.';
    const insight = document.getElementById('rhythmInsight');
    if (insight) insight.textContent = 'Please try again later.';
  }
}

document.querySelectorAll('.mobile-menu nav a').forEach((link) => link.addEventListener('click', () => {
  const menu = document.querySelector('.mobile-menu');
  if (menu) menu.open = false;
}));
if (document.getElementById('heroChart')) start();
