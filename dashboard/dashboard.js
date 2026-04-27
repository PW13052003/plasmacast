// =============================================
// CONSTANTS
// =============================================
const CITIES = {
  center_philly:       { name: "Philadelphia",  base: 60  },
  center_nyc:          { name: "New York City", base: 318 },
  center_la:           { name: "Los Angeles",   base: 145 },
  center_chicago:      { name: "Chicago",       base: 102 },
  center_houston:      { name: "Houston",       base: 90  },
  center_phoenix:      { name: "Phoenix",       base: 63  },
  center_san_antonio:  { name: "San Antonio",   base: 57  },
  center_san_diego:    { name: "San Diego",     base: 53  },
  center_dallas:       { name: "Dallas",        base: 50  },
  center_jacksonville: { name: "Jacksonville",  base: 38  },
};

const CITY_STATS = {
  center_philly:       { population: "1,573,916", mae: "5.6%", region: "Northeast"  },
  center_nyc:          { population: "8,478,072", mae: "5.5%", region: "Northeast"  },
  center_la:           { population: "3,878,704", mae: "5.9%", region: "West Coast" },
  center_chicago:      { population: "2,721,308", mae: "6.2%", region: "Midwest"    },
  center_houston:      { population: "2,390,125", mae: "6.4%", region: "South"      },
  center_phoenix:      { population: "1,673,164", mae: "6.4%", region: "Southwest"  },
  center_san_antonio:  { population: "1,526,656", mae: "6.0%", region: "South"      },
  center_san_diego:    { population: "1,404,452", mae: "5.4%", region: "West Coast" },
  center_dallas:       { population: "1,326,087", mae: "6.9%", region: "South"      },
  center_jacksonville: { population: "1,009,833", mae: "5.7%", region: "Southeast"  },
};

const CITY_COORDS = {
  center_philly:       { lat: 40.01,  lon: -75.13  },
  center_nyc:          { lat: 40.66,  lon: -73.94  },
  center_la:           { lat: 34.02,  lon: -118.41 },
  center_chicago:      { lat: 41.84,  lon: -87.68  },
  center_houston:      { lat: 29.79,  lon: -95.39  },
  center_phoenix:      { lat: 33.57,  lon: -112.09 },
  center_san_antonio:  { lat: 29.46,  lon: -98.52  },
  center_san_diego:    { lat: 32.81,  lon: -117.14 },
  center_dallas:       { lat: 32.79,  lon: -96.77  },
  center_jacksonville: { lat: 30.34,  lon: -81.66  },
};

const MAE_DATA = [
  { city: "Dallas",        mae: 6.9 },
  { city: "Houston",       mae: 6.4 },
  { city: "Phoenix",       mae: 6.4 },
  { city: "Chicago",       mae: 6.2 },
  { city: "San Antonio",   mae: 6.0 },
  { city: "Los Angeles",   mae: 5.9 },
  { city: "Jacksonville",  mae: 5.7 },
  { city: "Philadelphia",  mae: 5.6 },
  { city: "New York City", mae: 5.5 },
  { city: "San Diego",     mae: 5.4 },
];

const MODEL_CONFIG = [
  ["Algorithm",        "XGBoost Regressor"],
  ["Training Period",  "2020 – 2022 (3 years)"],
  ["Test Period",      "2023 (1 year)"],
  ["Train/Test Split", "75% / 25%"],
  ["Training Rows",    "10,820"],
  ["Test Rows",        "3,650"],
  ["MAE",              "5.12 donors/day"],
  ["RMSE",             "7.95 donors/day"],
  ["MAE % of Mean",    "5.9%"],
];

const FEATURES = [
  { name: "temp_max",          color: "#0077BB", group: "Weather"  },
  { name: "precipitation",     color: "#0077BB", group: "Weather"  },
  { name: "day_of_week",       color: "#EE7733", group: "Calendar" },
  { name: "is_holiday",        color: "#EE7733", group: "Calendar" },
  { name: "month",             color: "#EE7733", group: "Calendar" },
  { name: "year",              color: "#EE7733", group: "Calendar" },
  { name: "day_of_year",       color: "#EE7733", group: "Calendar" },
  { name: "is_weekend",        color: "#EE7733", group: "Calendar" },
  { name: "season",            color: "#EE7733", group: "Calendar" },
  { name: "donor_lag_7",       color: "#009988", group: "Lag"      },
  { name: "donor_lag_14",      color: "#009988", group: "Lag"      },
  { name: "rolling_7day_avg",  color: "#009988", group: "Rolling"  },
  { name: "rolling_14day_avg", color: "#009988", group: "Rolling"  },
];

// =============================================
// STATE
// =============================================
let selectedCity = "center_philly";

// =============================================
// THEME TOGGLE
// =============================================
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute("data-theme") === "dark";
  html.setAttribute("data-theme", isDark ? "light" : "dark");
  document.getElementById("theme-icon").textContent  = isDark ? "🌙" : "☀️";
  document.getElementById("theme-label").textContent = isDark ? "Dark Mode" : "Light Mode";
}

// =============================================
// PAGE NAVIGATION
// =============================================
function showPage(page, btn) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-link").forEach(b => b.classList.remove("active"));
  document.getElementById("page-" + page).classList.add("active");
  btn.classList.add("active");
  if (page === "insights") renderInsights();
  if (page === "metrics")  renderMetrics();
}

// =============================================
// HELPERS
// =============================================
function getDayName(date) {
  return date.toLocaleDateString("en-US", { weekday: "short" });
}

function getDateLabel(date) {
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function getDonorLevel(count, base) {
  const ratio = count / base;
  if (ratio >= 0.95) return "high";
  if (ratio >= 0.75) return "medium";
  return "low";
}

function getLevelColor(level) {
  if (level === "high")   return "#0077BB";
  if (level === "medium") return "#EE7733";
  return "#CC3311";
}

// =============================================
// ANIMATED COUNTER
// =============================================
function animateCounter(elementId, targetValue, duration = 800) {
  const element = document.getElementById(elementId);
  const start   = performance.now();
  const from    = 0;

  function update(currentTime) {
    const elapsed  = currentTime - start;
    const progress = Math.min(elapsed / duration, 1);

    // Ease out cubic — fast start, slow finish
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(from + (targetValue - from) * eased);

    element.textContent = current.toLocaleString();

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

// =============================================
// CUSTOM DROPDOWN
// =============================================
function buildDropdown() {
  const container = document.getElementById("dropdown-options");
  container.innerHTML = Object.entries(CITIES).map(([id, city]) => `
    <div class="custom-option ${id === selectedCity ? "selected" : ""}"
         onclick="selectCity('${id}')">
      ${city.name}
    </div>`).join("");
}

function updateCityStats(centerId) {
  const city  = CITIES[centerId];
  const stats = CITY_STATS[centerId];
  document.getElementById("stats-city-name").textContent  = city.name;
  document.getElementById("stats-population").textContent = stats.population;
  document.getElementById("stats-base").textContent       = `${city.base} donors`;
  document.getElementById("stats-mae").textContent        = stats.mae;
  document.getElementById("stats-region").textContent     = stats.region;
}

function toggleDropdown() {
  const options = document.getElementById("dropdown-options");
  const trigger = document.querySelector(".custom-select-trigger");
  const chevron = document.getElementById("chevron");
  const isOpen  = options.classList.contains("open");

  options.classList.toggle("open", !isOpen);
  trigger.classList.toggle("open", !isOpen);
  chevron.classList.toggle("open",  !isOpen);
}

function selectCity(centerId) {
  selectedCity = centerId;
  document.getElementById("selected-label").textContent = CITIES[centerId].name;

  document.getElementById("dropdown-options").classList.remove("open");
  document.querySelector(".custom-select-trigger").classList.remove("open");
  document.getElementById("chevron").classList.remove("open");

  buildDropdown();
  updateCityStats(centerId);
  loadForecast();
}

// Close dropdown when clicking outside
document.addEventListener("click", (e) => {
  const dropdown = document.getElementById("city-dropdown");
  if (dropdown && !dropdown.contains(e.target)) {
    document.getElementById("dropdown-options").classList.remove("open");
    const trigger = document.querySelector(".custom-select-trigger");
    if (trigger) trigger.classList.remove("open");
    const chevron = document.getElementById("chevron");
    if (chevron) chevron.classList.remove("open");
  }
});

// =============================================
// FETCH REAL WEATHER FORECAST
// =============================================
async function fetchWeatherForecast(centerId) {
  const coords = CITY_COORDS[centerId];
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${coords.lat}&longitude=${coords.lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=America%2FNew_York&forecast_days=7`;

  try {
    const res  = await fetch(url);
    const data = await res.json();
    return data.daily.time.map((date, i) => ({
      date:          date,
      temp_max:      data.daily.temperature_2m_max[i] ?? 18.0,
      temp_min:      data.daily.temperature_2m_min[i] ?? 12.0,
      precipitation: data.daily.precipitation_sum[i]  ?? 0.0,
    }));
  } catch {
    // Fallback to neutral weather if API fails
    return Array(7).fill({ temp_max: 18.0, temp_min: 12.0, precipitation: 0.0 });
  }
}

// =============================================
// LOAD FORECAST — calls FastAPI with real weather
// =============================================
async function loadForecast() {
  const centerId = selectedCity;
  const city = CITIES[centerId];

  document.getElementById("forecast-chart").innerHTML =
    '<div class="loading">Loading forecast...</div>';
  document.getElementById("today-count").textContent = "--";
  document.getElementById("week-total").textContent  = "--";
  document.getElementById("busiest-day").textContent = "--";
  document.getElementById("quietest-day").textContent = "--";

  const days = [];
  const today = new Date();
  for (let i = 0; i < 7; i++) {
    const d = new Date(today);
    d.setDate(today.getDate() + i);
    days.push(d);
  }

  // Fetch real weather forecast for this city
  const weatherForecast = await fetchWeatherForecast(centerId);

  const forecasts = [];
  for (let i = 0; i < 7; i++) {
    const day     = days[i];
    const weather = weatherForecast[i] || { temp_max: 18.0, precipitation: 0.0 };
    const dateStr = day.toISOString().split("T")[0];

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          center_id:         centerId,
          date:              dateStr,
          temp_max:          weather.temp_max,
          precipitation:     weather.precipitation,
          donor_lag_7:       city.base * 0.95,
          donor_lag_14:      city.base * 0.93,
          rolling_7day_avg:  city.base * 0.94,
          rolling_14day_avg: city.base * 0.93,
        }),
      });
      const data = await res.json();
      forecasts.push({
        day:           getDayName(day),
        date:          getDateLabel(day),
        count:         data.predicted_donors,
        level:         getDonorLevel(data.predicted_donors, city.base),
        temp_max:      Math.round(weather.temp_max),
        temp_min:      Math.round(weather.temp_min),
        precipitation: weather.precipitation,
      });
    } catch {
      forecasts.push({
        day:           getDayName(day),
        date:          getDateLabel(day),
        count:         city.base,
        level:         "medium",
        temp_max:      Math.round(weather.temp_max),
        temp_min:      Math.round(weather.temp_min),
        precipitation: weather.precipitation,
      });
    }
  }

  renderForecast(forecasts, city);
}

// =============================================
// RENDER FORECAST
// =============================================
function renderForecast(forecasts, city) {

  // Last updated timestamp
  const now = new Date();
  document.getElementById("last-updated").textContent =
    `Last updated: ${now.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })} at ${now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })}`;

  const counts = forecasts.map(f => f.count);
  const maxIdx = counts.indexOf(Math.max(...counts));
  const minIdx = counts.indexOf(Math.min(...counts));
  const total  = counts.reduce((a, b) => a + b, 0);
  const maxVal = Math.max(...counts);

  // --- Summary cards ---
  //document.getElementById("today-count").textContent    = forecasts[0].count;
  animateCounter("today-count", forecasts[0].count);
  document.getElementById("today-label").textContent    = `donors expected — ${forecasts[0].day} ${forecasts[0].date}`;
  document.getElementById("busiest-day").textContent    = forecasts[maxIdx].day;
  //document.getElementById("busiest-count").textContent  = `${forecasts[maxIdx].count} donors — ${forecasts[maxIdx].date}`;
  document.getElementById("busiest-count").textContent  = `${forecasts[maxIdx].count} donors — ${forecasts[maxIdx].date}`;
  document.getElementById("quietest-day").textContent   = forecasts[minIdx].day;
  //document.getElementById("quietest-count").textContent = `${forecasts[minIdx].count} donors — ${forecasts[minIdx].date}`;
  document.getElementById("quietest-count").textContent = `${forecasts[minIdx].count} donors — ${forecasts[minIdx].date}`;
  //document.getElementById("week-total").textContent     = total.toLocaleString();
  animateCounter("week-total", total);

  // --- SVG Chart ---
  const W = 900, H = 300;
  const padL = 55, padR = 24, padT = 24, padB = 64;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;

  const yMax   = Math.ceil((maxVal * 1.2) / 10) * 10;
  const yMin   = 0;
  const yTicks = 5;
  const yStep  = Math.ceil(yMax / yTicks / 10) * 10;

  const barCount  = forecasts.length;
  const barGroupW = chartW / barCount;
  const barW      = barGroupW * 0.52;

  function yPos(val) {
    return padT + chartH - ((val - yMin) / (yMax - yMin)) * chartH;
  }

  function xPos(i) {
    return padL + i * barGroupW + barGroupW / 2;
  }

  // Gridlines + Y axis ticks
  let gridLines = "";
  let yLabels   = "";
  for (let t = 0; t <= yTicks; t++) {
    const val = t * yStep;
    if (val > yMax + yStep) break;
    const y = yPos(val);
    gridLines += `
      <line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}"
        stroke="var(--border)" stroke-width="1"
        stroke-dasharray="${t === 0 ? "none" : "4,4"}"/>`;
    yLabels += `
      <text x="${padL - 10}" y="${y + 4}" text-anchor="end"
        font-size="11" fill="var(--text-muted)"
        font-family="Inter, sans-serif">${val}</text>`;
  }

  // Bars + X labels
  let bars    = "";
  let xLabels = "";
  forecasts.forEach((f, i) => {
    const x       = xPos(i);
    const bh      = Math.max(((f.count - yMin) / (yMax - yMin)) * chartH, 4);
    const by      = padT + chartH - bh;
    const col     = getLevelColor(f.level);
    const isToday = i === 0;

    bars += `
      <g style="cursor:pointer;">
        <rect x="${x - barW/2}" y="${by}" width="${barW}" height="${bh}"
          fill="${col}" rx="6" opacity="${isToday ? 1 : 0.80}"
          style="transition:opacity 0.2s;"
          onmouseover="this.setAttribute('opacity','1')"
          onmouseout="this.setAttribute('opacity','${isToday ? 1 : 0.80}')">
          <title>${f.day} ${f.date}: ${f.count} donors | ${f.temp}°C ${f.precipitation > 0 ? f.precipitation.toFixed(1) + "mm rain" : "dry"}</title>
        </rect>
        <text x="${x}" y="${by - 7}" text-anchor="middle"
          font-size="12" font-weight="700"
          fill="var(--text-primary)"
          font-family="Inter, sans-serif">${f.count}</text>
        ${isToday ? `
          <rect x="${x - barW/2 - 3}" y="${by - 3}"
            width="${barW + 6}" height="${bh + 3}"
            fill="none" stroke="${col}" stroke-width="2"
            rx="7" opacity="0.35"/>` : ""}
      </g>`;

    xLabels += `
      <text x="${x}" y="${H - padB + 18}" text-anchor="middle"
        font-size="12" font-weight="${isToday ? 700 : 500}"
        fill="${isToday ? "var(--accent)" : "var(--text-secondary)"}"
        font-family="Inter, sans-serif">${f.day}</text>
      <text x="${x}" y="${H - padB + 34}" text-anchor="middle"
        font-size="10" fill="var(--text-muted)"
        font-family="Inter, sans-serif">${f.date}</text>`;
  });

  // Y axis label
  const yAxisLabel = `
    <text x="14" y="${padT + chartH/2}" text-anchor="middle"
      font-size="11" fill="var(--text-muted)"
      font-family="Inter, sans-serif"
      transform="rotate(-90, 14, ${padT + chartH/2})">Donors</text>`;

  const svg = `
    <svg viewBox="0 0 ${W} ${H}" class="forecast-svg-wrap"
         style="width:100%; height:auto; display:block;">
      ${yAxisLabel}
      ${gridLines}
      ${yLabels}
      ${bars}
      ${xLabels}
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + chartH}"
        stroke="var(--border)" stroke-width="1.5"/>
      <line x1="${padL}" y1="${padT + chartH}" x2="${W - padR}" y2="${padT + chartH}"
        stroke="var(--border)" stroke-width="1.5"/>
    </svg>`;

  document.getElementById("forecast-chart").innerHTML = svg;

  // --- Table with weather column ---
  const rows = forecasts.map((f, i) => `
    <tr class="${i === 0 ? "today-row" : ""}">
      <td>${i === 0 ? "📍 Today" : f.day}</td>
      <td style="color:var(--text-secondary);">${f.date}</td>
      <td style="color:var(--text-secondary);">
        ↑${f.temp_max}°C  ↓${f.temp_min}°C
        ${f.precipitation > 0
          ? `&nbsp;🌧 ${f.precipitation.toFixed(1)}mm`
          : `&nbsp;☀️ Dry`}
      </td>
      <td style="text-align:right; font-weight:600;">${f.count}</td>
      <td style="text-align:right;">
        <span class="badge badge-${f.level}">
          ${f.level.charAt(0).toUpperCase() + f.level.slice(1)}
        </span>
      </td>
    </tr>`).join("");

  document.getElementById("forecast-table").innerHTML = `
    <table class="forecast-table">
      <thead>
        <tr>
          <th>Day</th>
          <th>Date</th>
          <th>Weather</th>
          <th>Predicted Donors</th>
          <th>Demand Level</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// =============================================
// RENDER INSIGHTS — MAE chart
// =============================================
function renderInsights() {
  const maxMae = Math.max(...MAE_DATA.map(d => d.mae));
  const color  = "#0077BB";

  const rows = MAE_DATA.map((d) => `
    <div class="mae-row">
      <div class="mae-city">${d.city}</div>
      <div class="mae-bar-wrap">
        <div class="mae-bar"
             style="width:${(d.mae / maxMae * 100).toFixed(1)}%;
                    background:${color}; opacity:0.75;">
        </div>
      </div>
      <div class="mae-value">${d.mae}%</div>
    </div>`).join("");

  document.getElementById("mae-chart").innerHTML = `
    ${rows}
    <div class="mae-overall-line">
      Overall model MAE:
      <strong style="color:var(--accent);">5.9%</strong>
      of average daily donor count across all cities
    </div>`;
}

// =============================================
// RENDER METRICS
// =============================================
function renderMetrics() {
  // Model config table
  document.getElementById("model-config-table").innerHTML =
    MODEL_CONFIG.map(([k, v]) => `
      <div class="config-row">
        <span class="config-key">${k}</span>
        <span class="config-val">${v}</span>
      </div>`).join("");

  // Features list
  document.getElementById("features-list").innerHTML = `
    <div class="features-grid">
      ${FEATURES.map(f => `
        <div class="feature-tag">
          <span class="feature-dot" style="background:${f.color};"></span>
          ${f.name}
        </div>`).join("")}
    </div>
    <div style="margin-top:14px; display:flex; gap:16px; flex-wrap:wrap;">
      <span style="font-size:0.75rem; color:var(--text-muted);
                   display:flex; align-items:center; gap:6px;">
        <span style="width:8px; height:8px; border-radius:50%;
                     background:#0077BB; display:inline-block;"></span>
        Weather
      </span>
      <span style="font-size:0.75rem; color:var(--text-muted);
                   display:flex; align-items:center; gap:6px;">
        <span style="width:8px; height:8px; border-radius:50%;
                     background:#EE7733; display:inline-block;"></span>
        Calendar
      </span>
      <span style="font-size:0.75rem; color:var(--text-muted);
                   display:flex; align-items:center; gap:6px;">
        <span style="width:8px; height:8px; border-radius:50%;
                     background:#009988; display:inline-block;"></span>
        Lag & Rolling
      </span>
    </div>`;
}

// =============================================
// INIT
// =============================================
updateCityStats(selectedCity);
buildDropdown();
loadForecast();