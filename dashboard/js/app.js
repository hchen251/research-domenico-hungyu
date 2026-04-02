// ==================== Configuration ====================
const CONFIG = {
    horizons: [1, 4, 12, 24, 60],
    groups: {
        1: 'Output and Income',
        2: 'Labor Market',
        3: 'Housing',
        4: 'Consumption, Orders & Inventories',
        5: 'Money and Credit',
        6: 'Interest and Exchange Rates',
        7: 'Prices',
        8: 'Stock Market'
    },
    seriesGroups: {
        // Group 1: Output and Income
        'RPI': 1, 'W875RX1': 1, 'DPCERA3M086SBEA': 1, 'CMRMTSPLx': 1, 'RETAILx': 1,
        'INDPRO': 1, 'IPFPNSS': 1, 'IPFINAL': 1, 'IPCONGD': 1, 'IPDCONGD': 1,
        'IPNCONGD': 1, 'IPBUSEQ': 1, 'IPMAT': 1, 'IPDMAT': 1, 'IPNMAT': 1,
        'IPMANSICS': 1, 'IPB51222S': 1, 'IPFUELS': 1, 'CUMFNS': 1,
        // Group 2: Labor Market
        'HWI': 2, 'HWIURATIO': 2, 'CLF16OV': 2, 'CE16OV': 2, 'UNRATE': 2,
        'UEMPMEAN': 2, 'UEMPLT5': 2, 'UEMP5TO14': 2, 'UEMP15OV': 2, 'UEMP15T26': 2,
        'UEMP27OV': 2, 'CLAIMSx': 2, 'PAYEMS': 2, 'USGOOD': 2, 'CES1021000001': 2,
        'USCONS': 2, 'MANEMP': 2, 'DMANEMP': 2, 'NDMANEMP': 2, 'SRVPRD': 2,
        'USTPU': 2, 'USWTRADE': 2, 'USTRADE': 2, 'USFIRE': 2, 'USGOVT': 2,
        'CES0600000007': 2, 'AWOTMAN': 2, 'AWHMAN': 2,
        'CES0600000008': 2, 'CES2000000008': 2, 'CES3000000008': 2,
        // Group 3: Housing
        'HOUST': 3, 'HOUSTNE': 3, 'HOUSTMW': 3, 'HOUSTS': 3, 'HOUSTW': 3,
        'PERMIT': 3, 'PERMITNE': 3, 'PERMITMW': 3, 'PERMITS': 3, 'PERMITW': 3,
        // Group 4: Consumption, Orders & Inventories
        'ACOGNO': 4, 'AMDMNOx': 4, 'ANDENOx': 4, 'AMDMUOx': 4, 'BUSINVx': 4,
        'ISRATIOx': 4, 'UMCSENTx': 4,
        // Group 5: Money and Credit
        'M1SL': 5, 'M2SL': 5, 'M2REAL': 5, 'BOGMBASE': 5, 'TOTRESNS': 5,
        'NONBORRES': 5, 'BUSLOANS': 5, 'REALLN': 5, 'NONREVSL': 5, 'CONSPI': 5,
        'DTCOLNVHFNM': 5, 'DTCTHFNM': 5, 'INVEST': 5,
        // Group 6: Interest and Exchange Rates
        'FEDFUNDS': 6, 'CP3Mx': 6, 'TB3MS': 6, 'TB6MS': 6, 'GS1': 6, 'GS5': 6,
        'GS10': 6, 'AAA': 6, 'BAA': 6, 'COMPAPFFx': 6, 'TB3SMFFM': 6, 'TB6SMFFM': 6,
        'T1YFFM': 6, 'T5YFFM': 6, 'T10YFFM': 6, 'AAAFFM': 6, 'BAAFFM': 6,
        'TWEXAFEGSMTHx': 6, 'EXSZUSx': 6, 'EXJPUSx': 6, 'EXUSUKx': 6, 'EXCAUSx': 6,
        // Group 7: Prices
        'WPSFD49207': 7, 'WPSFD49502': 7, 'WPSID61': 7, 'WPSID62': 7, 'OILPRICEx': 7,
        'PPICMM': 7, 'CPIAUCSL': 7, 'CPIAPPSL': 7, 'CPITRNSL': 7, 'CPIMEDSL': 7,
        'CUSR0000SAC': 7, 'CUSR0000SAD': 7, 'CUSR0000SAS': 7, 'CPIULFSL': 7,
        'CUSR0000SA0L2': 7, 'CUSR0000SA0L5': 7, 'PCEPI': 7,
        'DDURRG3M086SBEA': 7, 'DNDGRG3M086SBEA': 7, 'DSERRG3M086SBEA': 7,
        // Group 8: Stock Market
        'S&P 500': 8, 'S&P div yield': 8, 'S&P PE ratio': 8, 'VIXCLSx': 8,
        'S_P500': 8, 'S_PDivYield': 8, 'S_PPERatio': 8
    },
    historyMonths: 24,
    actualColor: '#2563eb',
    forecastColor: '#dc2626',
    bayesianColor: '#7c3aed',
    ciColor: 'rgba(124, 58, 237, 0.12)',
    dataPath: 'data/'
};

// Mapping from Bayesian column names (underscores) to DFM column names (special chars)
const BAYESIAN_COL_MAP = {
    'S_P500': 'S&P 500',
    'S_PDivYield': 'S&P div yield',
    'S_PPERatio': 'S&P PE ratio'
};

// Reverse map: DFM name → Bayesian name
const DFM_TO_BAYESIAN_MAP = {};
for (const [bayKey, dfmKey] of Object.entries(BAYESIAN_COL_MAP)) {
    DFM_TO_BAYESIAN_MAP[dfmKey] = bayKey;
}

// ==================== State ====================
let state = {
    seriesList: [],
    actualData: [],
    forecastData: [],           // DFM forecast (horizon-specific)
    bayesianData: null,         // { median: [], p5: [], p95: [] } — per horizon, like DFM
    selectedHorizon: 12,
    selectedGroups: new Set(),
    selectedSeries: new Set(),
    sortColumn: null,
    sortDirection: 'asc',
    showDFM: true,
    showBayesian: true,
    showCI: true
};

// ==================== DOM Elements ====================
const elements = {
    horizon: document.getElementById('horizon'),
    themeToggle: document.getElementById('theme-toggle'),
    groupCheckboxes: document.getElementById('group-checkboxes'),
    seriesCheckboxes: document.getElementById('series-checkboxes'),
    seriesSearch: document.getElementById('series-search'),
    selectAllGroups: document.getElementById('select-all-groups'),
    clearAllGroups: document.getElementById('clear-all-groups'),
    selectAllSeries: document.getElementById('select-all-series'),
    clearAllSeries: document.getElementById('clear-all-series'),
    selectedCount: document.getElementById('selected-count'),
    metricsBody: document.getElementById('metrics-body'),
    metricsTable: document.getElementById('metrics-table'),
    chartsContainer: document.getElementById('charts-container'),
    loading: document.getElementById('loading'),
    toggleDFM: document.getElementById('toggle-dfm'),
    toggleBayesian: document.getElementById('toggle-bayesian'),
    toggleCI: document.getElementById('toggle-ci')
};

// ==================== Chart Instances ====================
let chartInstances = {};

// ==================== Utility Functions ====================
function showLoading() {
    elements.loading.classList.add('active');
}

function hideLoading() {
    elements.loading.classList.remove('active');
}

function normalizeDate(date) {
    const d = new Date(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    return year + '-' + month;
}

function parseDate(dateStr) {
    if (!dateStr) return null;
    const formats = [
        /^(\d{1,2})\/(\d{1,2})\/(\d{4})$/,
        /^(\d{4})-(\d{2})-(\d{2})$/,
        /^(\d{4})\/(\d{2})\/(\d{2})$/
    ];
    for (const format of formats) {
        const match = dateStr.match(format);
        if (match) {
            if (format === formats[0]) {
                return new Date(parseInt(match[3]), parseInt(match[1]) - 1, parseInt(match[2]));
            } else {
                return new Date(parseInt(match[1]), parseInt(match[2]) - 1, parseInt(match[3]));
            }
        }
    }
    const d = new Date(dateStr);
    return isNaN(d.getTime()) ? null : d;
}

function getHorizonLabel(horizon) {
    if (horizon === 1) return '1 Month';
    return horizon + ' Months';
}

function getSeriesGroup(seriesName) {
    return CONFIG.seriesGroups[seriesName] || 1;
}

// Resolve the column key to use when looking up a series in Bayesian data
// Bayesian files use underscores for S&P columns
function bayesianColKey(seriesName) {
    return DFM_TO_BAYESIAN_MAP[seriesName] || seriesName;
}

// ==================== Data Loading ====================
async function loadCSV(path) {
    return new Promise(function(resolve, reject) {
        Papa.parse(path, {
            download: true,
            header: true,
            skipEmptyLines: true,
            dynamicTyping: false,
            complete: function(results) {
                console.log('Loaded ' + path + ': ' + results.data.length + ' rows');
                resolve(results.data);
            },
            error: function(error) {
                console.error('Error loading ' + path + ':', error);
                reject(error);
            }
        });
    });
}

function parseForecastRows(data) {
    const rows = [];
    for (const row of data) {
        const dateStr = row.sasdate;
        if (dateStr && (dateStr.toLowerCase().includes('transform') || dateStr.toLowerCase().includes(':'))) {
            continue;
        }
        const date = parseDate(dateStr);
        if (!date) continue;
        const parsedRow = { date: date, dateKey: normalizeDate(date) };
        for (const key of Object.keys(row)) {
            if (key !== 'sasdate' && key !== '') {
                const numVal = parseFloat(row[key]);
                parsedRow[key] = (row[key] === '' || row[key] === null || isNaN(numVal)) ? null : numVal;
            }
        }
        rows.push(parsedRow);
    }
    rows.sort(function(a, b) { return a.date - b.date; });
    return rows;
}

async function loadActualData() {
    const data = await loadCSV(CONFIG.dataPath + 'filled_2026-02-MD.csv');
    const actualData = [];
    const seriesSet = new Set();
    for (let i = 0; i < data.length; i++) {
        const row = data[i];
        const dateStr = row.sasdate;
        if (dateStr && (dateStr.toLowerCase().includes('transform') || dateStr.toLowerCase().includes(':'))) {
            continue;
        }
        const date = parseDate(dateStr);
        if (!date) continue;
        const parsedRow = { date: date, dateKey: normalizeDate(date) };
        for (const key of Object.keys(row)) {
            if (key !== 'sasdate' && key !== '') {
                const numVal = parseFloat(row[key]);
                parsedRow[key] = (row[key] === '' || row[key] === null || isNaN(numVal)) ? null : numVal;
                seriesSet.add(key);
            }
        }
        actualData.push(parsedRow);
    }
    actualData.sort(function(a, b) { return a.date - b.date; });
    state.seriesList = Array.from(seriesSet).map(function(name) {
        return { fred: name, group: getSeriesGroup(name), description: name };
    }).sort(function(a, b) { return a.fred.localeCompare(b.fred); });
    console.log('Actual data loaded: ' + actualData.length + ' rows, ' + state.seriesList.length + ' series');
    return actualData;
}

async function loadForecastData(horizon) {
    const filename = 'dfm_' + horizon + 'm.csv';
    try {
        const data = await loadCSV(CONFIG.dataPath + filename);
        const rows = parseForecastRows(data);
        console.log('DFM forecast ' + filename + ' loaded: ' + rows.length + ' rows');
        return rows;
    } catch (error) {
        console.warn('Could not load ' + filename + ':', error);
        return [];
    }
}

// Bayesian data is per-horizon, same structure as DFM.
// Files: bay_{h}m.csv, bay_{h}m_p5.csv, bay_{h}m_p95.csv
async function loadBayesianData(horizon) {
    const base = 'bay_' + horizon + 'm';
    try {
        const [medianRaw, p5Raw, p95Raw] = await Promise.all([
            loadCSV(CONFIG.dataPath + base + '.csv'),
            loadCSV(CONFIG.dataPath + base + '_p5.csv'),
            loadCSV(CONFIG.dataPath + base + '_p95.csv')
        ]);
        const result = {
            median: parseForecastRows(medianRaw),
            p5: parseForecastRows(p5Raw),
            p95: parseForecastRows(p95Raw)
        };
        console.log('Bayesian data loaded: ' + result.median.length + ' median rows');
        return result;
    } catch (error) {
        console.warn('Could not load Bayesian data:', error);
        return null;
    }
}

async function loadAllData() {
    showLoading();
    try {
        [state.actualData, state.forecastData, state.bayesianData] = await Promise.all([
            loadActualData(),
            loadForecastData(state.selectedHorizon),
            loadBayesianData(state.selectedHorizon)
        ]);
        hideLoading();
    } catch (error) {
        console.error('Error loading data:', error);
        hideLoading();
        alert('Error loading data. Check console for details.');
    }
}

async function loadForecastForHorizon(horizon) {
    [state.forecastData, state.bayesianData] = await Promise.all([
        loadForecastData(horizon),
        loadBayesianData(horizon)
    ]);
    console.log('DFM + Bayesian reloaded for horizon: ' + horizon);
}

// ==================== UI Rendering ====================
function renderGroupCheckboxes() {
    elements.groupCheckboxes.innerHTML = '';
    for (const groupId of Object.keys(CONFIG.groups)) {
        const groupName = CONFIG.groups[groupId];
        const div = document.createElement('div');
        div.className = 'checkbox-item';
        const seriesCount = state.seriesList.filter(function(s) {
            return s.group === parseInt(groupId);
        }).length;
        div.innerHTML =
            '<input type="checkbox" id="group-' + groupId + '" value="' + groupId + '">' +
            '<label for="group-' + groupId + '">' + groupId + '. ' + groupName + '</label>' +
            '<span class="group-badge">' + seriesCount + '</span>';
        const checkbox = div.querySelector('input');
        checkbox.addEventListener('change', function() {
            handleGroupChange(parseInt(groupId), checkbox.checked);
        });
        elements.groupCheckboxes.appendChild(div);
    }
}

function renderSeriesCheckboxes() {
    elements.seriesCheckboxes.innerHTML = '';
    const searchTerm = elements.seriesSearch.value.toLowerCase();
    let allSeries = state.seriesList.filter(function(s) {
        return s.fred.toLowerCase().includes(searchTerm) ||
               (s.description && s.description.toLowerCase().includes(searchTerm));
    });
    if (allSeries.length === 0) {
        elements.seriesCheckboxes.innerHTML = '<div class="empty-state"><p>No series found</p></div>';
        return;
    }
    const selectedGroupSeries = allSeries.filter(function(s) {
        return state.selectedGroups.has(s.group);
    });
    const unselectedGroupSeries = allSeries.filter(function(s) {
        return !state.selectedGroups.has(s.group);
    });
    const selectedByGroup = {};
    for (const series of selectedGroupSeries) {
        if (!selectedByGroup[series.group]) selectedByGroup[series.group] = [];
        selectedByGroup[series.group].push(series);
    }
    for (const groupId of Object.keys(selectedByGroup)) {
        selectedByGroup[groupId].sort(function(a, b) { return a.fred.localeCompare(b.fred); });
    }
    unselectedGroupSeries.sort(function(a, b) {
        return a.fred.toLowerCase().localeCompare(b.fred.toLowerCase());
    });
    const sortedGroupIds = Object.keys(selectedByGroup).sort(function(a, b) {
        return parseInt(a) - parseInt(b);
    });
    for (const groupId of sortedGroupIds) {
        const seriesList = selectedByGroup[groupId];
        const groupHeader = document.createElement('div');
        groupHeader.className = 'series-group-header selected-group';
        groupHeader.innerHTML = '<strong>' + CONFIG.groups[groupId] + '</strong>';
        groupHeader.style.cssText = 'padding: 0.5rem; margin-top: 0.5rem; font-size: 0.8rem; color: var(--accent-color); background-color: var(--bg-tertiary); border-radius: 4px;';
        elements.seriesCheckboxes.appendChild(groupHeader);
        for (const series of seriesList) {
            renderSeriesCheckbox(series, true);
        }
    }
    if (unselectedGroupSeries.length > 0 && state.selectedGroups.size > 0) {
        const otherHeader = document.createElement('div');
        otherHeader.className = 'series-group-header';
        otherHeader.innerHTML = '<strong>Other Series</strong>';
        otherHeader.style.cssText = 'padding: 0.5rem; margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-secondary);';
        elements.seriesCheckboxes.appendChild(otherHeader);
    }
    for (const series of unselectedGroupSeries) {
        renderSeriesCheckbox(series, false);
    }
}

function renderSeriesCheckbox(series, isInSelectedGroup) {
    const div = document.createElement('div');
    div.className = 'checkbox-item';
    if (isInSelectedGroup) div.classList.add('in-selected-group');
    const safeId = series.fred.replace(/[^a-zA-Z0-9]/g, '_');
    const isChecked = state.selectedSeries.has(series.fred) ? 'checked' : '';
    div.innerHTML =
        '<input type="checkbox" id="series-' + safeId + '" value="' + series.fred + '" ' + isChecked + '>' +
        '<label for="series-' + safeId + '" title="' + (series.description || '') + '">' + series.fred + '</label>' +
        '<span class="group-badge">' + series.group + '</span>';
    const checkbox = div.querySelector('input');
    checkbox.addEventListener('change', function() {
        handleSeriesChange(series.fred, checkbox.checked);
    });
    elements.seriesCheckboxes.appendChild(div);
}

function updateSelectedCount() {
    elements.selectedCount.textContent = state.selectedSeries.size + ' series selected';
}

function updateToggleButtons() {
    updateToggleBtn(elements.toggleDFM, state.showDFM, CONFIG.forecastColor);
    updateToggleBtn(elements.toggleBayesian, state.showBayesian, CONFIG.bayesianColor);
    if (elements.toggleCI) {
        elements.toggleCI.classList.toggle('toggle-active', state.showCI);
        elements.toggleCI.style.opacity = state.showBayesian ? '1' : '0.4';
        elements.toggleCI.disabled = !state.showBayesian;
    }
}

function updateToggleBtn(btn, active, color) {
    if (!btn) return;
    btn.classList.toggle('toggle-active', active);
    btn.style.borderColor = active ? color : 'var(--border-color)';
    btn.style.color = active ? color : 'var(--text-secondary)';
    btn.style.backgroundColor = active ? (color + '18') : 'var(--bg-tertiary)';
}

// ==================== Event Handlers ====================
function handleGroupChange(groupId, checked) {
    if (checked) {
        state.selectedGroups.add(groupId);
    } else {
        state.selectedGroups.delete(groupId);
        const groupSeries = state.seriesList.filter(function(s) { return s.group === groupId; });
        for (const series of groupSeries) state.selectedSeries.delete(series.fred);
    }
    renderSeriesCheckboxes();
    updateSelectedCount();
    updateDashboard();
}

function handleSeriesChange(seriesName, checked) {
    if (checked) {
        state.selectedSeries.add(seriesName);
    } else {
        state.selectedSeries.delete(seriesName);
    }
    updateSelectedCount();
    updateDashboard();
}

async function handleHorizonChange(horizon) {
    showLoading();
    state.selectedHorizon = horizon;
    await loadForecastForHorizon(horizon);
    hideLoading();
    updateDashboard();
}

function handleThemeToggle() {
    document.body.classList.toggle('dark-theme');
    document.body.classList.toggle('light-theme');
    const isDark = document.body.classList.contains('dark-theme');
    elements.themeToggle.textContent = isDark ? 'Light Mode' : 'Dark Mode';
    updateCharts();
}

function handleSelectAllGroups() {
    for (let i = 1; i <= 8; i++) {
        state.selectedGroups.add(i);
        const checkbox = document.getElementById('group-' + i);
        if (checkbox) checkbox.checked = true;
    }
    renderSeriesCheckboxes();
    updateSelectedCount();
    updateDashboard();
}

function handleClearAllGroups() {
    state.selectedGroups.clear();
    state.selectedSeries.clear();
    for (let i = 1; i <= 8; i++) {
        const checkbox = document.getElementById('group-' + i);
        if (checkbox) checkbox.checked = false;
    }
    renderSeriesCheckboxes();
    updateSelectedCount();
    updateDashboard();
}

function handleSelectAllSeries() {
    const visibleSeries = state.seriesList.filter(function(s) {
        return state.selectedGroups.has(s.group);
    });
    for (const series of visibleSeries) state.selectedSeries.add(series.fred);
    renderSeriesCheckboxes();
    updateSelectedCount();
    updateDashboard();
}

function handleClearAllSeries() {
    state.selectedSeries.clear();
    renderSeriesCheckboxes();
    updateSelectedCount();
    updateDashboard();
}

// ==================== Metrics Calculation ====================
function getSeriesData(seriesName) {
    const seriesInfo = state.seriesList.find(function(s) { return s.fred === seriesName; });
    if (!seriesInfo) return { actual: [], forecast: [], bayesian: null, seriesInfo: null };

    const actual = [];
    for (const row of state.actualData) {
        const value = row[seriesName];
        if (value !== null && value !== undefined && !isNaN(value)) {
            actual.push({ date: row.date, dateKey: row.dateKey, value: value });
        }
    }

    const forecast = [];
    for (const row of state.forecastData) {
        const value = row[seriesName];
        if (value !== null && value !== undefined && !isNaN(value)) {
            forecast.push({ date: row.date, dateKey: row.dateKey, value: value });
        }
    }

    // Bayesian: look up using the mapped column name
    let bayesian = null;
    if (state.bayesianData) {
        const bKey = bayesianColKey(seriesName);
        const median = [];
        const p5 = [];
        const p95 = [];
        for (const row of state.bayesianData.median) {
            const value = row[bKey];
            if (value !== null && value !== undefined && !isNaN(value)) {
                median.push({ date: row.date, dateKey: row.dateKey, value: value });
            }
        }
        for (const row of state.bayesianData.p5) {
            const value = row[bKey];
            if (value !== null && value !== undefined && !isNaN(value)) {
                p5.push({ date: row.date, dateKey: row.dateKey, value: value });
            }
        }
        for (const row of state.bayesianData.p95) {
            const value = row[bKey];
            if (value !== null && value !== undefined && !isNaN(value)) {
                p95.push({ date: row.date, dateKey: row.dateKey, value: value });
            }
        }
        if (median.length > 0) {
            bayesian = { median: median, p5: p5, p95: p95 };
        }
    }

    return { actual: actual, forecast: forecast, bayesian: bayesian, seriesInfo: seriesInfo };
}

function calculateMetrics(actual, forecast) {
    if (!actual || !forecast || actual.length === 0 || forecast.length === 0) return null;
    const forecastMap = new Map();
    for (const f of forecast) forecastMap.set(f.dateKey, f.value);
    const pairs = [];
    for (const a of actual) {
        if (forecastMap.has(a.dateKey)) {
            const fVal = forecastMap.get(a.dateKey);
            if (a.value !== null && fVal !== null && !isNaN(a.value) && !isNaN(fVal)) {
                pairs.push({ actual: a.value, forecast: fVal });
            }
        }
    }
    if (pairs.length === 0) return null;
    const n = pairs.length;
    let sumSqError = 0, sumAbsError = 0, sumAbsPctError = 0, validPctCount = 0;
    let sumActual = 0, sumForecast = 0, sumActualSq = 0, sumForecastSq = 0, sumProduct = 0;
    for (const p of pairs) {
        const error = p.forecast - p.actual;
        sumSqError += error * error;
        sumAbsError += Math.abs(error);
        if (p.actual !== 0) { sumAbsPctError += Math.abs(error / p.actual) * 100; validPctCount++; }
        sumActual += p.actual;
        sumForecast += p.forecast;
        sumActualSq += p.actual * p.actual;
        sumForecastSq += p.forecast * p.forecast;
        sumProduct += p.actual * p.forecast;
    }
    const rmse = Math.sqrt(sumSqError / n);
    const mae = sumAbsError / n;
    const mape = validPctCount > 0 ? sumAbsPctError / validPctCount : null;
    let correlation = null;
    if (n > 1) {
        const numerator = n * sumProduct - sumActual * sumForecast;
        const denomA = Math.sqrt(n * sumActualSq - sumActual * sumActual);
        const denomB = Math.sqrt(n * sumForecastSq - sumForecast * sumForecast);
        if (denomA > 0 && denomB > 0) correlation = numerator / (denomA * denomB);
    }
    return { rmse: rmse, mae: mae, mape: mape, correlation: correlation, n: n };
}

// ==================== Dashboard Updates ====================
function updateDashboard() {
    updateToggleButtons();
    updateMetricsTable();
    updateCharts();
}

function updateMetricsTable() {
    const rows = [];
    for (const seriesName of state.selectedSeries) {
        const data = getSeriesData(seriesName);
        if (!data.seriesInfo) continue;
        const dfmMetrics = state.showDFM ? calculateMetrics(data.actual, data.forecast) : null;
        const bayMetrics = (state.showBayesian && data.bayesian)
            ? calculateMetrics(data.actual, data.bayesian.median)
            : null;
        rows.push({
            series: seriesName,
            group: CONFIG.groups[data.seriesInfo.group],
            groupId: data.seriesInfo.group,
            dfm: dfmMetrics,
            bay: bayMetrics
        });
    }

    if (state.sortColumn) {
        rows.sort(function(a, b) {
            // Sort key supports both dfm.rmse and bay.rmse style keys
            let aVal, bVal;
            if (state.sortColumn === 'series') { aVal = a.series; bVal = b.series; }
            else if (state.sortColumn === 'group') { aVal = a.group; bVal = b.group; }
            else if (state.sortColumn.startsWith('dfm_')) {
                const k = state.sortColumn.replace('dfm_', '');
                aVal = a.dfm ? a.dfm[k] : null;
                bVal = b.dfm ? b.dfm[k] : null;
            } else if (state.sortColumn.startsWith('bay_')) {
                const k = state.sortColumn.replace('bay_', '');
                aVal = a.bay ? a.bay[k] : null;
                bVal = b.bay ? b.bay[k] : null;
            }
            if (typeof aVal === 'string') { aVal = aVal.toLowerCase(); bVal = (bVal || '').toLowerCase(); }
            if (aVal === null || aVal === undefined) return 1;
            if (bVal === null || bVal === undefined) return -1;
            if (state.sortDirection === 'asc') return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
            else return aVal < bVal ? 1 : aVal > bVal ? -1 : 0;
        });
    }

    elements.metricsBody.innerHTML = '';

    if (rows.length === 0) {
        elements.metricsBody.innerHTML =
            '<tr><td colspan="8" style="text-align: center; padding: 2rem; color: var(--text-secondary);">Select series to view metrics</td></tr>';
        return;
    }

    for (const row of rows) {
        const tr = document.createElement('tr');
        const fmt = function(v, decimals) { return v !== null && v !== undefined ? v.toFixed(decimals) : 'N/A'; };
        tr.innerHTML =
            '<td><strong>' + row.series + '</strong></td>' +
            '<td>' + row.group + '</td>' +
            // DFM columns
            '<td style="border-left: 2px solid ' + CONFIG.forecastColor + '22;">' + (row.dfm ? fmt(row.dfm.rmse, 4) : '—') + '</td>' +
            '<td>' + (row.dfm ? fmt(row.dfm.mape, 2) + '%' : '—') + '</td>' +
            '<td>' + (row.dfm ? fmt(row.dfm.correlation, 4) : '—') + '</td>' +
            // Bayesian columns
            '<td style="border-left: 2px solid ' + CONFIG.bayesianColor + '44;">' + (row.bay ? fmt(row.bay.rmse, 4) : '—') + '</td>' +
            '<td>' + (row.bay ? fmt(row.bay.mape, 2) + '%' : '—') + '</td>' +
            '<td>' + (row.bay ? fmt(row.bay.correlation, 4) : '—') + '</td>';
        elements.metricsBody.appendChild(tr);
    }
}

function updateCharts() {
    for (const chartId of Object.keys(chartInstances)) {
        chartInstances[chartId].destroy();
    }
    chartInstances = {};
    elements.chartsContainer.innerHTML = '';

    if (state.selectedSeries.size === 0) {
        elements.chartsContainer.innerHTML =
            '<div class="empty-state"><h3>No Series Selected</h3><p>Select groups and series from the sidebar to view charts</p></div>';
        return;
    }

    for (const seriesName of state.selectedSeries) {
        createChart(seriesName);
    }
}

function createChart(seriesName) {
    const data = getSeriesData(seriesName);
    if (!data.seriesInfo) return;
    if (data.actual.length === 0 && data.forecast.length === 0 && !data.bayesian) return;

    // Determine history window start
    let forecastStart = null;
    const allForecasts = [
        ...(state.showDFM ? data.forecast : []),
        ...(state.showBayesian && data.bayesian ? data.bayesian.median : [])
    ];
    if (allForecasts.length > 0) {
        const times = allForecasts.map(function(f) { return f.date.getTime(); });
        forecastStart = new Date(Math.min.apply(null, times));
    }

    let historyStart;
    if (forecastStart) {
        historyStart = new Date(forecastStart);
        historyStart.setMonth(historyStart.getMonth() - CONFIG.historyMonths);
    } else {
        const actualTimes = data.actual.map(function(a) { return a.date.getTime(); });
        historyStart = new Date(Math.min.apply(null, actualTimes));
    }

    const actualFiltered = data.actual.filter(function(a) { return a.date >= historyStart; });

    // Build unified date axis
    const allDateKeys = new Set();
    actualFiltered.forEach(function(a) { allDateKeys.add(a.dateKey); });
    if (state.showDFM) data.forecast.forEach(function(f) { allDateKeys.add(f.dateKey); });
    if (state.showBayesian && data.bayesian) {
        data.bayesian.median.forEach(function(f) { allDateKeys.add(f.dateKey); });
    }
    const sortedDateKeys = Array.from(allDateKeys).sort();

    // Build value arrays
    const actualMap = new Map(actualFiltered.map(function(a) { return [a.dateKey, a.value]; }));
    const forecastMap = new Map(data.forecast.map(function(f) { return [f.dateKey, f.value]; }));

    const actualValues = sortedDateKeys.map(function(k) { return actualMap.has(k) ? actualMap.get(k) : null; });
    const forecastValues = sortedDateKeys.map(function(k) { return forecastMap.has(k) ? forecastMap.get(k) : null; });

    let bayMedianValues = [], bayP5Values = [], bayP95Values = [];
    if (data.bayesian) {
        const bayMedianMap = new Map(data.bayesian.median.map(function(f) { return [f.dateKey, f.value]; }));
        const bayP5Map = new Map(data.bayesian.p5.map(function(f) { return [f.dateKey, f.value]; }));
        const bayP95Map = new Map(data.bayesian.p95.map(function(f) { return [f.dateKey, f.value]; }));
        bayMedianValues = sortedDateKeys.map(function(k) { return bayMedianMap.has(k) ? bayMedianMap.get(k) : null; });
        bayP5Values = sortedDateKeys.map(function(k) { return bayP5Map.has(k) ? bayP5Map.get(k) : null; });
        bayP95Values = sortedDateKeys.map(function(k) { return bayP95Map.has(k) ? bayP95Map.get(k) : null; });
    }

    const labels = sortedDateKeys.map(function(key) {
        const parts = key.split('-');
        return parts[1] + '/' + parts[0];
    });

    // Metrics for chart header
    const dfmMetrics = state.showDFM ? calculateMetrics(data.actual, data.forecast) : null;
    const bayMetrics = (state.showBayesian && data.bayesian)
        ? calculateMetrics(data.actual, data.bayesian.median)
        : null;

    // Build card HTML
    const chartId = 'chart-' + seriesName.replace(/[^a-zA-Z0-9]/g, '_');

    let metricsHtml = '<span>Horizon: ' + getHorizonLabel(state.selectedHorizon) + '</span>';
    if (dfmMetrics) {
        metricsHtml +=
            '<span style="color:' + CONFIG.forecastColor + '">DFM RMSE: ' + dfmMetrics.rmse.toFixed(2) + '</span>' +
            '<span style="color:' + CONFIG.forecastColor + '">MAPE: ' + (dfmMetrics.mape !== null ? dfmMetrics.mape.toFixed(1) + '%' : 'N/A') + '</span>';
    }
    if (bayMetrics) {
        metricsHtml +=
            '<span style="color:' + CONFIG.bayesianColor + '">Bayes RMSE: ' + bayMetrics.rmse.toFixed(2) + '</span>' +
            '<span style="color:' + CONFIG.bayesianColor + '">MAPE: ' + (bayMetrics.mape !== null ? bayMetrics.mape.toFixed(1) + '%' : 'N/A') + '</span>';
    }

    // Build legend items
    let legendHtml =
        '<div class="legend-item"><div class="legend-line actual"></div><span>Actual</span></div>';
    if (state.showDFM) {
        legendHtml +=
            '<div class="legend-item"><div class="legend-line forecast"></div><span>DFM Forecast</span></div>';
    }
    if (state.showBayesian && data.bayesian) {
        legendHtml +=
            '<div class="legend-item"><div class="legend-line bayesian"></div><span>Bayesian Forecast</span></div>';
        if (state.showCI) {
            legendHtml +=
                '<div class="legend-item"><div class="legend-ci-band"></div><span>90% CI</span></div>';
        }
    }

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML =
        '<div class="chart-header">' +
            '<h3>' + seriesName + ' <span style="font-weight: normal; color: var(--text-secondary); font-size: 0.85rem;">— ' + (data.seriesInfo.description || '') + '</span></h3>' +
            '<div class="chart-metrics">' + metricsHtml + '</div>' +
        '</div>' +
        '<div class="chart-wrapper">' +
            '<canvas id="' + chartId + '"></canvas>' +
        '</div>' +
        '<div class="chart-legend">' + legendHtml + '</div>' +
        '<div class="chart-controls">' +
            '<button onclick="resetZoom(\'' + chartId + '\')">Reset Zoom</button>' +
        '</div>';

    elements.chartsContainer.appendChild(card);

    // Build Chart.js datasets
    const isDark = document.body.classList.contains('dark-theme');
    const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
    const textColor = isDark ? '#e9ecef' : '#212529';

    const datasets = [
        {
            label: 'Actual',
            data: actualValues,
            borderColor: CONFIG.actualColor,
            backgroundColor: CONFIG.actualColor,
            borderWidth: 2,
            pointRadius: 2,
            pointHoverRadius: 5,
            tension: 0.1,
            spanGaps: true,
            order: 1
        }
    ];

    if (state.showDFM) {
        datasets.push({
            label: 'DFM Forecast',
            data: forecastValues,
            borderColor: CONFIG.forecastColor,
            backgroundColor: CONFIG.forecastColor,
            borderWidth: 2,
            borderDash: [5, 5],
            pointRadius: 2,
            pointHoverRadius: 5,
            tension: 0.1,
            spanGaps: true,
            order: 2
        });
    }

    if (state.showBayesian && data.bayesian) {
        // CI upper band (p95) — drawn first so it's behind
        if (state.showCI) {
            datasets.push({
                label: 'CI Upper (P95)',
                data: bayP95Values,
                borderColor: 'transparent',
                backgroundColor: CONFIG.ciColor,
                borderWidth: 0,
                pointRadius: 0,
                fill: '+1',   // fill down to next dataset (p5)
                tension: 0.1,
                spanGaps: true,
                order: 5
            });
            datasets.push({
                label: 'CI Lower (P5)',
                data: bayP5Values,
                borderColor: 'transparent',
                backgroundColor: CONFIG.ciColor,
                borderWidth: 0,
                pointRadius: 0,
                fill: false,
                tension: 0.1,
                spanGaps: true,
                order: 5
            });
        }
        // Median line on top
        datasets.push({
            label: 'Bayesian Forecast',
            data: bayMedianValues,
            borderColor: CONFIG.bayesianColor,
            backgroundColor: CONFIG.bayesianColor,
            borderWidth: 2,
            borderDash: [4, 3],
            pointRadius: 2,
            pointHoverRadius: 5,
            tension: 0.1,
            spanGaps: true,
            order: 3
        });
    }

    const ctx = document.getElementById(chartId).getContext('2d');
    chartInstances[chartId] = new Chart(ctx, {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            if (value === null) return null;
                            const label = context.dataset.label;
                            if (label === 'CI Upper (P95)' || label === 'CI Lower (P5)') return null;
                            return label + ': ' + value.toFixed(4);
                        }
                    }
                },
                zoom: {
                    pan: { enabled: true, mode: 'x' },
                    zoom: {
                        wheel: { enabled: true },
                        pinch: { enabled: true },
                        mode: 'x'
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: gridColor },
                    ticks: { color: textColor, maxRotation: 45, minRotation: 0, autoSkip: true, maxTicksLimit: 12 }
                },
                y: {
                    grid: { color: gridColor },
                    ticks: { color: textColor }
                }
            }
        }
    });
}

function resetZoom(chartId) {
    if (chartInstances[chartId]) chartInstances[chartId].resetZoom();
}

// ==================== Table Sorting ====================
function setupTableSorting() {
    const headers = elements.metricsTable.querySelectorAll('th[data-sort]');
    headers.forEach(function(header) {
        header.addEventListener('click', function() {
            const column = header.getAttribute('data-sort');
            if (state.sortColumn === column) {
                state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                state.sortColumn = column;
                state.sortDirection = 'asc';
            }
            headers.forEach(function(h) { h.classList.remove('sorted-asc', 'sorted-desc'); });
            header.classList.add('sorted-' + state.sortDirection);
            updateMetricsTable();
        });
    });
}

// ==================== Event Listeners ====================
function setupEventListeners() {
    elements.horizon.addEventListener('change', function(e) {
        handleHorizonChange(parseInt(e.target.value));
    });

    elements.themeToggle.addEventListener('click', handleThemeToggle);

    elements.selectAllGroups.addEventListener('click', handleSelectAllGroups);
    elements.clearAllGroups.addEventListener('click', handleClearAllGroups);
    elements.selectAllSeries.addEventListener('click', handleSelectAllSeries);
    elements.clearAllSeries.addEventListener('click', handleClearAllSeries);

    elements.seriesSearch.addEventListener('input', function() { renderSeriesCheckboxes(); });

    elements.toggleDFM.addEventListener('click', function() {
        state.showDFM = !state.showDFM;
        updateDashboard();
    });

    elements.toggleBayesian.addEventListener('click', function() {
        state.showBayesian = !state.showBayesian;
        updateDashboard();
    });

    elements.toggleCI.addEventListener('click', function() {
        if (!state.showBayesian) return;
        state.showCI = !state.showCI;
        updateDashboard();
    });

    setupTableSorting();
}

// ==================== Initialize ====================
async function init() {
    console.log('Initializing DFM + Bayesian dashboard...');
    setupEventListeners();
    updateToggleButtons();
    await loadAllData();
    renderGroupCheckboxes();
    renderSeriesCheckboxes();
    updateDashboard();
    console.log('Dashboard initialized');
}

init();
