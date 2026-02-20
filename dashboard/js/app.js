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
    // Series to group mapping (based on FRED-MD categories)
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
        'S&P 500': 8, 'S&P div yield': 8, 'S&P PE ratio': 8, 'VIXCLSx': 8
    },
    historyMonths: 24,
    actualColor: '#2563eb',
    forecastColor: '#dc2626',
    dataPath: 'data/'
};

// ==================== State ====================
let state = {
    seriesList: [],
    actualData: [],
    forecastData: [],
    selectedHorizon: 12,
    selectedGroups: new Set(),
    selectedSeries: new Set(),
    sortColumn: null,
    sortDirection: 'asc'
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
    loading: document.getElementById('loading')
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
    
    // Try different date formats
    const formats = [
        /^(\d{1,2})\/(\d{1,2})\/(\d{4})$/,  // m/d/Y or mm/dd/YYYY
        /^(\d{4})-(\d{2})-(\d{2})$/,         // YYYY-MM-DD
        /^(\d{4})\/(\d{2})\/(\d{2})$/        // YYYY/MM/DD
    ];
    
    for (const format of formats) {
        const match = dateStr.match(format);
        if (match) {
            if (format === formats[0]) {
                // m/d/Y format
                return new Date(parseInt(match[3]), parseInt(match[1]) - 1, parseInt(match[2]));
            } else {
                // YYYY-MM-DD or YYYY/MM/DD format
                return new Date(parseInt(match[1]), parseInt(match[2]) - 1, parseInt(match[3]));
            }
        }
    }
    
    // Fallback
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

async function loadActualData() {
    const data = await loadCSV(CONFIG.dataPath + '2025-12-MD.csv');
    
    const actualData = [];
    const seriesSet = new Set();
    
    for (let i = 0; i < data.length; i++) {
        const row = data[i];
        const dateStr = row.sasdate;
        
        // Skip transform row
        if (dateStr && (dateStr.toLowerCase().includes('transform') || dateStr.toLowerCase().includes(':'))) {
            continue;
        }
        
        const date = parseDate(dateStr);
        if (!date) {
            continue;
        }
        
        const parsedRow = { 
            date: date,
            dateKey: normalizeDate(date)
        };
        
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
    
    // Build series list from actual data columns
    state.seriesList = Array.from(seriesSet).map(function(name) {
        return {
            fred: name,
            group: getSeriesGroup(name),
            description: name
        };
    }).sort(function(a, b) {
        return a.fred.localeCompare(b.fred);
    });
    
    console.log('Actual data loaded: ' + actualData.length + ' rows, ' + state.seriesList.length + ' series');
    
    return actualData;
}

async function loadForecastData(horizon) {
    const filename = horizon + 'm.csv';
    try {
        const data = await loadCSV(CONFIG.dataPath + filename);
        
        const forecastData = [];
        
        for (const row of data) {
            const dateStr = row.sasdate;
            
            // Skip any header/transform rows
            if (dateStr && (dateStr.toLowerCase().includes('transform') || dateStr.toLowerCase().includes(':'))) {
                continue;
            }
            
            const date = parseDate(dateStr);
            if (!date) {
                continue;
            }
            
            const parsedRow = { 
                date: date,
                dateKey: normalizeDate(date)
            };
            
            for (const key of Object.keys(row)) {
                if (key !== 'sasdate' && key !== '') {
                    const numVal = parseFloat(row[key]);
                    parsedRow[key] = (row[key] === '' || row[key] === null || isNaN(numVal)) ? null : numVal;
                }
            }
            forecastData.push(parsedRow);
        }
        
        forecastData.sort(function(a, b) { return a.date - b.date; });
        
        console.log('Forecast ' + filename + ' loaded: ' + forecastData.length + ' rows');
        
        return forecastData;
    } catch (error) {
        console.warn('Could not load ' + filename + ':', error);
        return [];
    }
}

async function loadAllData() {
    showLoading();
    
    try {
        state.actualData = await loadActualData();
        state.forecastData = await loadForecastData(state.selectedHorizon);
        hideLoading();
    } catch (error) {
        console.error('Error loading data:', error);
        hideLoading();
        alert('Error loading data. Check console for details.');
    }
}

async function loadForecastForHorizon(horizon) {
    state.forecastData = await loadForecastData(horizon);
    console.log('Forecast data loaded for horizon: ' + horizon);
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
        if (!selectedByGroup[series.group]) {
            selectedByGroup[series.group] = [];
        }
        selectedByGroup[series.group].push(series);
    }
    
    for (const groupId of Object.keys(selectedByGroup)) {
        selectedByGroup[groupId].sort(function(a, b) {
            return a.fred.localeCompare(b.fred);
        });
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
    if (isInSelectedGroup) {
        div.classList.add('in-selected-group');
    }
    
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

// ==================== Event Handlers ====================
function handleGroupChange(groupId, checked) {
    if (checked) {
        state.selectedGroups.add(groupId);
    } else {
        state.selectedGroups.delete(groupId);
        
        const groupSeries = state.seriesList.filter(function(s) { 
            return s.group === groupId; 
        });
        for (const series of groupSeries) {
            state.selectedSeries.delete(series.fred);
        }
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
    for (const series of visibleSeries) {
        state.selectedSeries.add(series.fred);
    }
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
    const seriesInfo = state.seriesList.find(function(s) { 
        return s.fred === seriesName; 
    });
    if (!seriesInfo) {
        return { actual: [], forecast: [], seriesInfo: null };
    }
    
    const actual = [];
    for (const row of state.actualData) {
        const value = row[seriesName];
        if (value !== null && value !== undefined && !isNaN(value)) {
            actual.push({
                date: row.date,
                dateKey: row.dateKey,
                value: value
            });
        }
    }
    
    const forecast = [];
    for (const row of state.forecastData) {
        const value = row[seriesName];
        if (value !== null && value !== undefined && !isNaN(value)) {
            forecast.push({
                date: row.date,
                dateKey: row.dateKey,
                value: value
            });
        }
    }
    
    return { actual: actual, forecast: forecast, seriesInfo: seriesInfo };
}

function calculateMetrics(actual, forecast) {
    if (!actual || !forecast || actual.length === 0 || forecast.length === 0) {
        return null;
    }
    
    const forecastMap = new Map();
    for (const f of forecast) {
        forecastMap.set(f.dateKey, f.value);
    }
    
    const pairs = [];
    for (const a of actual) {
        if (forecastMap.has(a.dateKey)) {
            const fVal = forecastMap.get(a.dateKey);
            if (a.value !== null && fVal !== null && !isNaN(a.value) && !isNaN(fVal)) {
                pairs.push({
                    actual: a.value,
                    forecast: fVal,
                    dateKey: a.dateKey
                });
            }
        }
    }
    
    if (pairs.length === 0) {
        return null;
    }
    
    const n = pairs.length;
    let sumSqError = 0;
    let sumAbsError = 0;
    let sumAbsPctError = 0;
    let validPctCount = 0;
    
    let sumActual = 0;
    let sumForecast = 0;
    let sumActualSq = 0;
    let sumForecastSq = 0;
    let sumProduct = 0;
    
    for (const p of pairs) {
        const error = p.forecast - p.actual;
        sumSqError += error * error;
        sumAbsError += Math.abs(error);
        
        if (p.actual !== 0) {
            sumAbsPctError += Math.abs(error / p.actual) * 100;
            validPctCount++;
        }
        
        sumActual += p.actual;
        sumForecast += p.forecast;
        sumActualSq += p.actual * p.actual;
        sumForecastSq += p.forecast * p.forecast;
        sumProduct += p.actual * p.forecast;
    }
    
    const rmse = Math.sqrt(sumSqError / n);
    const mae = sumAbsError / n;
    const mape = validPctCount > 0 ? sumAbsPctError / validPctCount : null;
    
    // Correlation
    let correlation = null;
    if (n > 1) {
        const numerator = n * sumProduct - sumActual * sumForecast;
        const denomA = Math.sqrt(n * sumActualSq - sumActual * sumActual);
        const denomB = Math.sqrt(n * sumForecastSq - sumForecast * sumForecast);
        if (denomA > 0 && denomB > 0) {
            correlation = numerator / (denomA * denomB);
        }
    }
    
    return { rmse: rmse, mae: mae, mape: mape, correlation: correlation, n: n };
}

// ==================== Dashboard Updates ====================
function updateDashboard() {
    updateMetricsTable();
    updateCharts();
}

function updateMetricsTable() {
    const rows = [];
    
    for (const seriesName of state.selectedSeries) {
        const data = getSeriesData(seriesName);
        
        if (!data.seriesInfo) continue;
        
        const metrics = calculateMetrics(data.actual, data.forecast);
        
        rows.push({
            series: seriesName,
            group: CONFIG.groups[data.seriesInfo.group],
            groupId: data.seriesInfo.group,
            rmse: metrics ? metrics.rmse : null,
            mae: metrics ? metrics.mae : null,
            mape: metrics ? metrics.mape : null,
            correlation: metrics ? metrics.correlation : null,
            n: metrics ? metrics.n : 0
        });
    }
    
    if (state.sortColumn) {
        rows.sort(function(a, b) {
            let aVal = a[state.sortColumn];
            let bVal = b[state.sortColumn];
            
            if (typeof aVal === 'string') {
                aVal = aVal.toLowerCase();
                bVal = (bVal || '').toLowerCase();
            }
            
            if (aVal === null || aVal === undefined) return 1;
            if (bVal === null || bVal === undefined) return -1;
            
            if (state.sortDirection === 'asc') {
                return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
            } else {
                return aVal < bVal ? 1 : aVal > bVal ? -1 : 0;
            }
        });
    }
    
    elements.metricsBody.innerHTML = '';
    
    if (rows.length === 0) {
        elements.metricsBody.innerHTML = 
            '<tr><td colspan="6" style="text-align: center; padding: 2rem; color: var(--text-secondary);">Select series to view metrics</td></tr>';
        return;
    }
    
    for (const row of rows) {
        const tr = document.createElement('tr');
        tr.innerHTML = 
            '<td><strong>' + row.series + '</strong></td>' +
            '<td>' + row.group + '</td>' +
            '<td>' + (row.rmse !== null ? row.rmse.toFixed(4) : 'N/A') + '</td>' +
            '<td>' + (row.mae !== null ? row.mae.toFixed(4) : 'N/A') + '</td>' +
            '<td>' + (row.mape !== null ? row.mape.toFixed(2) + '%' : 'N/A') + '</td>' +
            '<td>' + (row.correlation !== null ? row.correlation.toFixed(4) : 'N/A') + '</td>';
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
    if (data.actual.length === 0 && data.forecast.length === 0) {
        return;
    }
    
    let forecastStart = null;
    if (data.forecast.length > 0) {
        const forecastTimes = data.forecast.map(function(f) { return f.date.getTime(); });
        forecastStart = new Date(Math.min.apply(null, forecastTimes));
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
    
    const metrics = calculateMetrics(data.actual, data.forecast);
    
    const card = document.createElement('div');
    card.className = 'chart-card';
    const chartId = 'chart-' + seriesName.replace(/[^a-zA-Z0-9]/g, '_');
    
    let metricsHtml = '<span>No overlap</span>';
    if (metrics) {
        metricsHtml = 
            '<span>RMSE: ' + metrics.rmse.toFixed(2) + '</span>' +
            '<span>MAPE: ' + (metrics.mape !== null ? metrics.mape.toFixed(1) + '%' : 'N/A') + '</span>' +
            '<span>Horizon: ' + getHorizonLabel(state.selectedHorizon) + '</span>';
    } else {
        metricsHtml = '<span>Horizon: ' + getHorizonLabel(state.selectedHorizon) + '</span>';
    }
    
    card.innerHTML = 
        '<div class="chart-header">' +
            '<h3>' + seriesName + ' <span style="font-weight: normal; color: var(--text-secondary); font-size: 0.85rem;">- ' + (data.seriesInfo.description || '') + '</span></h3>' +
            '<div class="chart-metrics">' + metricsHtml + '</div>' +
        '</div>' +
        '<div class="chart-wrapper">' +
            '<canvas id="' + chartId + '"></canvas>' +
        '</div>' +
        '<div class="chart-legend">' +
            '<div class="legend-item"><div class="legend-line actual"></div><span>Actual</span></div>' +
            '<div class="legend-item"><div class="legend-line forecast"></div><span>Forecast</span></div>' +
        '</div>' +
        '<div class="chart-controls">' +
            '<button onclick="resetZoom(\'' + chartId + '\')">Reset Zoom</button>' +
        '</div>';
    
    elements.chartsContainer.appendChild(card);
    
    const allDateKeys = new Set();
    actualFiltered.forEach(function(a) { allDateKeys.add(a.dateKey); });
    data.forecast.forEach(function(f) { allDateKeys.add(f.dateKey); });
    
    const sortedDateKeys = Array.from(allDateKeys).sort();
    
    const actualMap = new Map(actualFiltered.map(function(a) { return [a.dateKey, a.value]; }));
    const forecastMap = new Map(data.forecast.map(function(f) { return [f.dateKey, f.value]; }));
    
    const actualValues = sortedDateKeys.map(function(key) {
        return actualMap.has(key) ? actualMap.get(key) : null;
    });
    const forecastValues = sortedDateKeys.map(function(key) {
        return forecastMap.has(key) ? forecastMap.get(key) : null;
    });
    
    const labels = sortedDateKeys.map(function(key) {
        const parts = key.split('-');
        return parts[1] + '/' + parts[0];
    });
    
    const ctx = document.getElementById(chartId).getContext('2d');
    
    const isDark = document.body.classList.contains('dark-theme');
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    const textColor = isDark ? '#e9ecef' : '#212529';
    
    chartInstances[chartId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Actual',
                    data: actualValues,
                    borderColor: CONFIG.actualColor,
                    backgroundColor: CONFIG.actualColor,
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    tension: 0.1,
                    spanGaps: true
                },
                {
                    label: 'Forecast',
                    data: forecastValues,
                    borderColor: CONFIG.forecastColor,
                    backgroundColor: CONFIG.forecastColor,
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    tension: 0.1,
                    spanGaps: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            if (value === null) return null;
                            return context.dataset.label + ': ' + value.toFixed(4);
                        }
                    }
                },
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'x'
                    },
                    zoom: {
                        wheel: {
                            enabled: true
                        },
                        pinch: {
                            enabled: true
                        },
                        mode: 'x'
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: gridColor
                    },
                    ticks: {
                        color: textColor,
                        maxRotation: 45,
                        minRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 12
                    }
                },
                y: {
                    grid: {
                        color: gridColor
                    },
                    ticks: {
                        color: textColor
                    }
                }
            }
        }
    });
}

function resetZoom(chartId) {
    if (chartInstances[chartId]) {
        chartInstances[chartId].resetZoom();
    }
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
            
            headers.forEach(function(h) {
                h.classList.remove('sorted-asc', 'sorted-desc');
            });
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
    
    elements.seriesSearch.addEventListener('input', function() {
        renderSeriesCheckboxes();
    });
    
    setupTableSorting();
}

// ==================== Initialize ====================
async function init() {
    console.log('Initializing dashboard...');
    setupEventListeners();
    await loadAllData();
    renderGroupCheckboxes();
    renderSeriesCheckboxes();
    updateDashboard();
    console.log('Dashboard initialized');
}

init();