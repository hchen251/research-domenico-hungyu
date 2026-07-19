// ============================================================
//  CONFIG
// ============================================================
const CONFIG = {
    api: '',   // same origin — server.py serves both API and static
    groups: {
        1:'Output and Income', 2:'Labor Market', 3:'Housing',
        4:'Consumption, Orders & Inventories', 5:'Money and Credit',
        6:'Interest and Exchange Rates', 7:'Prices', 8:'Stock Market'
    },
    historyMonths: 24,
    actualColor:   '#2563eb',
    forecastColor: '#dc2626',
    bayesianColor: '#7c3aed',
    ciColor:       'rgba(124,58,237,0.12)'
};

// ============================================================
//  API CLIENT
//  All data comes from the FastAPI server — no CSV parsing in JS.
//  date_key format returned by API: YYYY-MM
// ============================================================
async function apiFetch(path) {
    const url = CONFIG.api + path;
    const res = await fetch(url);
    if (!res.ok) {
        const txt = await res.text().catch(() => res.statusText);
        throw new Error(`API ${res.status} — ${url}\n${txt}`);
    }
    return res.json();
}

// Convert YYYY-MM date key to a JS timestamp (first of month, UTC noon)
function keyToTs(key) {
    const [y, m] = key.split('-').map(Number);
    return Date.UTC(y, m - 1, 15);   // mid-month avoids DST ambiguity
}

// Display label from YYYY-MM key
function keyToLabel(key) {
    const [y, m] = key.split('-');
    return m + '/' + y;              // e.g. "01/2000"
}

// Shared variable list (loaded once, used by both tabs)
let varList = [];  // [{name, group_id, group_name}]

async function ensureVarList() {
    if (varList.length) return varList;
    varList = await apiFetch('/api/variables');
    return varList;
}

// ============================================================
//  TAB MANAGEMENT
// ============================================================
let activeTab = 'spaghetti';

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + tab));
        activeTab = tab;
        if (tab === 'forecast' && !fxState.initialized) fxInit();
    });
});

document.getElementById('theme-toggle').addEventListener('click', () => {
    document.body.classList.toggle('dark-theme');
    document.body.classList.toggle('light-theme');
    document.getElementById('theme-toggle').textContent =
        document.body.classList.contains('dark-theme') ? 'Light Mode' : 'Dark Mode';
    if (spag.chart) spagRebuildChart();
    if (activeTab === 'forecast') fxUpdateCharts();
});

// ============================================================
//  ████  SPAGHETTI MODULE  ████
// ============================================================
const spag = {
    variable:   null,
    horizon:    1,
    nLines:     50,
    showCI:     true,
    showReal:   true,
    originFreq: 1,     // 1 = every 1Y, 5 = every 5Y
    data:       null,  // API response: {variable, horizon, n_draws, origins}
    actualPts:  null,  // [{ts, value}] from /api/actual
    tsMin:      0,
    tsMax:      0,
    viewLo:     0,     // 0–1000 slider
    viewHi:     1000,
    chart:      null,
    cache:      new Map()  // key: `${variable}_h${horizon}` → data
};

// ── Custom Chart.js plugin: draws all spaghetti fans on canvas ────────────
// Palette: cycles through these colours for successive forecast fans
const FAN_COLORS = [
    '#2563eb',  // blue
    '#dc2626',  // red
    '#16a34a',  // green
    '#9333ea',  // purple
    '#ea580c',  // orange
    '#0891b2',  // cyan
    '#be185d',  // pink
];
// Dash patterns mirror the reference chart style
const FAN_DASH = [
    [6, 4],     // blue dashed  (Fed style)
    [4, 3, 1, 3], // red dash-dot (Market style)
    [8, 4],
    [4, 4],
    [6, 3, 1, 3],
    [10, 4],
    [3, 3],
];

const SpaghettiPlugin = {
    id: 'spaghettiRenderer',
    beforeDatasetsDraw(chart) {
        if (!spag.data || !spag.actualPts) return;
        const { ctx, chartArea: ca, scales: { x, y } } = chart;
        if (!ca) return;

        const dark   = document.body.classList.contains('dark-theme');
        const dotBdr = dark ? '#1e293b' : '#ffffff';

        // Actual value lookup: ts → value
        const actualMap = new Map(spag.actualPts.map(p => [p.ts, p.value]));

        // Filter origins by frequency
        const origins = (spag.data.origins || []).filter(o => {
            if (spag.originFreq <= 1) return true;
            const year = parseInt(o.origin_date.slice(0, 4), 10);
            return !isNaN(year) && year % spag.originFreq === 0;
        });

        ctx.save();
        ctx.beginPath();
        ctx.rect(ca.left, ca.top, ca.width, ca.height);
        ctx.clip();

        origins.forEach((origin, oi) => {
            const steps = origin.steps || [];
            if (!steps.length) return;

            const originTs = keyToTs(origin.origin_date);
            const anchorY  = actualMap.get(originTs) ?? null;
            const ox       = x.getPixelForValue(originTs);
            const oy       = anchorY !== null ? y.getPixelForValue(anchorY) : null;

            // Pick colour & dash pattern for this fan (cycles)
            const fanColor = FAN_COLORS[oi % FAN_COLORS.length];
            const fanDash  = FAN_DASH[oi % FAN_DASH.length];
            // Make colour semi-transparent when many paths
            const alpha    = spag.nLines > 20 ? 0.35 : 0.75;
            const pathRgb  = hexToRgb(fanColor);
            const pathClr  = pathRgb
                ? 'rgba(' + pathRgb + ',' + alpha + ')'
                : fanColor;

            // ── CI band: p5 and p95 as dashed lines, no fill ─────────
            if (spag.showCI) {
                const valid = steps.filter(s => s.p5 !== null && s.p95 !== null);
                if (valid.length) {
                    ['p95', 'p5'].forEach(key => {
                        ctx.beginPath();
                        let started = false;
                        if (oy !== null) { ctx.moveTo(ox, oy); started = true; }
                        valid.forEach(s => {
                            const px = x.getPixelForValue(keyToTs(s.forecast_date));
                            const py = y.getPixelForValue(s[key]);
                            started ? ctx.lineTo(px, py) : (ctx.moveTo(px, py), started = true);
                        });
                        ctx.strokeStyle = pathRgb
                            ? 'rgba(' + pathRgb + ',0.5)'
                            : fanColor;
                        ctx.lineWidth = 1;
                        ctx.setLineDash([3, 5]);
                        ctx.stroke();
                        ctx.setLineDash([]);
                    });
                }
            }

            // ── Spaghetti paths ───────────────────────────────────────
            const nDraw = spag.nLines;
            if (nDraw > 0 && steps[0] && steps[0].draws && steps[0].draws.length) {
                const n = Math.min(nDraw, steps[0].draws.length);
                for (let d = 0; d < n; d++) {
                    ctx.beginPath();
                    let started = false;
                    if (oy !== null) { ctx.moveTo(ox, oy); started = true; }
                    steps.forEach(s => {
                        const val = s.draws[d];
                        if (val === null || val === undefined) return;
                        const px = x.getPixelForValue(keyToTs(s.forecast_date));
                        const py = y.getPixelForValue(val);
                        started ? ctx.lineTo(px, py) : (ctx.moveTo(px, py), started = true);
                    });
                    ctx.strokeStyle = pathClr;
                    ctx.lineWidth   = nDraw <= 5 ? 1.5 : nDraw <= 20 ? 1.1 : 0.8;
                    ctx.setLineDash(fanDash);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            }

            // ── Gray origin dot (like reference chart) ────────────────
            if (oy !== null) {
                ctx.beginPath();
                ctx.arc(ox, oy, 4.5, 0, Math.PI * 2);
                ctx.fillStyle   = dark ? '#94a3b8' : '#6b7280';
                ctx.strokeStyle = dotBdr;
                ctx.lineWidth   = 1.5;
                ctx.fill();
                ctx.stroke();
            }

            // ── Realized terminal dot ─────────────────────────────────
            if (spag.showReal) {
                steps.forEach(s => {
                    if (s.realized === null || s.realized === undefined) return;
                    const px = x.getPixelForValue(keyToTs(s.forecast_date));
                    const py = y.getPixelForValue(s.realized);
                    ctx.beginPath();
                    ctx.arc(px, py, 3, 0, Math.PI * 2);
                    ctx.fillStyle   = fanColor;
                    ctx.strokeStyle = dotBdr;
                    ctx.lineWidth   = 1;
                    ctx.fill();
                    ctx.stroke();
                });
            }
        });

        ctx.restore();
    }
};

// Helper: '#2563eb' → '37,99,235'
function hexToRgb(hex) {
    const r = /^#([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return r ? parseInt(r[1],16)+','+parseInt(r[2],16)+','+parseInt(r[3],16) : null;
}

Chart.register(SpaghettiPlugin);

// ── Rebuild the Chart.js instance ────────────────────────────────────────
function spagRebuildChart() {
    if (!spag.data || !spag.actualPts) return;

    const canvas = document.getElementById('spag-canvas');
    if (spag.chart) { spag.chart.destroy(); spag.chart = null; }

    const dark    = document.body.classList.contains('dark-theme');
    const grid    = dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)';
    const tickClr = dark ? '#94a3b8' : '#64748b';

    // Time extent: actual data + furthest forecast date
    const actualTs = spag.actualPts.map(p => p.ts);
    const fcTs = (spag.data.origins || [])
        .flatMap(o => o.steps.map(s => keyToTs(s.forecast_date)));

    spag.tsMin = actualTs.length ? Math.min(...actualTs) : Date.now() - 2e12;
    spag.tsMax = fcTs.length     ? Math.max(...fcTs)     : Date.now();

    const span    = spag.tsMax - spag.tsMin;
    const viewMin = spag.tsMin + span * (spag.viewLo / 1000);
    const viewMax = spag.tsMin + span * (spag.viewHi / 1000);

    // Actual dataset (x: timestamp, y: value)
    const actData  = spag.actualPts.map(p => ({ x: p.ts, y: p.value }));
    const haloClr  = document.body.classList.contains('dark-theme')
        ? 'rgba(15,23,42,0.65)' : 'rgba(255,255,255,0.70)';

    const ctx = canvas.getContext('2d');
    spag.chart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                // Halo: thick background stroke so actual line pops over dense fans
                { label: '_halo', data: actData, borderColor: haloClr,
                  backgroundColor: 'transparent', borderWidth: 10,
                  pointRadius: 0, tension: 0.1, spanGaps: true, order: 2 },
                // Actual line: bold black, on top of all fans
                { label: 'Actual', data: actData,
                  borderColor: dark ? '#f1f5f9' : '#111827',
                  backgroundColor: 'transparent',
                  borderWidth: 2.5, pointRadius: 0, pointHoverRadius: 5,
                  tension: 0.1, spanGaps: true, order: 0 }
            ]
        },
        options: {
            animation: false,
            responsive:          true,
            maintainAspectRatio: false,
            parsing: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: ctx => ctx.length ? keyToLabel(
                            new Date(ctx[0].parsed.x).toISOString().slice(0, 7)) : '',
                        label: ctx => {
                            if (ctx.dataset.label === '_halo') return null;
                            return ctx.dataset.label === 'Actual' && ctx.parsed.y !== null
                                ? 'Actual: ' + ctx.parsed.y.toFixed(4) : null;
                        }
                    }
                },
                zoom: {
                    pan:  { enabled: true, mode: 'x' },
                    zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' }
                },
                spaghettiRenderer: {}
            },
            scales: {
                x: {
                    type: 'linear', min: viewMin, max: viewMax,
                    grid:  { color: grid },
                    ticks: { color: tickClr, maxTicksLimit: 12,
                             callback: v => keyToLabel(new Date(v).toISOString().slice(0, 7)) }
                },
                y: { grid: { color: grid }, ticks: { color: tickClr } }
            }
        }
    });

    spagUpdateRangeSlider();
    spagUpdateStatInfo();
}

function spagUpdateView() {
    if (!spag.chart) return;
    const span    = spag.tsMax - spag.tsMin;
    const viewMin = spag.tsMin + span * (spag.viewLo / 1000);
    const viewMax = spag.tsMin + span * (spag.viewHi / 1000);
    spag.chart.options.scales.x.min = viewMin;
    spag.chart.options.scales.x.max = viewMax;
    spag.chart.update('none');
    const lo = document.getElementById('range-lo-label');
    const hi = document.getElementById('range-hi-label');
    lo.textContent = keyToLabel(new Date(viewMin).toISOString().slice(0, 7));
    hi.textContent = keyToLabel(new Date(viewMax).toISOString().slice(0, 7));
}

function spagUpdateRangeSlider() {
    document.getElementById('range-wrap').style.display = 'block';
    spagSyncFill();
    const span = spag.tsMax - spag.tsMin;
    const lo   = document.getElementById('range-lo-label');
    const hi   = document.getElementById('range-hi-label');
    lo.textContent = keyToLabel(new Date(spag.tsMin + span * spag.viewLo / 1000).toISOString().slice(0,7));
    hi.textContent = keyToLabel(new Date(spag.tsMin + span * spag.viewHi / 1000).toISOString().slice(0,7));
}

function spagSyncFill() {
    const fill = document.getElementById('range-fill');
    fill.style.left  = (spag.viewLo / 10) + '%';
    fill.style.width = ((spag.viewHi - spag.viewLo) / 10) + '%';
}

function spagUpdateStatInfo() {
    const el      = document.getElementById('spag-stat-info');
    const origins = spag.data?.origins || [];
    const visible = spag.originFreq <= 1
        ? origins
        : origins.filter(o => parseInt(o.origin_date.slice(0,4),10) % spag.originFreq === 0);
    el.textContent = `${visible.length} origins · ${spag.nLines} paths · h=${spag.horizon}m`;
}

// ── Main render ────────────────────────────────────────────────────────────
async function spagRender(variable, horizon) {
    if (!variable) return;

    const empty    = document.getElementById('spag-empty');
    const loading  = document.getElementById('spag-loading');
    const loadMsg  = document.getElementById('spag-loading-msg');
    const canvas   = document.getElementById('spag-canvas');
    const legend   = document.getElementById('spag-legend');
    const rangeWrap= document.getElementById('range-wrap');

    empty.style.display   = 'none';
    loading.style.display = 'flex';
    canvas.style.display  = 'none';
    legend.style.display  = 'none';
    rangeWrap.style.display = 'none';

    try {
        loadMsg.textContent = 'Loading actual data…';
        const actualRows = await apiFetch(`/api/actual/${encodeURIComponent(variable)}`);
        // actualRows: [{date_key: "YYYY-MM", value: number}]
        spag.actualPts = actualRows.map(r => ({ ts: keyToTs(r.date_key), value: r.value }));

        loadMsg.textContent = 'Loading backtest draws…';
        const cacheKey = `${variable}_h${horizon}`;
        let data = spag.cache.get(cacheKey);
        if (!data) {
            const freq = spag.originFreq;
            data = await apiFetch(
                `/api/spaghetti/${encodeURIComponent(variable)}/${horizon}?freq=${freq}`
            );
            spag.cache.set(cacheKey, data);
        }

        spag.variable  = variable;
        spag.horizon   = horizon;
        spag.data      = data;
        spag.viewLo    = 0;
        spag.viewHi    = 1000;
        document.getElementById('range-lo').value = 0;
        document.getElementById('range-hi').value = 1000;

        canvas.style.display = 'block';
        spagRebuildChart();
        legend.style.display = 'flex';

    } catch (e) {
        console.error('Spaghetti load failed:', e);
        empty.querySelector('h3').textContent = 'Failed to load data';
        empty.querySelector('p').textContent  = e.message;
        empty.style.display = 'flex';
    } finally {
        loading.style.display = 'none';
    }
}

// ── Control wiring ─────────────────────────────────────────────────────────
function spagSetupControls() {
    // Variable selector
    document.getElementById('spag-var').addEventListener('change', e => {
        spag.variable = e.target.value || null;
        if (spag.variable) spagRender(spag.variable, spag.horizon);
    });

    // Horizon pills
    document.querySelectorAll('.h-pill').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.h-pill').forEach(b => b.classList.toggle('active', b === btn));
            spag.horizon = parseInt(btn.dataset.h);
            if (spag.variable) spagRender(spag.variable, spag.horizon);
        });
    });

    // Origin frequency
    document.querySelectorAll('.seg-btn[data-freq]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.seg-btn[data-freq]')
                    .forEach(b => b.classList.toggle('active', b === btn));
            spag.originFreq = parseInt(btn.dataset.freq);
            spag.cache.clear();  // invalidate cache — server filters by freq
            if (spag.variable) spagRender(spag.variable, spag.horizon);
        });
    });

    // Density slider
    document.getElementById('spag-density').addEventListener('input', e => {
        spag.nLines = parseInt(e.target.value);
        document.getElementById('density-val').textContent = spag.nLines;
        if (spag.chart) spag.chart.update('none');
    });

    // CI toggle
    document.getElementById('ci-toggle').addEventListener('click', function() {
        spag.showCI = !spag.showCI;
        this.classList.toggle('active', spag.showCI);
        this.textContent = spag.showCI ? 'On' : 'Off';
        if (spag.chart) spag.chart.update('none');
    });

    // Realized toggle
    document.getElementById('realized-toggle').addEventListener('click', function() {
        spag.showReal = !spag.showReal;
        this.classList.toggle('active', spag.showReal);
        this.textContent = spag.showReal ? 'On' : 'Off';
        if (spag.chart) spag.chart.update('none');
    });

    // Dual range slider
    const lo = document.getElementById('range-lo');
    const hi = document.getElementById('range-hi');
    function onRange() {
        let lv = parseInt(lo.value), hv = parseInt(hi.value);
        if (lv >= hv - 20) {
            if (document.activeElement === lo) lv = hv - 20;
            else                               hv = lv + 20;
            lo.value = lv; hi.value = hv;
        }
        spag.viewLo = lv; spag.viewHi = hv;
        spagSyncFill();
        spagUpdateView();
    }
    lo.addEventListener('input', onRange);
    hi.addEventListener('input', onRange);
}

// ── Populate variable selector from API ───────────────────────────────────
async function spagInitVarSelector() {
    const sel = document.getElementById('spag-var');
    try {
        const vars = await ensureVarList();
        sel.innerHTML = '<option value="">— select —</option>';

        // Group variables
        const byGroup = {};
        vars.forEach(v => {
            const g = v.group_id;
            (byGroup[g] = byGroup[g] || []).push(v);
        });

        Object.keys(byGroup).sort((a,b) => +a - +b).forEach(gid => {
            const og = document.createElement('optgroup');
            og.label = byGroup[gid][0].group_name;
            byGroup[gid].sort((a,b) => a.name.localeCompare(b.name)).forEach(v => {
                const opt = document.createElement('option');
                opt.value = v.name; opt.textContent = v.name;
                og.appendChild(opt);
            });
            sel.appendChild(og);
        });
    } catch (e) {
        sel.innerHTML = '<option value="">API unavailable</option>';
        console.error('Failed to load variable list:', e);
    }
}

// ============================================================
//  ████  FORECAST MODULE  ████
// ============================================================
const fxState = {
    initialized: false,
    varList:     [],    // from API
    selectedHorizon: 12,
    selectedGroups:  new Set(),
    selectedSeries:  new Set(),
    sortColumn:      null,
    sortDirection:   'asc',
    showDFM:      true,
    showBayesian: true,
    showCI:       true,
    // per-series caches
    actualCache:  new Map(),   // variable → [{date_key, value}]
    fcCache:      new Map()    // `${variable}_h${h}` → {dfm, bay_med, bay_p5, bay_p95}
};

let fxCharts = {};

function fxShowLoading() { document.getElementById('loading').classList.add('active'); }
function fxHideLoading() { document.getElementById('loading').classList.remove('active'); }

async function fxGetActual(variable) {
    if (fxState.actualCache.has(variable)) return fxState.actualCache.get(variable);
    const rows = await apiFetch(`/api/actual/${encodeURIComponent(variable)}`);
    // rows: [{date_key: "YYYY-MM", value}]
    fxState.actualCache.set(variable, rows);
    return rows;
}

async function fxGetForecast(variable, horizon) {
    const key = `${variable}_h${horizon}`;
    if (fxState.fcCache.has(key)) return fxState.fcCache.get(key);
    try {
        const data = await apiFetch(`/api/forecast/${encodeURIComponent(variable)}/${horizon}`);
        fxState.fcCache.set(key, data);
        return data;
    } catch {
        return { dfm: [], bay_med: [], bay_p5: [], bay_p95: [] };
    }
}

function fxCalcMetrics(actual, forecast) {
    // actual/forecast: [{date_key, value}]
    if (!actual?.length || !forecast?.length) return null;
    const fmap = new Map(forecast.map(f => [f.date_key, f.value]));
    const pairs = actual
        .filter(a => fmap.has(a.date_key) && a.value !== null && fmap.get(a.date_key) !== null)
        .map(a => ({ a: a.value, f: fmap.get(a.date_key) }));
    if (!pairs.length) return null;
    const n = pairs.length;
    let ss=0, sap=0, vpc=0, sa=0, sf=0, sa2=0, sf2=0, sp=0;
    pairs.forEach(({a, f}) => {
        const e = f - a;
        ss  += e * e;
        if (a !== 0) { sap += Math.abs(e / a) * 100; vpc++; }
        sa  += a; sf += f; sa2 += a*a; sf2 += f*f; sp += a*f;
    });
    let corr = null;
    if (n > 1) {
        const num = n*sp - sa*sf;
        const da  = Math.sqrt(n*sa2 - sa*sa);
        const db  = Math.sqrt(n*sf2 - sf*sf);
        if (da > 0 && db > 0) corr = num / (da * db);
    }
    return { rmse: Math.sqrt(ss/n), mape: vpc ? sap/vpc : null, correlation: corr, n };
}

function fxUpdateToggleBtn(btn, active, color) {
    if (!btn) return;
    btn.classList.toggle('toggle-active', active);
    btn.style.borderColor     = active ? color : 'var(--border-color)';
    btn.style.color           = active ? color : 'var(--text-secondary)';
    btn.style.backgroundColor = active ? color + '18' : 'var(--bg-tertiary)';
}

function fxUpdateToggleButtons() {
    fxUpdateToggleBtn(document.getElementById('toggle-dfm'),      fxState.showDFM,      CONFIG.forecastColor);
    fxUpdateToggleBtn(document.getElementById('toggle-bayesian'), fxState.showBayesian, CONFIG.bayesianColor);
    const ci = document.getElementById('toggle-ci');
    if (ci) { ci.classList.toggle('toggle-active', fxState.showCI); ci.style.opacity = fxState.showBayesian ? '1' : '0.4'; ci.disabled = !fxState.showBayesian; }
}

function fxRenderGroupCheckboxes() {
    const el = document.getElementById('group-checkboxes');
    el.innerHTML = '';
    const groups = {};
    fxState.varList.forEach(v => { (groups[v.group_id] = groups[v.group_id] || {name: v.group_name, count: 0}).count++; });
    Object.keys(groups).sort((a,b) => +a - +b).forEach(id => {
        const g = groups[id];
        const div = document.createElement('div'); div.className = 'checkbox-item';
        div.innerHTML = `<input type="checkbox" id="group-${id}" value="${id}">
            <label for="group-${id}">${id}. ${g.name}</label>
            <span class="group-badge">${g.count}</span>`;
        div.querySelector('input').addEventListener('change', e => fxHandleGroupChange(parseInt(id), e.target.checked));
        el.appendChild(div);
    });
}

function fxRenderSeriesCheckboxes() {
    const el     = document.getElementById('series-checkboxes');
    const search = document.getElementById('series-search').value.toLowerCase();
    el.innerHTML = '';

    let all = fxState.varList.filter(v => v.name.toLowerCase().includes(search));
    if (!all.length) { el.innerHTML = '<div class="empty-state"><p>No series found</p></div>'; return; }

    const inGrp  = all.filter(v =>  fxState.selectedGroups.has(v.group_id));
    const outGrp = all.filter(v => !fxState.selectedGroups.has(v.group_id));

    const byGroup = {};
    inGrp.forEach(v => { (byGroup[v.group_id] = byGroup[v.group_id] || []).push(v); });

    Object.keys(byGroup).sort((a,b) => +a - +b).forEach(gid => {
        const hdr = document.createElement('div');
        hdr.style.cssText = 'padding:.5rem;margin-top:.5rem;font-size:.8rem;color:var(--accent-color);background:var(--bg-tertiary);border-radius:4px;';
        hdr.innerHTML = `<strong>${byGroup[gid][0].group_name}</strong>`;
        el.appendChild(hdr);
        byGroup[gid].sort((a,b) => a.name.localeCompare(b.name)).forEach(v => fxRenderSeriesRow(v, true));
    });

    if (outGrp.length && fxState.selectedGroups.size > 0) {
        const hdr = document.createElement('div');
        hdr.style.cssText = 'padding:.5rem;margin-top:.5rem;font-size:.8rem;color:var(--text-secondary);';
        hdr.innerHTML = '<strong>Other Series</strong>';
        el.appendChild(hdr);
    }
    outGrp.sort((a,b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()))
          .forEach(v => fxRenderSeriesRow(v, false));
}

function fxRenderSeriesRow(v, highlight) {
    const el  = document.getElementById('series-checkboxes');
    const div = document.createElement('div');
    div.className = 'checkbox-item' + (highlight ? ' in-selected-group' : '');
    const sid = v.name.replace(/[^a-zA-Z0-9]/g, '_');
    const chk = fxState.selectedSeries.has(v.name) ? 'checked' : '';
    div.innerHTML = `<input type="checkbox" id="series-${sid}" value="${v.name}" ${chk}>
        <label for="series-${sid}">${v.name}</label>
        <span class="group-badge">${v.group_id}</span>`;
    div.querySelector('input').addEventListener('change', e => fxHandleSeriesChange(v.name, e.target.checked));
    el.appendChild(div);
}

function fxHandleGroupChange(id, checked) {
    if (checked) fxState.selectedGroups.add(id);
    else {
        fxState.selectedGroups.delete(id);
        fxState.varList.filter(v => v.group_id === id).forEach(v => fxState.selectedSeries.delete(v.name));
    }
    fxRenderSeriesCheckboxes();
    document.getElementById('selected-count').textContent = fxState.selectedSeries.size + ' series selected';
    fxUpdate();
}

function fxHandleSeriesChange(name, checked) {
    checked ? fxState.selectedSeries.add(name) : fxState.selectedSeries.delete(name);
    document.getElementById('selected-count').textContent = fxState.selectedSeries.size + ' series selected';
    fxUpdate();
}

function fxUpdate() {
    fxUpdateToggleButtons();
    fxUpdateMetricsTable();
    fxUpdateCharts();
}

async function fxUpdateMetricsTable() {
    const body = document.getElementById('metrics-body');
    if (!fxState.selectedSeries.size) {
        body.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:2rem;color:var(--text-secondary)">Select series to view metrics</td></tr>';
        return;
    }

    const rows = [];
    await Promise.all(Array.from(fxState.selectedSeries).map(async name => {
        const [actual, fc] = await Promise.all([
            fxGetActual(name),
            fxGetForecast(name, fxState.selectedHorizon)
        ]);
        const info = fxState.varList.find(v => v.name === name);
        rows.push({
            name,
            group: info?.group_name || '',
            groupId: info?.group_id || 1,
            dfm: fxState.showDFM ? fxCalcMetrics(actual, fc.dfm) : null,
            bay: (fxState.showBayesian && fc.bay_med?.length) ? fxCalcMetrics(actual, fc.bay_med) : null
        });
    }));

    if (fxState.sortColumn) {
        rows.sort((a, b) => {
            let av, bv;
            if      (fxState.sortColumn === 'series') { av = a.name;  bv = b.name; }
            else if (fxState.sortColumn === 'group')  { av = a.group; bv = b.group; }
            else if (fxState.sortColumn.startsWith('dfm_')) { const k = fxState.sortColumn.slice(4); av = a.dfm?.[k]; bv = b.dfm?.[k]; }
            else if (fxState.sortColumn.startsWith('bay_')) { const k = fxState.sortColumn.slice(4); av = a.bay?.[k]; bv = b.bay?.[k]; }
            if (typeof av === 'string') { av = av.toLowerCase(); bv = (bv||'').toLowerCase(); }
            if (av == null) return 1; if (bv == null) return -1;
            return (fxState.sortDirection === 'asc') ? (av > bv ? 1 : av < bv ? -1 : 0) : (av < bv ? 1 : av > bv ? -1 : 0);
        });
    }

    const fmt = (v, dp) => v !== null && v !== undefined ? v.toFixed(dp) : 'N/A';
    body.innerHTML = '';
    rows.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML =
            `<td><strong>${row.name}</strong></td><td>${row.group}</td>` +
            `<td style="border-left:2px solid ${CONFIG.forecastColor}22">${row.dfm ? fmt(row.dfm.rmse,4) : '—'}</td>` +
            `<td>${row.dfm ? fmt(row.dfm.mape,2)+'%' : '—'}</td>` +
            `<td>${row.dfm ? fmt(row.dfm.correlation,4) : '—'}</td>` +
            `<td style="border-left:2px solid ${CONFIG.bayesianColor}44">${row.bay ? fmt(row.bay.rmse,4) : '—'}</td>` +
            `<td>${row.bay ? fmt(row.bay.mape,2)+'%' : '—'}</td>` +
            `<td>${row.bay ? fmt(row.bay.correlation,4) : '—'}</td>`;
        body.appendChild(tr);
    });
}

function fxUpdateCharts() {
    Object.values(fxCharts).forEach(c => c.destroy());
    fxCharts = {};
    const container = document.getElementById('charts-container');
    container.innerHTML = '';
    if (!fxState.selectedSeries.size) {
        container.innerHTML = '<div class="empty-state"><h3>No Series Selected</h3><p>Select groups and series from the sidebar</p></div>';
        return;
    }
    fxState.selectedSeries.forEach(name => fxCreateChart(name));
}

async function fxCreateChart(name) {
    const [actual, fc] = await Promise.all([
        fxGetActual(name),
        fxGetForecast(name, fxState.selectedHorizon)
    ]);

    const info = fxState.varList.find(v => v.name === name);

    // Determine history window
    const allFcKeys = [
        ...(fxState.showDFM ? fc.dfm : []),
        ...(fxState.showBayesian ? fc.bay_med : [])
    ].map(r => r.date_key);

    let histStart;
    if (allFcKeys.length) {
        const minFcKey = allFcKeys.sort()[0];
        const [fy, fm] = minFcKey.split('-').map(Number);
        const d = new Date(fy, fm - 1 - CONFIG.historyMonths, 1);
        histStart = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}`;
    } else {
        histStart = actual[0]?.date_key || '1959-01';
    }

    const actFiltered = actual.filter(r => r.date_key >= histStart);

    // Build unified date axis
    const allKeys = new Set([
        ...actFiltered.map(r => r.date_key),
        ...(fxState.showDFM ? fc.dfm.map(r => r.date_key) : []),
        ...(fxState.showBayesian ? fc.bay_med.map(r => r.date_key) : [])
    ]);
    const keys = Array.from(allKeys).sort();

    const amap   = new Map(actFiltered.map(r => [r.date_key, r.value]));
    const fmap   = new Map(fc.dfm.map(r => [r.date_key, r.value]));
    const bmmap  = new Map(fc.bay_med.map(r => [r.date_key, r.value]));
    const bp5map = new Map(fc.bay_p5.map(r => [r.date_key, r.value]));
    const b95map = new Map(fc.bay_p95.map(r => [r.date_key, r.value]));

    const labels   = keys.map(keyToLabel);
    const actVals  = keys.map(k => amap.get(k)   ?? null);
    const fcVals   = keys.map(k => fmap.get(k)   ?? null);
    const medVals  = keys.map(k => bmmap.get(k)  ?? null);
    const p5Vals   = keys.map(k => bp5map.get(k) ?? null);
    const p95Vals  = keys.map(k => b95map.get(k) ?? null);

    const dark    = document.body.classList.contains('dark-theme');
    const grid    = dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
    const textCol = dark ? '#e9ecef' : '#212529';
    const chartId = 'fxchart-' + name.replace(/[^a-zA-Z0-9]/g, '_');

    const dfmM  = fxState.showDFM ? fxCalcMetrics(actual, fc.dfm) : null;
    const bayM  = (fxState.showBayesian && fc.bay_med?.length) ? fxCalcMetrics(actual, fc.bay_med) : null;
    const hLbl  = fxState.selectedHorizon === 1 ? '1 Month' : fxState.selectedHorizon + ' Months';

    let mHtml = `<span>Horizon: ${hLbl}</span>`;
    if (dfmM) mHtml += `<span style="color:${CONFIG.forecastColor}">DFM RMSE: ${dfmM.rmse.toFixed(2)}</span><span style="color:${CONFIG.forecastColor}">MAPE: ${dfmM.mape!=null?dfmM.mape.toFixed(1)+'%':'N/A'}</span>`;
    if (bayM) mHtml += `<span style="color:${CONFIG.bayesianColor}">Bayes RMSE: ${bayM.rmse.toFixed(2)}</span><span style="color:${CONFIG.bayesianColor}">MAPE: ${bayM.mape!=null?bayM.mape.toFixed(1)+'%':'N/A'}</span>`;

    let legHtml = '<div class="legend-item"><div class="legend-line actual"></div><span>Actual</span></div>';
    if (fxState.showDFM) legHtml += '<div class="legend-item"><div class="legend-line forecast"></div><span>DFM Forecast</span></div>';
    if (fxState.showBayesian && fc.bay_med?.length) {
        legHtml += '<div class="legend-item"><div class="legend-line bayesian"></div><span>Bayesian Forecast</span></div>';
        if (fxState.showCI) legHtml += '<div class="legend-item"><div class="legend-ci-band"></div><span>90% CI</span></div>';
    }

    const container = document.getElementById('charts-container');
    const card = document.createElement('div'); card.className = 'chart-card';
    card.innerHTML = `
        <div class="chart-header">
            <h3>${name} <span style="font-weight:normal;color:var(--text-secondary);font-size:.85rem">— ${info?.group_name||''}</span></h3>
            <div class="chart-metrics">${mHtml}</div>
        </div>
        <div class="chart-wrapper"><canvas id="${chartId}"></canvas></div>
        <div class="chart-legend">${legHtml}</div>
        <div class="chart-controls"><button onclick="fxResetZoom('${chartId}')">Reset Zoom</button></div>`;
    container.appendChild(card);

    const datasets = [{
        label:'Actual', data:actVals, borderColor:CONFIG.actualColor, backgroundColor:CONFIG.actualColor,
        borderWidth:2, pointRadius:2, pointHoverRadius:5, tension:.1, spanGaps:true, order:1
    }];
    if (fxState.showDFM) datasets.push({
        label:'DFM Forecast', data:fcVals, borderColor:CONFIG.forecastColor, backgroundColor:CONFIG.forecastColor,
        borderWidth:2, borderDash:[5,5], pointRadius:2, pointHoverRadius:5, tension:.1, spanGaps:true, order:2
    });
    if (fxState.showBayesian && fc.bay_med?.length) {
        if (fxState.showCI) {
            datasets.push({ label:'CI Upper (P95)', data:p95Vals, borderColor:'transparent', backgroundColor:CONFIG.ciColor, borderWidth:0, pointRadius:0, fill:'+1', tension:.1, spanGaps:true, order:5 });
            datasets.push({ label:'CI Lower (P5)',  data:p5Vals,  borderColor:'transparent', backgroundColor:CONFIG.ciColor, borderWidth:0, pointRadius:0, fill:false, tension:.1, spanGaps:true, order:5 });
        }
        datasets.push({ label:'Bayesian Forecast', data:medVals, borderColor:CONFIG.bayesianColor, backgroundColor:CONFIG.bayesianColor, borderWidth:2, borderDash:[4,3], pointRadius:2, pointHoverRadius:5, tension:.1, spanGaps:true, order:3 });
    }

    const ctx = document.getElementById(chartId)?.getContext('2d');
    if (!ctx) return;

    fxCharts[chartId] = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive:true, maintainAspectRatio:false,
            interaction:{ intersect:false, mode:'index' },
            plugins: {
                legend:{ display:false },
                tooltip:{ callbacks:{ label: c => {
                    if (c.parsed.y === null) return null;
                    const l = c.dataset.label;
                    if (l === 'CI Upper (P95)' || l === 'CI Lower (P5)') return null;
                    return l + ': ' + c.parsed.y.toFixed(4);
                }}},
                zoom:{ pan:{enabled:true,mode:'x'}, zoom:{wheel:{enabled:true},pinch:{enabled:true},mode:'x'} }
            },
            scales: {
                x:{ grid:{color:grid}, ticks:{color:textCol,maxRotation:45,minRotation:0,autoSkip:true,maxTicksLimit:12} },
                y:{ grid:{color:grid}, ticks:{color:textCol} }
            }
        }
    });
}

function fxResetZoom(id) { if (fxCharts[id]) fxCharts[id].resetZoom(); }
window.fxResetZoom = fxResetZoom;

function fxSetupTableSorting() {
    document.getElementById('metrics-table')?.querySelectorAll('th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.getAttribute('data-sort');
            fxState.sortDirection = (fxState.sortColumn === col && fxState.sortDirection === 'asc') ? 'desc' : 'asc';
            fxState.sortColumn = col;
            document.querySelectorAll('#metrics-table th[data-sort]').forEach(h => h.classList.remove('sorted-asc','sorted-desc'));
            th.classList.add('sorted-' + fxState.sortDirection);
            fxUpdateMetricsTable();
        });
    });
}

async function fxInit() {
    if (fxState.initialized) return;
    fxShowLoading();
    try {
        fxState.varList = await ensureVarList();
        fxState.initialized = true;
        fxRenderGroupCheckboxes();
        fxRenderSeriesCheckboxes();
        fxUpdate();
    } catch (e) {
        console.error('Forecast init error:', e);
        document.getElementById('charts-container').innerHTML =
            `<div class="empty-state"><h3>Failed to connect to API</h3><p>${e.message}</p><p>Make sure server.py is running.</p></div>`;
    } finally {
        fxHideLoading();
    }

    document.getElementById('horizon')?.addEventListener('change', e => {
        fxState.selectedHorizon = parseInt(e.target.value);
        fxState.fcCache.clear();
        fxUpdate();
    });
    document.getElementById('select-all-groups')?.addEventListener('click', () => {
        fxState.varList.forEach(v => { fxState.selectedGroups.add(v.group_id); const cb=document.getElementById('group-'+v.group_id); if(cb) cb.checked=true; });
        fxRenderSeriesCheckboxes(); document.getElementById('selected-count').textContent=fxState.selectedSeries.size+' series selected'; fxUpdate();
    });
    document.getElementById('clear-all-groups')?.addEventListener('click', () => {
        fxState.selectedGroups.clear(); fxState.selectedSeries.clear();
        document.querySelectorAll('#group-checkboxes input').forEach(cb => cb.checked=false);
        fxRenderSeriesCheckboxes(); document.getElementById('selected-count').textContent='0 series selected'; fxUpdate();
    });
    document.getElementById('select-all-series')?.addEventListener('click', () => {
        fxState.varList.filter(v => fxState.selectedGroups.has(v.group_id)).forEach(v => fxState.selectedSeries.add(v.name));
        fxRenderSeriesCheckboxes(); document.getElementById('selected-count').textContent=fxState.selectedSeries.size+' series selected'; fxUpdate();
    });
    document.getElementById('clear-all-series')?.addEventListener('click', () => {
        fxState.selectedSeries.clear(); fxRenderSeriesCheckboxes();
        document.getElementById('selected-count').textContent='0 series selected'; fxUpdate();
    });
    document.getElementById('series-search')?.addEventListener('input', fxRenderSeriesCheckboxes);
    document.getElementById('toggle-dfm')?.addEventListener('click',      () => { fxState.showDFM      = !fxState.showDFM;      fxUpdate(); });
    document.getElementById('toggle-bayesian')?.addEventListener('click', () => { fxState.showBayesian = !fxState.showBayesian; fxUpdate(); });
    document.getElementById('toggle-ci')?.addEventListener('click',       () => { if (!fxState.showBayesian) return; fxState.showCI = !fxState.showCI; fxUpdate(); });
    fxSetupTableSorting();
}

// ============================================================
//  INIT
// ============================================================
spagSetupControls();
spagInitVarSelector();   // pre-populate dropdown on page load
