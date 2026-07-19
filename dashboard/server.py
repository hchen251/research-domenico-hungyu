#!/usr/bin/env python3
"""
server.py
──────────────────────────────────────────────────────────────────────────────
FastAPI backend for the DFM Dashboard.
Serves all data from backtest.db and hosts the static dashboard files.

INSTALL
  pip install fastapi uvicorn

RUN
  python server.py                         # default: port 8000
  python server.py --port 8080
  python server.py --db path/to/backtest.db --static path/to/dashboard/

ENDPOINTS
  GET /api/variables
      → [{name, group_id, group_name}]

  GET /api/actual/{variable}
      → [{date_key, value}]   YYYY-MM date keys, chronological

  GET /api/forecast/{variable}/{horizon}
      → {dfm:    [{date_key, value}],
         bay_med: [{date_key, value}],
         bay_p5:  [{date_key, value}],
         bay_p95: [{date_key, value}]}

  GET /api/spaghetti/{variable}/{horizon}
      → {variable, horizon, n_draws,
         origins: [{origin_date, steps: [{forecast_date,
                    mean, p5, p50, p95, realized, draws}]}]}

  GET /api/coverage/{variable}/{horizon}
      → [{origin_date, forecast_date, in_band}]  (p5-p95 coverage)

  GET /health  →  {status: "ok", db: "..."}
"""

import json
import os
import sqlite3
import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ── CLI args ─────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--db',     default='backtest.db', help='SQLite database file')
    p.add_argument('--static', default='.',           help='Static files directory (dashboard)')
    p.add_argument('--host',   default='0.0.0.0')
    p.add_argument('--port',   default=8000, type=int)
    return p.parse_args()

ARGS = get_args()

# ── Series group map (mirrors app.js CONFIG.seriesGroups) ─────────────────────
SERIES_GROUPS = {
    **{k: 1 for k in ('RPI','W875RX1','DPCERA3M086SBEA','CMRMTSPLx','RETAILx',
       'INDPRO','IPFPNSS','IPFINAL','IPCONGD','IPDCONGD','IPNCONGD','IPBUSEQ',
       'IPMAT','IPDMAT','IPNMAT','IPMANSICS','IPB51222S','IPFUELS','CUMFNS')},
    **{k: 2 for k in ('HWI','HWIURATIO','CLF16OV','CE16OV','UNRATE','UEMPMEAN',
       'UEMPLT5','UEMP5TO14','UEMP15OV','UEMP15T26','UEMP27OV','CLAIMSx',
       'PAYEMS','USGOOD','CES1021000001','USCONS','MANEMP','DMANEMP','NDMANEMP',
       'SRVPRD','USTPU','USWTRADE','USTRADE','USFIRE','USGOVT','CES0600000007',
       'AWOTMAN','AWHMAN','CES0600000008','CES2000000008','CES3000000008')},
    **{k: 3 for k in ('HOUST','HOUSTNE','HOUSTMW','HOUSTS','HOUSTW',
       'PERMIT','PERMITNE','PERMITMW','PERMITS','PERMITW')},
    **{k: 4 for k in ('ACOGNO','AMDMNOx','ANDENOx','AMDMUOx','BUSINVx','ISRATIOx','UMCSENTx')},
    **{k: 5 for k in ('M1SL','M2SL','M2REAL','BOGMBASE','TOTRESNS','NONBORRES',
       'BUSLOANS','REALLN','NONREVSL','CONSPI','DTCOLNVHFNM','DTCTHFNM','INVEST')},
    **{k: 6 for k in ('FEDFUNDS','CP3Mx','TB3MS','TB6MS','GS1','GS5','GS10','AAA',
       'BAA','COMPAPFFx','TB3SMFFM','TB6SMFFM','T1YFFM','T5YFFM','T10YFFM',
       'AAAFFM','BAAFFM','TWEXAFEGSMTHx','EXSZUSx','EXJPUSx','EXUSUKx','EXCAUSx')},
    **{k: 7 for k in ('WPSFD49207','WPSFD49502','WPSID61','WPSID62','OILPRICEx',
       'PPICMM','CPIAUCSL','CPIAPPSL','CPITRNSL','CPIMEDSL','CUSR0000SAC',
       'CUSR0000SAD','CUSR0000SAS','CPIULFSL','CUSR0000SA0L2','CUSR0000SA0L5',
       'PCEPI','DDURRG3M086SBEA','DNDGRG3M086SBEA','DSERRG3M086SBEA')},
    **{k: 8 for k in ('S&P 500','S&P div yield','S&P PE ratio','VIXCLSx',
       'S_P500','S_PDivYield','S_PPERatio')},
}
GROUP_NAMES = {
    1:'Output and Income', 2:'Labor Market', 3:'Housing',
    4:'Consumption, Orders & Inventories', 5:'Money and Credit',
    6:'Interest and Exchange Rates', 7:'Prices', 8:'Stock Market'
}

# ── DB connection pool (one per request via context manager) ──────────────────
@contextmanager
def get_db():
    con = sqlite3.connect(ARGS.db, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def db_rows(query: str, params: tuple = ()) -> List[dict]:
    with get_db() as con:
        cur = con.execute(query, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title='DFM Dashboard API', docs_url='/api/docs')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*']
)


# ── /health ───────────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    return {'status': 'ok', 'db': os.path.abspath(ARGS.db)}


# ── /api/variables ────────────────────────────────────────────────────────────
@app.get('/api/variables')
def variables():
    """All variables present in the backtest stats table."""
    rows = db_rows('SELECT DISTINCT variable FROM stats ORDER BY variable')
    result = []
    for r in rows:
        name = r['variable']
        gid  = SERIES_GROUPS.get(name, 1)
        result.append({
            'name':       name,
            'group_id':   gid,
            'group_name': GROUP_NAMES.get(gid, 'Other')
        })
    return result


# ── /api/actual/{variable} ────────────────────────────────────────────────────
@app.get('/api/actual/{variable}')
def actual(variable: str):
    """
    Returns realized values for a variable from the backtest stats table.
    Uses forecast_date + realized to reconstruct the actual time series.
    Deduplicates so each date appears only once.
    """
    rows = db_rows(
        '''SELECT DISTINCT forecast_date AS date_key, realized AS value
           FROM stats
           WHERE variable=? AND realized IS NOT NULL
           ORDER BY forecast_date''',
        (variable,)
    )
    if not rows:
        raise HTTPException(404, f'Variable "{variable}" not found or has no realized values')
    return rows


# ── /api/forecast/{variable}/{horizon} ────────────────────────────────────────
@app.get('/api/forecast/{variable}/{horizon}')
def forecast(variable: str, horizon: int):
    """
    Point forecasts — returns empty lists if forecast files
    haven't been loaded into the DB yet.
    """
    return {'dfm': [], 'bay_med': [], 'bay_p5': [], 'bay_p95': []}


# ── /api/spaghetti/{variable}/{horizon} ───────────────────────────────────────
@app.get('/api/spaghetti/{variable}/{horizon}')
def spaghetti(
    variable: str,
    horizon:  int,
    freq:     int = Query(default=1, ge=1, le=60,
                          description='Origin frequency in years (1 or 5)')
):
    """
    Full spaghetti backtest data for one variable × horizon.
    Origins are filtered to every `freq` years (default: every year).
    Returns nested {origins: [{origin_date, steps: [{...draws}]}]}.
    """
    # Stats and draws in one joined query
    rows = db_rows(
        '''
        SELECT s.origin_date, s.forecast_date,
               s.mean, s.p5, s.p50, s.p95, s.realized,
               d.draws_json
        FROM   stats s
        LEFT JOIN draws d
               ON  s.horizon       = d.horizon
               AND s.origin_date   = d.origin_date
               AND s.forecast_date = d.forecast_date
               AND s.variable      = d.variable
        WHERE  s.variable = ? AND s.horizon = ?
        ORDER  BY s.origin_date, s.forecast_date
        ''',
        (variable, horizon)
    )

    if not rows:
        raise HTTPException(
            404,
            f'No backtest data for variable="{variable}" horizon={horizon}. '
            'Run dfm_backtest.m then build_db.py first.'
        )

    # Group by origin_date
    from collections import defaultdict
    origin_map: dict[str, list] = defaultdict(list)

    for r in rows:
        od = r['origin_date']
        # Filter by frequency: keep only origins whose year is divisible by freq
        if freq > 1:
            try:
                year = int(od[:4])
                if year % freq != 0:
                    continue
            except (ValueError, TypeError):
                pass

        draws = json.loads(r['draws_json']) if r['draws_json'] else []

        origin_map[od].append({
            'forecast_date': r['forecast_date'],
            'mean':          r['mean'],
            'p5':            r['p5'],
            'p50':           r['p50'],
            'p95':           r['p95'],
            'realized':      r['realized'],
            'draws':         draws
        })

    # Infer n_draws from first row that has draws
    n_draws = 0
    for steps in origin_map.values():
        for step in steps:
            if step['draws']:
                n_draws = len(step['draws'])
                break
        if n_draws:
            break

    origins = [
        {'origin_date': od, 'steps': steps}
        for od, steps in sorted(origin_map.items())
    ]

    return {
        'variable': variable,
        'horizon':  horizon,
        'n_draws':  n_draws,
        'origins':  origins
    }


# ── /api/coverage/{variable}/{horizon} ────────────────────────────────────────
@app.get('/api/coverage/{variable}/{horizon}')
def coverage(variable: str, horizon: int):
    """
    For each (origin, forecast_date), whether the realized value fell
    inside the p5–p95 interval.  Used for coverage diagnostics.
    """
    rows = db_rows(
        '''SELECT origin_date, forecast_date, p5, p95, realized
           FROM stats
           WHERE variable=? AND horizon=? AND realized IS NOT NULL
           ORDER BY origin_date, forecast_date''',
        (variable, horizon)
    )
    result = []
    for r in rows:
        in_band = None
        if r['p5'] is not None and r['p95'] is not None and r['realized'] is not None:
            in_band = r['p5'] <= r['realized'] <= r['p95']
        result.append({
            'origin_date':   r['origin_date'],
            'forecast_date': r['forecast_date'],
            'in_band':       in_band
        })
    total   = sum(1 for r in result if r['in_band'] is not None)
    covered = sum(1 for r in result if r['in_band'])
    return {
        'data':          result,
        'coverage_rate': round(covered / total, 4) if total else None,
        'n':             total
    }


# ── Static files (serve dashboard) ────────────────────────────────────────────
# Mount last so /api/* routes take priority
static_dir = Path(ARGS.static).resolve()
if static_dir.exists():
    app.mount('/', StaticFiles(directory=str(static_dir), html=True), name='static')
else:
    print(f'WARNING: static directory not found: {static_dir}')


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    print(f'DB      : {os.path.abspath(ARGS.db)}')
    print(f'Static  : {static_dir}')
    print(f'Docs    : http://localhost:{ARGS.port}/api/docs')
    print(f'App     : http://localhost:{ARGS.port}/')
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level='info')
