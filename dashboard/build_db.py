#!/usr/bin/env python3
"""
build_db.py  —  zero external dependencies, Python 3.6+
Reads dfm_backtest.m CSV output and writes backtest.db (SQLite).

USAGE
  python3 build_db.py
  python3 build_db.py --backtest backtest_out/ --db backtest.db
  python3 build_db.py --horizons 1 12
  python3 build_db.py --rebuild
"""

import csv
import json
import math
import os
import sqlite3
import argparse
from datetime import datetime

HORIZONS = [1, 3, 12, 24, 60]

DB_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS stats (
    horizon       INTEGER NOT NULL,
    origin_date   TEXT    NOT NULL,
    forecast_date TEXT    NOT NULL,
    variable      TEXT    NOT NULL,
    mean          REAL,
    p5            REAL,
    p50           REAL,
    p95           REAL,
    realized      REAL,
    PRIMARY KEY (horizon, origin_date, forecast_date, variable)
);

CREATE TABLE IF NOT EXISTS draws (
    horizon       INTEGER NOT NULL,
    origin_date   TEXT    NOT NULL,
    forecast_date TEXT    NOT NULL,
    variable      TEXT    NOT NULL,
    draws_json    TEXT    NOT NULL,
    PRIMARY KEY (horizon, origin_date, forecast_date, variable)
);

CREATE INDEX IF NOT EXISTS idx_stats_var ON stats (variable, horizon);
CREATE INDEX IF NOT EXISTS idx_draws_var ON draws (variable, horizon);
"""


def parse_date(s):
    """Convert M/D/YYYY or YYYY-MM-DD to YYYY-MM. Returns original string on failure."""
    if not s:
        return s
    s = s.strip()
    try:
        if '/' in s:
            parts = s.split('/')
            if len(parts) == 3:
                m, d, y = int(parts[0]), int(parts[1]), int(parts[2])
                return '{:04d}-{:02d}'.format(y, m)
        for fmt in ('%Y-%m-%d', '%Y-%m'):
            try:
                dt = datetime.strptime(s, fmt)
                return '{:04d}-{:02d}'.format(dt.year, dt.month)
            except ValueError:
                continue
    except (ValueError, IndexError):
        pass
    return s


def safe_float(x):
    """Return float or None — never NaN/Inf."""
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return None


def read_stats(path, horizon):
    """Read backtest_h##_stats.csv → list of tuples."""
    rows = []
    with open(path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            od  = parse_date(row.get('origin_date', ''))
            fd  = parse_date(row.get('forecast_date', ''))
            var = row.get('variable', '').strip()
            if not var:
                continue
            rows.append((
                horizon, od, fd, var,
                safe_float(row.get('mean')),
                safe_float(row.get('p5')),
                safe_float(row.get('p50')),
                safe_float(row.get('p95')),
                safe_float(row.get('realized'))
            ))
    return rows


def read_draws(path, horizon):
    """Read backtest_h##_draws.csv → list of tuples with draws_json."""
    rows = []
    with open(path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        draw_cols = sorted(
            c for c in reader.fieldnames
            if len(c) >= 2 and c[0] == 'd' and c[1:].isdigit()
        )
        if not draw_cols:
            print('  WARN: no d### columns in {}'.format(path))
            return []
        for row in reader:
            od  = parse_date(row.get('origin_date', ''))
            fd  = parse_date(row.get('forecast_date', ''))
            var = row.get('variable', '').strip()
            if not var:
                continue
            draws = [safe_float(row[c]) for c in draw_cols]
            rows.append((
                horizon, od, fd, var,
                json.dumps(draws, separators=(',', ':'))
            ))
    return rows


def bulk_insert(con, table, rows, chunk=50000):
    if not rows:
        return 0
    placeh = ','.join(['?'] * len(rows[0]))
    sql    = 'INSERT OR REPLACE INTO {} VALUES ({})'.format(table, placeh)
    total  = 0
    for i in range(0, len(rows), chunk):
        con.executemany(sql, rows[i:i + chunk])
        con.commit()
        total += len(rows[i:i + chunk])
    return total


def parse_args():
    p = argparse.ArgumentParser(description='Build SQLite DB from dfm_backtest.m output')
    p.add_argument('--backtest',  default='backtest_out/', help='Backtest CSV directory')
    p.add_argument('--db',        default='backtest.db',   help='Output SQLite file')
    p.add_argument('--horizons',  nargs='+', type=int, default=HORIZONS)
    p.add_argument('--rebuild',   action='store_true',     help='Delete and recreate DB')
    return p.parse_args()


def main():
    args = parse_args()

    if args.rebuild and os.path.exists(args.db):
        os.remove(args.db)
        print('Removed existing DB: {}'.format(args.db))

    print('DB : {}'.format(os.path.abspath(args.db)))
    con = sqlite3.connect(args.db)
    con.executescript(DB_SCHEMA)
    con.commit()
    sep = '-' * 60

    # Stats
    print('\n{}\nBacktest stats\n{}'.format(sep, sep))
    for h in args.horizons:
        path = os.path.join(args.backtest, 'backtest_h{:02d}_stats.csv'.format(h))
        if not os.path.exists(path):
            print('  SKIP (not found): {}'.format(path))
            continue
        print('  Loading {} ...'.format(os.path.basename(path)), end=' ', flush=True)
        rows = read_stats(path, h)
        n = bulk_insert(con, 'stats', rows)
        print('{:,} rows'.format(n))

    # Draws
    print('\n{}\nBacktest draws\n{}'.format(sep, sep))
    for h in args.horizons:
        path = os.path.join(args.backtest, 'backtest_h{:02d}_draws.csv'.format(h))
        if not os.path.exists(path):
            print('  SKIP (not found): {}'.format(path))
            continue
        print('  Loading {} ...'.format(os.path.basename(path)), end=' ', flush=True)
        rows = read_draws(path, h)
        n = bulk_insert(con, 'draws', rows)
        print('{:,} rows'.format(n))

    # Summary
    print('\n{}\nSummary\n{}'.format(sep, sep))
    for tbl in ('stats', 'draws'):
        cnt = con.execute('SELECT COUNT(*) FROM {}'.format(tbl)).fetchone()[0]
        print('  {:8s}  {:>12,} rows'.format(tbl, cnt))
    size_mb = os.path.getsize(args.db) / 1048576
    print('\n  File: {}  ({:.1f} MB)'.format(args.db, size_mb))
    con.close()
    print('  Done.\n')


if __name__ == '__main__':
    main()
