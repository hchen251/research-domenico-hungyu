"""
Dynamic Factor Model (state-space / Kalman filter, MLE) forecasting for FRED-MD vintages.

Backtesting:
  --backtest 4   Holds out last 4 months, trains on earlier data, forecasts 4 months.
                 Output contains only the forecasted values for the held-out period.

Outputs a CSV with forecasted LEVEL values only.

Usage examples:
  python dfm_forecast_v2.py --data 2014-12.csv --series "INDPRO,UNRATE" --k_factors 2 --factor_order 1 --horizon 12 --output out.csv
  python dfm_forecast_v2.py --data 2014-12.csv --series "INDPRO,UNRATE" --k_factors 2 --factor_order 1 --backtest 4 --output backtest.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

GROUP_MAP: Dict[int, str] = {
    1: "Output and Income",
    2: "Labor Market",
    3: "Housing",
    4: "Consumption, Orders, and Inventories",
    5: "Money and Credit",
    6: "Interest and Exchange Rates",
    7: "Prices",
    8: "Stock Market",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DFM forecasts for FRED-MD panels.")

    p.add_argument("--data", type=str, required=True, help="Path to FRED-MD CSV.")
    p.add_argument("--appendix", type=str, default=None, help="Path to appendix CSV.")

    sel = p.add_argument_group("series selection")
    sel.add_argument("--series", type=str, default=None, help="Comma-separated series codes.")
    sel.add_argument("--series_file", type=str, default=None, help="File with series codes.")
    sel.add_argument("--series_index", type=str, default=None, help="1-indexed positions, e.g. '1,2,10-20'.")
    sel.add_argument("--group", type=str, default=None, help="Group code(s) or name(s).")
    sel.add_argument("--all_series", action="store_true", help="Use all series.")

    util = p.add_argument_group("utilities")
    util.add_argument("--list_series", action="store_true", help="Print series and exit.")
    util.add_argument("--list_groups", action="store_true", help="Print groups and exit.")

    model = p.add_argument_group("model")
    model.add_argument("--k_factors", type=int, default=3, help="Number of factors.")
    model.add_argument("--factor_order", type=int, default=1, help="Factor AR order.")
    model.add_argument("--error_cov_type", type=str, default="diagonal", choices=["diagonal", "unstructured"])
    model.add_argument("--error_order", type=int, default=0, help="Idiosyncratic AR order.")

    out = p.add_argument_group("output")
    out.add_argument("--horizon", type=int, default=12, help="Forecast horizon.")
    out.add_argument("--backtest", type=int, default=None, help="Hold out N months for backtesting.")
    out.add_argument("--include_history", action="store_true", help="Include last actual row.")
    out.add_argument("--output", type=str, default="dfm_forecast_output.csv", help="Output path.")
    out.add_argument("--output_format", type=str, default="wide", choices=["wide", "long"])
    out.add_argument("--date_col_name", type=str, default="sasdate")
    out.add_argument("--omit_value_suffix", action="store_true")
    out.add_argument("--verbose", action="store_true")
    out.add_argument("--progress_every", type=int, default=25)

    opt = p.add_argument_group("optimization")
    opt.add_argument("--maxiter", type=int, default=500)
    opt.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs", "powell", "bfgs", "nm"])
    opt.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def read_panel_with_tcodes(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Read FRED-MD CSV with transformation codes in first row."""
    raw = pd.read_csv(data_path)

    if raw.shape[0] < 3:
        raise ValueError("Data file too short.")

    date_col = raw.columns[0]
    first_val = str(raw.iloc[0][date_col]).strip().lower()

    if "transform" not in first_val:
        raise ValueError(f"Expected 'Transform' in first row, got '{raw.iloc[0][date_col]}'")

    # Extract tcodes from first row
    tcode_row = raw.iloc[0, 1:]  # Skip date column
    tcodes = pd.Series(index=raw.columns[1:], dtype=float)
    for col in raw.columns[1:]:
        val = raw.iloc[0][col]
        try:
            tcodes[col] = float(val)
        except (ValueError, TypeError):
            tcodes[col] = np.nan

    # Extract data (skip tcode row)
    df = raw.iloc[1:].copy().reset_index(drop=True)

    # Parse dates
    date_formats = ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d"]
    parsed = None
    for fmt in date_formats:
        try:
            parsed = pd.to_datetime(df[date_col], format=fmt, errors="raise")
            break
        except (ValueError, TypeError):
            continue

    if parsed is None:
        parsed = pd.to_datetime(df[date_col], errors="coerce")

    df[date_col] = parsed
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()

    # Convert to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Set frequency
    try:
        df = df.asfreq("MS")
    except Exception:
        pass

    return df, tcodes


def read_appendix(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def parse_groups(group_arg: str) -> List[int]:
    parts = [p.strip() for p in (group_arg or "").split(",") if p.strip()]
    out: List[int] = []

    for p in parts:
        if p.isdigit():
            g = int(p)
            if g not in GROUP_MAP:
                raise ValueError(f"Unknown group {g}. Valid: {sorted(GROUP_MAP)}")
            out.append(g)
        else:
            p_low = p.lower()
            matches = [k for k, v in GROUP_MAP.items() if p_low in v.lower()]
            if matches:
                out.extend(matches)
            else:
                raise ValueError(f"Could not match group '{p}'.")

    return sorted(set(out))


def select_series_by_groups(appendix: pd.DataFrame, groups: List[int], available: List[str]) -> List[str]:
    if "fred" not in appendix.columns or "group" not in appendix.columns:
        raise ValueError("Appendix must have 'fred' and 'group' columns.")

    subset = appendix[appendix["group"].isin(groups)]
    candidates = subset["fred"].astype(str).tolist()
    cols = [c for c in candidates if c in available]
    if not cols:
        raise ValueError("No matching series for groups.")
    return cols


def parse_index_spec(spec: str, n_cols: int) -> List[int]:
    out: List[int] = []
    parts = [p.strip() for p in (spec or "").split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = int(a), int(b)
            if a < 1 or b < 1:
                raise ValueError("Indices must be >= 1.")
            if a > b:
                a, b = b, a
            out.extend(range(a - 1, b))
        else:
            i = int(p)
            if i < 1:
                raise ValueError("Indices must be >= 1.")
            out.append(i - 1)

    out = sorted(set(out))
    if not out:
        raise ValueError("Empty index selection.")
    if out[-1] >= n_cols:
        raise ValueError(f"Index out of bounds. Max: {n_cols}.")

    return out


def apply_tcode_transform(x: pd.Series, tcode: int) -> pd.Series:
    """
    Apply FRED-MD transformation.
    
    1: level (no transform)
    2: first difference
    3: second difference
    4: log
    5: first difference of log
    6: second difference of log
    7: first difference of percent change
    """
    x = x.astype(float).copy()

    if tcode == 1:
        return x
    elif tcode == 2:
        return x.diff()
    elif tcode == 3:
        return x.diff().diff()
    elif tcode == 4:
        return np.log(x.where(x > 0, np.nan))
    elif tcode == 5:
        return np.log(x.where(x > 0, np.nan)).diff()
    elif tcode == 6:
        return np.log(x.where(x > 0, np.nan)).diff().diff()
    elif tcode == 7:
        return x.pct_change().diff()
    else:
        raise ValueError(f"Invalid tcode: {tcode}")


def transform_panel(
    levels: pd.DataFrame,
    tcodes: pd.Series,
    verbose: bool = False,
    progress_every: int = 25,
) -> Tuple[pd.DataFrame, List[str]]:
    """Transform panel and return (transformed_df, valid_columns)."""
    result = {}
    valid_cols = []
    n = len(levels.columns)

    for i, c in enumerate(levels.columns):
        if c not in tcodes.index:
            if verbose:
                print(f"  Warning: '{c}' not in tcodes, skipping.", flush=True)
            continue

        tc = tcodes[c]
        if pd.isna(tc):
            if verbose:
                print(f"  Warning: tcode for '{c}' is NaN, skipping.", flush=True)
            continue

        result[c] = apply_tcode_transform(levels[c], int(tc))
        valid_cols.append(c)

        if verbose and ((i + 1) % progress_every == 0 or i == n - 1):
            print(f"[transform] {i + 1}/{n}", flush=True)

    return pd.DataFrame(result, index=levels.index), valid_cols


def standardize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    mu = df.mean(skipna=True)
    sigma = df.std(skipna=True, ddof=0)
    sigma = sigma.replace(0.0, 1.0).fillna(1.0)
    z = (df - mu) / sigma
    return z, mu, sigma


def unstandardize_series(z: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return z * sigma + mu


def fit_dfm(
    endog: pd.DataFrame,
    k_factors: int,
    factor_order: int,
    error_order: int,
    error_cov_type: str,
    maxiter: int,
    optimizer: str,
    seed: int,
) -> Tuple:
    """Fit DFM and return (result, kept_columns)."""
    np.random.seed(seed)

    # Drop all-NaN columns
    keep = [c for c in endog.columns if endog[c].notna().any()]
    endog_clean = endog[keep].copy()

    if len(keep) == 0:
        raise ValueError("All series are NaN.")
    if len(keep) < k_factors:
        raise ValueError(f"Only {len(keep)} series but k_factors={k_factors}.")

    model = DynamicFactor(
        endog=endog_clean,
        k_factors=k_factors,
        factor_order=factor_order,
        error_order=error_order,
        error_cov_type=error_cov_type,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(method=optimizer, maxiter=maxiter, disp=True)

    return res, keep


def build_forecast_index(last_date: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    """Create monthly index starting after last_date."""
    # Ensure we start from first of next month
    year, month = last_date.year, last_date.month
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1
    start = pd.Timestamp(year=year, month=month, day=1)
    return pd.date_range(start=start, periods=steps, freq="MS")


def invert_tcode(
    fc: np.ndarray,
    level_hist: pd.Series,
    tcode: int,
) -> np.ndarray:
    """
    Invert transformation to get levels.
    
    fc: forecast values in transformed space
    level_hist: historical level data
    tcode: transformation code
    """
    n = len(fc)
    if n == 0:
        return np.array([])

    hist = level_hist.dropna()
    if hist.empty:
        return np.full(n, np.nan)

    last = float(hist.iloc[-1])
    result = np.empty(n)

    # tcode 1: no transform
    if tcode == 1:
        return fc.copy()

    # tcode 2: first difference
    # x_t = x_{t-1} + d_t
    elif tcode == 2:
        prev = last
        for i in range(n):
            if np.isnan(fc[i]):
                result[i] = np.nan
            else:
                prev = prev + fc[i]
                result[i] = prev
        return result

    # tcode 3: second difference
    # d_t = d_{t-1} + dd_t, x_t = x_{t-1} + d_t
    elif tcode == 3:
        if len(hist) < 2:
            return np.full(n, np.nan)
        prev2 = float(hist.iloc[-2])
        last_d = last - prev2
        d_prev = last_d
        x_prev = last
        for i in range(n):
            if np.isnan(fc[i]):
                result[i] = np.nan
            else:
                d_prev = d_prev + fc[i]
                x_prev = x_prev + d_prev
                result[i] = x_prev
        return result

    # tcode 4: log
    # x_t = exp(fc_t)
    elif tcode == 4:
        return np.exp(fc)

    # tcode 5: first difference of log
    # log(x_t) = log(x_{t-1}) + dlog_t
    elif tcode == 5:
        if last <= 0:
            return np.full(n, np.nan)
        log_prev = np.log(last)
        for i in range(n):
            if np.isnan(fc[i]):
                result[i] = np.nan
            else:
                log_prev = log_prev + fc[i]
                result[i] = np.exp(log_prev)
        return result

    # tcode 6: second difference of log
    # dlog_t = dlog_{t-1} + ddlog_t
    # log(x_t) = log(x_{t-1}) + dlog_t
    elif tcode == 6:
        if len(hist) < 2 or last <= 0:
            return np.full(n, np.nan)
        prev2 = float(hist.iloc[-2])
        if prev2 <= 0:
            return np.full(n, np.nan)
        last_dlog = np.log(last) - np.log(prev2)
        dlog_prev = last_dlog
        log_prev = np.log(last)
        for i in range(n):
            if np.isnan(fc[i]):
                result[i] = np.nan
            else:
                dlog_prev = dlog_prev + fc[i]
                log_prev = log_prev + dlog_prev
                result[i] = np.exp(log_prev)
        return result

    # tcode 7: first difference of percent change
    # pct_t = pct_{t-1} + dpct_t
    # x_t = x_{t-1} * (1 + pct_t)
    elif tcode == 7:
        if len(hist) < 2:
            return np.full(n, np.nan)
        prev2 = float(hist.iloc[-2])
        if prev2 == 0:
            return np.full(n, np.nan)
        last_pct = (last - prev2) / prev2
        pct_prev = last_pct
        x_prev = last
        for i in range(n):
            if np.isnan(fc[i]):
                result[i] = np.nan
            else:
                pct_prev = pct_prev + fc[i]
                x_prev = x_prev * (1.0 + pct_prev)
                result[i] = x_prev
        return result

    else:
        return np.full(n, np.nan)


def forecast_to_levels(
    fcast_transformed: pd.DataFrame,
    levels: pd.DataFrame,
    tcodes: pd.Series,
    verbose: bool = False,
    progress_every: int = 25,
) -> pd.DataFrame:
    """Convert forecasts from transformed space to levels."""
    result = {}
    cols = list(fcast_transformed.columns)
    n = len(cols)

    for i, c in enumerate(cols):
        if c not in levels.columns or c not in tcodes.index:
            result[c] = np.full(len(fcast_transformed), np.nan)
            continue

        tc = tcodes[c]
        if pd.isna(tc):
            result[c] = np.full(len(fcast_transformed), np.nan)
            continue

        fc_vals = fcast_transformed[c].values
        level_hist = levels[c]
        result[c] = invert_tcode(fc_vals, level_hist, int(tc))

        if verbose and ((i + 1) % progress_every == 0 or i == n - 1):
            print(f"[invert] {i + 1}/{n}", flush=True)

    return pd.DataFrame(result, index=fcast_transformed.index)


def format_output(
    df: pd.DataFrame,
    output_format: str,
    include_history: bool,
    last_date: Optional[pd.Timestamp],
    last_levels: Optional[pd.Series],
    omit_suffix: bool,
    date_col: str,
) -> pd.DataFrame:
    out = df.copy()

    # Add history row if requested
    if include_history and last_date is not None and last_levels is not None:
        common = [c for c in out.columns if c in last_levels.index]
        if common:
            hist = pd.DataFrame([last_levels[common]], index=[last_date], columns=common)
            out = pd.concat([hist, out], axis=0)

    if output_format == "wide":
        if not omit_suffix:
            out.columns = [f"{c}_value" for c in out.columns]
        out.index.name = date_col
        return out.reset_index()
    else:
        out = out.reset_index()
        out = out.melt(id_vars=[out.columns[0]], var_name="series", value_name="forecast_value")
        out.columns = [date_col, "series", "forecast_value"]
        return out


def run_forecast(
    levels: pd.DataFrame,
    tcodes: pd.Series,
    k_factors: int,
    factor_order: int,
    error_order: int,
    error_cov_type: str,
    maxiter: int,
    optimizer: str,
    seed: int,
    horizon: int,
    verbose: bool,
    progress_every: int,
) -> pd.DataFrame:
    """Main forecasting pipeline."""

    # Step 1: Transform
    if verbose:
        print("Step 1/5: Transforming...", flush=True)
    t0 = time.perf_counter()
    transformed, valid_cols = transform_panel(levels, tcodes, verbose, progress_every)
    if verbose:
        print(f"  Done ({time.perf_counter() - t0:.1f}s). {len(valid_cols)} series.", flush=True)

    # Remove all-NaN rows
    transformed = transformed.dropna(how="all")
    if transformed.empty:
        raise ValueError("All data is NaN after transformation.")

    # Step 2: Standardize
    if verbose:
        print("Step 2/5: Standardizing...", flush=True)
    t0 = time.perf_counter()
    std_data, mu, sigma = standardize(transformed)
    if verbose:
        print(f"  Done ({time.perf_counter() - t0:.1f}s).", flush=True)

    # Step 3: Fit model
    if verbose:
        print(f"Step 3/5: Fitting DFM (k={k_factors}, order={factor_order})...", flush=True)
    t0 = time.perf_counter()
    res, keep_cols = fit_dfm(
        std_data, k_factors, factor_order, error_order,
        error_cov_type, maxiter, optimizer, seed
    )
    if verbose:
        print(f"  Done ({time.perf_counter() - t0:.1f}s). {len(keep_cols)} series kept.", flush=True)

    print(res.summary())

    # Step 4: Forecast
    if verbose:
        print("Step 4/5: Forecasting...", flush=True)
    t0 = time.perf_counter()
    fcast = res.get_forecast(steps=horizon)
    fcast_std = fcast.predicted_mean
    if verbose:
        print(f"  Done ({time.perf_counter() - t0:.1f}s).", flush=True)

    # Ensure fcast_std is DataFrame with correct columns
    if isinstance(fcast_std, pd.Series):
        fcast_std = fcast_std.to_frame()
    if not isinstance(fcast_std, pd.DataFrame):
        fcast_std = pd.DataFrame(fcast_std)
    
    # Verify and set column names
    if len(fcast_std.columns) != len(keep_cols):
        raise ValueError(f"Forecast has {len(fcast_std.columns)} columns but expected {len(keep_cols)}")
    fcast_std.columns = keep_cols

    # Unstandardize
    fcast_trans = pd.DataFrame(index=fcast_std.index)
    for c in keep_cols:
        fcast_trans[c] = unstandardize_series(fcast_std[c].values, mu[c], sigma[c])

    # Set index
    last_date = transformed.index[-1]
    fcast_trans.index = build_forecast_index(last_date, horizon)

    # Step 5: Invert to levels
    if verbose:
        print("Step 5/5: Converting to levels...", flush=True)
    t0 = time.perf_counter()
    
    # Only use columns we have
    levels_sub = levels[[c for c in keep_cols if c in levels.columns]]
    tcodes_sub = tcodes[keep_cols]
    
    levels_fcst = forecast_to_levels(fcast_trans, levels_sub, tcodes_sub, verbose, progress_every)
    if verbose:
        print(f"  Done ({time.perf_counter() - t0:.1f}s).", flush=True)

    return levels_fcst


def main() -> None:
    args = parse_args()
    levels_all, tcodes_all = read_panel_with_tcodes(args.data)

    if args.list_groups:
        for k, v in sorted(GROUP_MAP.items()):
            print(f"{k}: {v}")
        sys.exit(0)

    if args.list_series:
        for i, c in enumerate(levels_all.columns, 1):
            print(f"{i:3d}  {c}")
        print(f"\nTotal: {len(levels_all.columns)}")
        sys.exit(0)

    avail = list(levels_all.columns)

    # Select series
    if args.all_series:
        series = avail
    elif args.series_file:
        with open(args.series_file) as f:
            series = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        bad = [s for s in series if s not in avail]
        if bad:
            raise ValueError(f"Not found: {bad[:5]}")
    elif args.series:
        series = [s.strip() for s in args.series.split(",") if s.strip()]
        bad = [s for s in series if s not in avail]
        if bad:
            raise ValueError(f"Not found: {bad}")
    elif args.series_index:
        idxs = parse_index_spec(args.series_index, len(avail))
        series = [avail[i] for i in idxs]
    elif args.group:
        if not args.appendix:
            raise ValueError("--appendix required with --group")
        appendix = read_appendix(args.appendix)
        groups = parse_groups(args.group)
        series = select_series_by_groups(appendix, groups, avail)
    else:
        raise ValueError("Specify --series, --series_index, --group, or --all_series")

    levels = levels_all[series].copy()
    tcodes = tcodes_all[series].copy()

    if args.verbose:
        print(f"Selected {len(series)} series.", flush=True)

    # Backtest mode
    if args.backtest is not None:
        h = args.backtest
        if h < 1 or h >= len(levels):
            raise ValueError(f"Invalid --backtest {h}")

        if args.verbose:
            print(f"\n=== BACKTEST: {h} months ===\n", flush=True)

        train = levels.iloc[:-h]
        holdout_idx = levels.index[-h:]

        if args.verbose:
            print(f"Train: {train.index[0]} - {train.index[-1]}", flush=True)
            print(f"Test:  {holdout_idx[0]} - {holdout_idx[-1]}", flush=True)

        fcst = run_forecast(
            train, tcodes, args.k_factors, args.factor_order,
            args.error_order, args.error_cov_type, args.maxiter,
            args.optimizer, args.seed, h, args.verbose, args.progress_every
        )

        # Align index
        if len(fcst) == len(holdout_idx):
            fcst.index = holdout_idx

        out = format_output(fcst, args.output_format, False, None, None,
                           args.omit_value_suffix, args.date_col_name)
        out.to_csv(args.output, index=False)
        print(f"Wrote: {args.output}")

    # Regular forecast
    else:
        fcst = run_forecast(
            levels, tcodes, args.k_factors, args.factor_order,
            args.error_order, args.error_cov_type, args.maxiter,
            args.optimizer, args.seed, args.horizon, args.verbose, args.progress_every
        )

        out = format_output(
            fcst, args.output_format, args.include_history,
            levels.index[-1] if args.include_history else None,
            levels.iloc[-1] if args.include_history else None,
            args.omit_value_suffix, args.date_col_name
        )
        out.to_csv(args.output, index=False)
        print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()