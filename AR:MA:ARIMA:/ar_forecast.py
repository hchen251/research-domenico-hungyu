"""
Autoregressive (AR) Model Forecasting
=====================================
AR(p) model: y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t

Modes:
- Forecast: Train on all data, predict future periods
- Backtest: Train on data excluding holdout, test on holdout
"""

import sys
import io
import warnings
import time
import argparse
from typing import Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import pacf

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)


def log(msg: str):
    print(msg, flush=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(path: str, has_tcodes: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV data."""
    raw = pd.read_csv(path)
    date_col = raw.columns[0]
    
    if has_tcodes:
        tcodes = pd.Series(index=raw.columns[1:], dtype=float)
        for col in raw.columns[1:]:
            try:
                tcodes[col] = float(raw.iloc[0][col])
            except:
                tcodes[col] = 1.0
        df = raw.iloc[1:].copy().reset_index(drop=True)
    else:
        df = raw.copy()
        tcodes = pd.Series(1.0, index=raw.columns[1:])
    
    for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d"]:
        try:
            df[date_col] = pd.to_datetime(df[date_col], format=fmt)
            break
        except:
            continue
    
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    return df, tcodes


def apply_tcode(x: pd.Series, tcode: int) -> pd.Series:
    """Apply FRED-MD transformation to make series stationary."""
    x = x.astype(float)
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
    return x


def invert_tcode(fc: np.ndarray, hist: pd.Series, tcode: int) -> np.ndarray:
    """Invert transformation to get levels."""
    n = len(fc)
    h = hist.dropna()
    
    if h.empty:
        return np.full(n, np.nan)
    
    last = float(h.iloc[-1])
    result = np.empty(n)
    
    if tcode == 1:
        return fc.copy()
    elif tcode == 2:
        prev = last
        for i in range(n):
            if np.isnan(fc[i]):
                result[i] = np.nan
            else:
                prev += fc[i]
                result[i] = prev
    elif tcode == 3:
        if len(h) < 2:
            return np.full(n, np.nan)
        d = last - float(h.iloc[-2])
        x = last
        for i in range(n):
            if np.isnan(fc[i]):
                result[i] = np.nan
            else:
                d += fc[i]
                x += d
                result[i] = x
    elif tcode == 4:
        return np.exp(fc)
    elif tcode == 5:
        if last <= 0:
            return np.full(n, np.nan)
        lp = np.log(last)
        for i in range(n):
            if np.isnan(fc[i]):
                result[i] = np.nan
            else:
                lp += fc[i]
                result[i] = np.exp(lp)
    elif tcode == 6:
        if len(h) < 2 or last <= 0 or float(h.iloc[-2]) <= 0:
            return np.full(n, np.nan)
        dl = np.log(last) - np.log(float(h.iloc[-2]))
        lp = np.log(last)
        for i in range(n):
            if np.isnan(fc[i]):
                result[i] = np.nan
            else:
                dl += fc[i]
                lp += dl
                result[i] = np.exp(lp)
    elif tcode == 7:
        if len(h) < 2 or h.iloc[-2] == 0:
            return np.full(n, np.nan)
        g = (last - float(h.iloc[-2])) / float(h.iloc[-2])
        x = last
        for i in range(n):
            if np.isnan(fc[i]):
                result[i] = np.nan
            else:
                g += fc[i]
                x = x * (1 + g)
                result[i] = x
    else:
        return fc.copy()
    
    return result


# =============================================================================
# AR MODEL
# =============================================================================

def select_ar_order(y: pd.Series, max_lags: int = 12) -> int:
    """Select optimal AR order using PACF."""
    y_clean = y.dropna()
    if len(y_clean) < max_lags + 10:
        return 1
    
    try:
        pacf_vals = pacf(y_clean, nlags=max_lags, method='ywm')
        threshold = 1.96 / np.sqrt(len(y_clean))
        
        for lag in range(1, len(pacf_vals)):
            if abs(pacf_vals[lag]) < threshold:
                return max(1, lag - 1) if lag > 1 else 1
        
        return max_lags
    except:
        return 1


def forecast_ar_series(
    y: pd.Series,
    horizon: int,
    ar_lags: int = None,
    max_lags: int = 12
) -> np.ndarray:
    """Forecast a single series using AR model."""
    y_clean = y.dropna()
    
    if len(y_clean) < 20:
        return np.full(horizon, np.nan)
    
    if ar_lags is None:
        ar_lags = select_ar_order(y_clean, max_lags)
    
    ar_lags = max(1, min(ar_lags, len(y_clean) // 3))
    
    try:
        model = AutoReg(y_clean, lags=ar_lags, old_names=False)
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)
        return forecast.values
    except:
        return np.full(horizon, np.nan)


# =============================================================================
# FORECAST MODE (No holdout - predict future)
# =============================================================================

def run_ar_forecast(
    levels: pd.DataFrame,
    tcodes: pd.Series,
    horizon: int,
    ar_lags: int,
    verbose: bool
) -> pd.DataFrame:
    """
    Run AR model forecast (predict future periods).
    
    Uses ALL data for training, then forecasts 'horizon' periods into the future.
    """
    
    all_cols = list(levels.columns)
    n_series = len(all_cols)
    
    if verbose:
        log(f"\n{'='*60}")
        log("AR MODEL - FORECAST MODE")
        log(f"{'='*60}")
        log(f"  Training data:    {levels.index[0].strftime('%Y-%m')} to {levels.index[-1].strftime('%Y-%m')}")
        log(f"  Training periods: {len(levels)}")
        log(f"  Forecast horizon: {horizon} periods")
        log(f"  AR lags:          {'Auto' if ar_lags is None else ar_lags}")
        log(f"  Number of series: {n_series}")
        log(f"{'='*60}\n")
    
    # Transform all data
    transformed = pd.DataFrame(index=levels.index)
    for col in all_cols:
        tc = tcodes.get(col, 1)
        if pd.isna(tc):
            tc = 1
        transformed[col] = apply_tcode(levels[col], int(tc))
    
    # Generate forecast dates
    last_date = levels.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq='MS'
    )
    
    # Forecast each series
    forecasts = pd.DataFrame(index=forecast_dates)
    successful = 0
    failed = 0
    
    for i, col in enumerate(all_cols):
        if verbose and (i % 20 == 0 or i == n_series - 1):
            log(f"  Processing {i+1}/{n_series}: {col}")
        
        try:
            y = transformed[col].dropna()
            
            if len(y) < 20:
                forecasts[col] = np.nan
                failed += 1
                continue
            
            fc_transformed = forecast_ar_series(y, horizon, ar_lags)
            
            tc = tcodes.get(col, 1)
            if pd.isna(tc):
                tc = 1
            fc_levels = invert_tcode(fc_transformed, levels[col], int(tc))
            
            forecasts[col] = fc_levels
            successful += 1
            
        except:
            forecasts[col] = np.nan
            failed += 1
    
    if verbose:
        log(f"\n  Successful: {successful}, Failed: {failed}")
    
    # Format output
    df = forecasts.reset_index()
    df = df.rename(columns={df.columns[0]: 'sasdate'})
    
    try:
        df['sasdate'] = pd.to_datetime(df['sasdate']).dt.strftime('%-m/%-d/%Y')
    except ValueError:
        df['sasdate'] = pd.to_datetime(df['sasdate']).dt.strftime('%#m/%#d/%Y')
    
    return df


# =============================================================================
# BACKTEST MODE (With holdout - evaluate on historical data)
# =============================================================================

def run_ar_backtest(
    levels: pd.DataFrame,
    tcodes: pd.Series,
    holdout: int,
    ar_lags: int,
    verbose: bool
) -> pd.DataFrame:
    """
    Run AR model backtest.
    
    Splits data into training and holdout, trains on training data,
    and evaluates forecasts against actual holdout values.
    """
    
    n_obs = len(levels)
    all_cols = list(levels.columns)
    n_series = len(all_cols)
    
    if n_obs <= holdout:
        raise ValueError(f"Not enough data. Have {n_obs}, need more than {holdout}")
    
    # Split data
    train_levels = levels.iloc[:-holdout]
    test_dates = levels.index[-holdout:]
    
    if verbose:
        log(f"\n{'='*60}")
        log("AR MODEL - BACKTEST MODE")
        log(f"{'='*60}")
        log(f"  Total observations: {n_obs}")
        log(f"  Training period:    {train_levels.index[0].strftime('%Y-%m')} to {train_levels.index[-1].strftime('%Y-%m')}")
        log(f"  Holdout period:     {test_dates[0].strftime('%Y-%m')} to {test_dates[-1].strftime('%Y-%m')}")
        log(f"  Training periods:   {len(train_levels)}")
        log(f"  Holdout periods:    {holdout}")
        log(f"  AR lags:            {'Auto' if ar_lags is None else ar_lags}")
        log(f"  Number of series:   {n_series}")
        log(f"{'='*60}\n")
    
    # Transform training data
    train_transformed = pd.DataFrame(index=train_levels.index)
    for col in all_cols:
        tc = tcodes.get(col, 1)
        if pd.isna(tc):
            tc = 1
        train_transformed[col] = apply_tcode(train_levels[col], int(tc))
    
    # Forecast each series
    forecasts = pd.DataFrame(index=test_dates)
    successful = 0
    failed = 0
    
    for i, col in enumerate(all_cols):
        if verbose and (i % 20 == 0 or i == n_series - 1):
            log(f"  Processing {i+1}/{n_series}: {col}")
        
        try:
            y = train_transformed[col].dropna()
            
            if len(y) < 20:
                forecasts[col] = np.nan
                failed += 1
                continue
            
            fc_transformed = forecast_ar_series(y, holdout, ar_lags)
            
            tc = tcodes.get(col, 1)
            if pd.isna(tc):
                tc = 1
            fc_levels = invert_tcode(fc_transformed, train_levels[col], int(tc))
            
            forecasts[col] = fc_levels
            successful += 1
            
        except:
            forecasts[col] = np.nan
            failed += 1
    
    if verbose:
        log(f"\n  Successful: {successful}, Failed: {failed}")
    
    # Format output
    df = forecasts.reset_index()
    df = df.rename(columns={df.columns[0]: 'sasdate'})
    
    try:
        df['sasdate'] = pd.to_datetime(df['sasdate']).dt.strftime('%-m/%-d/%Y')
    except ValueError:
        df['sasdate'] = pd.to_datetime(df['sasdate']).dt.strftime('%#m/%#d/%Y')
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AR Model Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  FORECAST: Use all data to predict future periods
  BACKTEST: Hold out data to evaluate model accuracy

Examples:
  # Forecast next 12 months (using all data for training)
  python ar_forecast.py --data data.csv --horizon 12 --output ar_forecast.csv
  
  # Backtest with 12-month holdout (evaluate on historical data)
  python ar_forecast.py --data data.csv --backtest 12 --output ar_backtest.csv
  
  # Backtest with specific AR order
  python ar_forecast.py --data data.csv --backtest 12 --ar_lags 4 --output ar_backtest.csv
        """
    )
    
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--no-tcodes", action="store_true", help="No transformation code row")
    parser.add_argument("--ar_lags", type=int, default=None, help="AR order (default: auto-select)")
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--horizon", type=int, help="Forecast mode: predict H periods ahead")
    mode_group.add_argument("--backtest", type=int, help="Backtest mode: holdout H periods")
    
    parser.add_argument("--output", type=str, default="ar_output.csv", help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    
    args = parser.parse_args()
    
    t0 = time.perf_counter()
    
    if args.verbose:
        log(f"\nLoading {args.data}...")
    
    levels, tcodes = load_data(args.data, has_tcodes=not args.no_tcodes)
    
    if args.verbose:
        log(f"Data: {len(levels)} rows x {len(levels.columns)} columns")
    
    # Run appropriate mode
    if args.horizon:
        result = run_ar_forecast(
            levels, tcodes,
            args.horizon, args.ar_lags,
            args.verbose
        )
    else:  # args.backtest
        result = run_ar_backtest(
            levels, tcodes,
            args.backtest, args.ar_lags,
            args.verbose
        )
    
    result.to_csv(args.output, index=False)
    
    elapsed = time.perf_counter() - t0
    log(f"\nSaved to {args.output} ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()