"""
ARIMA Model Forecasting
=======================
ARIMA(p,d,q): AutoRegressive Integrated Moving Average

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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

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
    """Apply FRED-MD transformation."""
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
# ARIMA MODEL
# =============================================================================

def select_arima_order(y: pd.Series, max_p: int = 3, max_q: int = 3) -> Tuple[int, int, int]:
    """Select ARIMA order using AIC."""
    y_clean = y.dropna()
    
    if len(y_clean) < 30:
        return (1, 0, 1)
    
    best_aic = np.inf
    best_order = (1, 0, 1)
    
    # For transformed data, d=0 since already stationary
    d = 0
    
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue
            
            try:
                model = ARIMA(y_clean, order=(p, d, q))
                fitted = model.fit()
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except:
                continue
    
    return best_order


def forecast_arima_series(
    y: pd.Series,
    horizon: int,
    order: Tuple[int, int, int] = None,
    auto_select: bool = True
) -> np.ndarray:
    """Forecast a single series using ARIMA model."""
    y_clean = y.dropna()
    
    if len(y_clean) < 20:
        return np.full(horizon, np.nan)
    
    if order is None and auto_select:
        order = select_arima_order(y_clean)
    elif order is None:
        order = (1, 0, 1)
    
    try:
        model = ARIMA(y_clean, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)
        return forecast.values
    except:
        try:
            model = ARIMA(y_clean, order=(0, 1, 0))
            fitted = model.fit()
            return fitted.forecast(steps=horizon).values
        except:
            return np.full(horizon, y_clean.iloc[-1])


# =============================================================================
# FORECAST MODE
# =============================================================================

def run_arima_forecast(
    levels: pd.DataFrame,
    tcodes: pd.Series,
    horizon: int,
    order: Tuple[int, int, int],
    auto_order: bool,
    verbose: bool
) -> pd.DataFrame:
    """Run ARIMA model forecast (predict future periods)."""
    
    all_cols = list(levels.columns)
    n_series = len(all_cols)
    
    if verbose:
        log(f"\n{'='*60}")
        log("ARIMA MODEL - FORECAST MODE")
        log(f"{'='*60}")
        log(f"  Training data:    {levels.index[0].strftime('%Y-%m')} to {levels.index[-1].strftime('%Y-%m')}")
        log(f"  Training periods: {len(levels)}")
        log(f"  Forecast horizon: {horizon} periods")
        log(f"  ARIMA order:      {'Auto-select' if auto_order else order}")
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
            
            if len(y) < 30:
                forecasts[col] = np.nan
                failed += 1
                continue
            
            if auto_order:
                use_order = select_arima_order(y)
            else:
                use_order = order
            
            fc_transformed = forecast_arima_series(y, horizon, order=use_order, auto_select=False)
            
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
    
    df = forecasts.reset_index()
    df = df.rename(columns={df.columns[0]: 'sasdate'})
    
    try:
        df['sasdate'] = pd.to_datetime(df['sasdate']).dt.strftime('%-m/%-d/%Y')
    except ValueError:
        df['sasdate'] = pd.to_datetime(df['sasdate']).dt.strftime('%#m/%#d/%Y')
    
    return df


# =============================================================================
# BACKTEST MODE
# =============================================================================

def run_arima_backtest(
    levels: pd.DataFrame,
    tcodes: pd.Series,
    holdout: int,
    order: Tuple[int, int, int],
    auto_order: bool,
    verbose: bool
) -> pd.DataFrame:
    """Run ARIMA model backtest."""
    
    n_obs = len(levels)
    all_cols = list(levels.columns)
    n_series = len(all_cols)
    
    if n_obs <= holdout:
        raise ValueError(f"Not enough data. Have {n_obs}, need more than {holdout}")
    
    train_levels = levels.iloc[:-holdout]
    test_dates = levels.index[-holdout:]
    
    if verbose:
        log(f"\n{'='*60}")
        log("ARIMA MODEL - BACKTEST MODE")
        log(f"{'='*60}")
        log(f"  Total observations: {n_obs}")
        log(f"  Training period:    {train_levels.index[0].strftime('%Y-%m')} to {train_levels.index[-1].strftime('%Y-%m')}")
        log(f"  Holdout period:     {test_dates[0].strftime('%Y-%m')} to {test_dates[-1].strftime('%Y-%m')}")
        log(f"  Training periods:   {len(train_levels)}")
        log(f"  Holdout periods:    {holdout}")
        log(f"  ARIMA order:        {'Auto-select' if auto_order else order}")
        log(f"  Number of series:   {n_series}")
        log(f"{'='*60}\n")
    
    train_transformed = pd.DataFrame(index=train_levels.index)
    for col in all_cols:
        tc = tcodes.get(col, 1)
        if pd.isna(tc):
            tc = 1
        train_transformed[col] = apply_tcode(train_levels[col], int(tc))
    
    forecasts = pd.DataFrame(index=test_dates)
    successful = 0
    failed = 0
    
    for i, col in enumerate(all_cols):
        if verbose and (i % 20 == 0 or i == n_series - 1):
            log(f"  Processing {i+1}/{n_series}: {col}")
        
        try:
            y = train_transformed[col].dropna()
            
            if len(y) < 30:
                forecasts[col] = np.nan
                failed += 1
                continue
            
            if auto_order:
                use_order = select_arima_order(y)
            else:
                use_order = order
            
            fc_transformed = forecast_arima_series(y, holdout, order=use_order, auto_select=False)
            
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
        description="ARIMA Model Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  FORECAST: Use all data to predict future periods
  BACKTEST: Hold out data to evaluate model accuracy

Examples:
  # Forecast next 12 months with auto-selected order
  python arima_forecast.py --data data.csv --horizon 12 --auto --output arima_forecast.csv
  
  # Backtest with 12-month holdout and auto order selection
  python arima_forecast.py --data data.csv --backtest 12 --auto --output arima_backtest.csv
  
  # Backtest with fixed ARIMA(2,0,2)
  python arima_forecast.py --data data.csv --backtest 12 --p 2 --d 0 --q 2 --output arima_backtest.csv
        """
    )
    
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--no-tcodes", action="store_true", help="No transformation code row")
    parser.add_argument("--p", type=int, default=1, help="AR order (default: 1)")
    parser.add_argument("--d", type=int, default=0, help="Differencing order (default: 0)")
    parser.add_argument("--q", type=int, default=1, help="MA order (default: 1)")
    parser.add_argument("--auto", action="store_true", help="Auto-select ARIMA order using AIC")
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--horizon", type=int, help="Forecast mode: predict H periods ahead")
    mode_group.add_argument("--backtest", type=int, help="Backtest mode: holdout H periods")
    
    parser.add_argument("--output", type=str, default="arima_output.csv", help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    
    args = parser.parse_args()
    
    t0 = time.perf_counter()
    
    if args.verbose:
        log(f"\nLoading {args.data}...")
    
    levels, tcodes = load_data(args.data, has_tcodes=not args.no_tcodes)
    
    if args.verbose:
        log(f"Data: {len(levels)} rows x {len(levels.columns)} columns")
    
    order = (args.p, args.d, args.q)
    
    if args.horizon:
        result = run_arima_forecast(
            levels, tcodes,
            args.horizon, order, args.auto,
            args.verbose
        )
    else:
        result = run_arima_backtest(
            levels, tcodes,
            args.backtest, order, args.auto,
            args.verbose
        )
    
    result.to_csv(args.output, index=False)
    
    elapsed = time.perf_counter() - t0
    log(f"\nSaved to {args.output} ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()