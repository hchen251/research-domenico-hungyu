from __future__ import annotations

import sys
import io
import os
import multiprocessing
import argparse
import warnings
import time
from typing import List, Tuple
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from statsmodels.tsa.api import VAR

# Force unbuffered output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

# CPU setup
PHYSICAL_CORES = max(1, multiprocessing.cpu_count() // 2)
os.environ['OMP_NUM_THREADS'] = str(PHYSICAL_CORES)
os.environ['MKL_NUM_THREADS'] = str(PHYSICAL_CORES)
os.environ['OPENBLAS_NUM_THREADS'] = str(PHYSICAL_CORES)


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


# =============================================================================
# TRANSFORMATIONS
# =============================================================================

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


def transform_data(levels: pd.DataFrame, tcodes: pd.Series) -> pd.DataFrame:
    """Transform all series."""
    result = {}
    for col in levels.columns:
        tc = tcodes.get(col, 1)
        if pd.isna(tc):
            tc = 1
        result[col] = apply_tcode(levels[col], int(tc))
    return pd.DataFrame(result, index=levels.index)


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
# FAST DFM MODEL
# =============================================================================

class FastDFM:
    """Fast Dynamic Factor Model using PCA + VAR."""
    
    def __init__(self, n_factors: int = 2, factor_order: int = 1):
        self.n_factors = n_factors
        self.factor_order = factor_order
        self.pca = None
        self.var_result = None
        self.loadings = None
        self.factors = None
        self.columns = None
        self.imputer = None
    
    def fit(self, data: pd.DataFrame) -> 'FastDFM':
        """Fit PCA + VAR model."""
        self.columns = data.columns.tolist()
        
        self.imputer = SimpleImputer(strategy='mean')
        data_filled = pd.DataFrame(
            self.imputer.fit_transform(data),
            index=data.index,
            columns=data.columns
        )
        
        self.pca = PCA(n_components=self.n_factors)
        self.factors = pd.DataFrame(
            self.pca.fit_transform(data_filled),
            index=data.index,
            columns=[f'F{i+1}' for i in range(self.n_factors)]
        )
        self.loadings = self.pca.components_.T
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(self.factors)
            self.var_result = model.fit(maxlags=self.factor_order)
        
        return self
    
    def forecast(self, steps: int) -> pd.DataFrame:
        """Forecast series."""
        factor_fc = self.var_result.forecast(
            self.factors.values[-self.factor_order:],
            steps=steps
        )
        series_fc = factor_fc @ self.loadings.T
        return pd.DataFrame(series_fc, columns=self.columns)
    
    @property
    def variance_explained(self) -> float:
        """Return total variance explained by factors."""
        return self.pca.explained_variance_ratio_.sum() * 100


# =============================================================================
# FORECASTING PIPELINE
# =============================================================================

def run_forecast(
    levels: pd.DataFrame,
    tcodes: pd.Series,
    k_factors: int,
    factor_order: int,
    horizon: int,
    verbose: bool
) -> pd.DataFrame:
    """Run forecasting pipeline."""
    
    all_cols = list(levels.columns)
    
    if verbose:
        log("Transforming...")
    transformed = transform_data(levels, tcodes).dropna(how='all')
    
    if verbose:
        log("Standardizing...")
    mu = transformed.mean().fillna(0)
    sigma = transformed.std().replace(0, 1).fillna(1)
    std_data = (transformed - mu) / sigma
    
    # Filter valid series
    valid_cols = []
    for col in std_data.columns:
        n = std_data[col].notna().sum()
        s = std_data[col].dropna().std()
        if n >= 100 and s > 0 and np.isfinite(s):
            valid_cols.append(col)
    
    if verbose:
        log(f"Using {len(valid_cols)}/{len(all_cols)} series")
    
    if len(valid_cols) < k_factors:
        raise ValueError(f"Only {len(valid_cols)} valid series, need {k_factors}")
    
    std_filtered = std_data[valid_cols]
    
    if verbose:
        log(f"Fitting model ({k_factors} factors, order {factor_order})...")
    
    model = FastDFM(n_factors=k_factors, factor_order=factor_order)
    model.fit(std_filtered)
    
    if verbose:
        log(f"Variance explained: {model.variance_explained:.1f}%")
        log(f"Forecasting {horizon} steps...")
    
    fcast_std = model.forecast(steps=horizon)
    
    # Unstandardize
    fcast_trans = pd.DataFrame(index=range(horizon))
    for col in valid_cols:
        fcast_trans[col] = fcast_std[col].values * sigma[col] + mu[col]
    
    # Build forecast index
    last = transformed.index[-1]
    y, m = last.year, last.month
    if m == 12:
        y, m = y + 1, 1
    else:
        m += 1
    fcast_trans.index = pd.date_range(
        pd.Timestamp(year=y, month=m, day=1),
        periods=horizon,
        freq='MS'
    )
    
    # Invert to levels
    result = pd.DataFrame(index=fcast_trans.index)
    for col in valid_cols:
        tc = tcodes.get(col, 1)
        if pd.isna(tc):
            tc = 1
        hist = levels[col] if col in levels.columns else pd.Series(dtype=float)
        result[col] = invert_tcode(fcast_trans[col].values, hist, int(tc))
    
    for col in all_cols:
        if col not in result.columns:
            result[col] = np.nan
    
    return result[all_cols]


# =============================================================================
# ONE-STEP-AHEAD ROLLING WINDOW FORECAST
# =============================================================================

def run_rolling_one_step_ahead(
    levels: pd.DataFrame,
    tcodes: pd.Series,
    k_factors: int,
    factor_order: int,
    train_window: int,
    verbose: bool
) -> pd.DataFrame:
    """
    Run one-step-ahead rolling window forecast.
    
    For each window:
    - Train on 'train_window' months of data
    - Forecast exactly 1 month ahead
    - Roll forward by 1 month
    - Repeat
    
    Parameters:
    -----------
    levels : pd.DataFrame
        Full historical data with dates as index
    tcodes : pd.Series
        Transformation codes for each series
    k_factors : int
        Number of factors for DFM
    factor_order : int
        VAR order for factor dynamics
    train_window : int
        Number of months in training window (e.g., 120)
    verbose : bool
        Print progress updates
    
    Returns:
    --------
    pd.DataFrame
        Forecasts with sasdate as first column, then series columns
        Each row is a one-step-ahead forecast for that date
    """
    
    n_obs = len(levels)
    horizon = 1  # One-step-ahead
    
    # Minimum data requirement
    min_required = train_window + horizon
    if n_obs < min_required:
        raise ValueError(
            f"Not enough data for rolling window. "
            f"Have {n_obs} observations, need at least {min_required} "
            f"(train_window={train_window} + horizon={horizon})"
        )
    
    # Calculate number of forecasts we can make
    # First forecast: train on [0, train_window-1], forecast for index train_window
    # Last forecast: train on [n_obs-train_window-1, n_obs-2], forecast for index n_obs-1
    n_forecasts = n_obs - train_window
    
    if verbose:
        log(f"\n{'='*60}")
        log("ONE-STEP-AHEAD ROLLING WINDOW FORECAST")
        log(f"{'='*60}")
        log(f"  Total observations:    {n_obs}")
        log(f"  Training window size:  {train_window} months")
        log(f"  Forecast horizon:      {horizon} month (one-step-ahead)")
        log(f"  Number of forecasts:   {n_forecasts}")
        log(f"  First forecast date:   {levels.index[train_window].strftime('%Y-%m-%d')}")
        log(f"  Last forecast date:    {levels.index[n_obs-1].strftime('%Y-%m-%d')}")
        log(f"{'='*60}\n")
    
    all_forecasts = []
    successful = 0
    failed = 0
    
    for i in range(n_forecasts):
        # Define training window
        train_start = i
        train_end = i + train_window  # exclusive index for iloc
        
        # Target date (the date we're forecasting)
        target_idx = train_end  # = i + train_window
        target_date = levels.index[target_idx]
        
        # Extract training data
        train_data = levels.iloc[train_start:train_end].copy()
        
        # Progress update
        if verbose and (i % 20 == 0 or i == n_forecasts - 1):
            pct = (i + 1) / n_forecasts * 100
            train_start_date = train_data.index[0].strftime('%Y-%m')
            train_end_date = train_data.index[-1].strftime('%Y-%m')
            log(f"  [{pct:5.1f}%] Window {i+1}/{n_forecasts}: "
                f"Train {train_start_date} to {train_end_date} → "
                f"Forecast {target_date.strftime('%Y-%m')}")
        
        try:
            # Run forecast (returns DataFrame with 1 row for horizon=1)
            forecast = run_forecast(
                train_data, tcodes,
                k_factors, factor_order,
                horizon=1,
                verbose=False
            )
            
            # Build output row
            forecast_row = {'sasdate': target_date}
            
            # Add all series values
            for col in forecast.columns:
                forecast_row[col] = forecast[col].iloc[0]
            
            all_forecasts.append(forecast_row)
            successful += 1
            
        except Exception as e:
            failed += 1
            if verbose and failed <= 5:
                log(f"    Warning: Window {i+1} failed - {str(e)[:60]}")
            elif verbose and failed == 6:
                log(f"    (Suppressing further warnings...)")
            continue
    
    # Create result DataFrame
    result = pd.DataFrame(all_forecasts)
    
    if len(result) == 0:
        raise ValueError("All rolling windows failed. Check your data.")
    
    # Format date column
    try:
        result['sasdate'] = pd.to_datetime(result['sasdate']).dt.strftime('%-m/%-d/%Y')
    except ValueError:
        # Windows uses different format specifier
        result['sasdate'] = pd.to_datetime(result['sasdate']).dt.strftime('%#m/%#d/%Y')
    
    if verbose:
        log(f"\n{'='*60}")
        log("ROLLING FORECAST COMPLETE")
        log(f"{'='*60}")
        log(f"  Successful forecasts: {successful}")
        log(f"  Failed forecasts:     {failed}")
        log(f"  Output rows:          {len(result)}")
        log(f"  Output columns:       {len(result.columns)}")
        log(f"{'='*60}\n")
    
    return result


# =============================================================================
# MULTI-HORIZON ROLLING WINDOW FORECAST
# =============================================================================

def run_rolling_multi_horizon(
    levels: pd.DataFrame,
    tcodes: pd.Series,
    k_factors: int,
    factor_order: int,
    horizon: int,
    train_window: int,
    verbose: bool
) -> pd.DataFrame:
    """
    Run h-step-ahead rolling window forecast.
    
    Similar to one-step-ahead, but forecasts 'horizon' months ahead.
    Only keeps the final (h-th) forecast from each window.
    
    Parameters:
    -----------
    horizon : int
        Forecast horizon (e.g., 12 for 12-month-ahead forecasts)
    """
    
    n_obs = len(levels)
    
    min_required = train_window + horizon
    if n_obs < min_required:
        raise ValueError(
            f"Not enough data. Have {n_obs}, need {min_required}"
        )
    
    n_forecasts = n_obs - train_window - horizon + 1
    
    if verbose:
        log(f"\n{'='*60}")
        log(f"{horizon}-STEP-AHEAD ROLLING WINDOW FORECAST")
        log(f"{'='*60}")
        log(f"  Total observations:    {n_obs}")
        log(f"  Training window size:  {train_window} months")
        log(f"  Forecast horizon:      {horizon} months")
        log(f"  Number of forecasts:   {n_forecasts}")
        log(f"{'='*60}\n")
    
    all_forecasts = []
    successful = 0
    failed = 0
    
    for i in range(n_forecasts):
        train_start = i
        train_end = i + train_window
        
        # Target date is 'horizon' months after end of training
        target_idx = train_end + horizon - 1
        target_date = levels.index[target_idx]
        
        train_data = levels.iloc[train_start:train_end].copy()
        
        if verbose and (i % 20 == 0 or i == n_forecasts - 1):
            pct = (i + 1) / n_forecasts * 100
            log(f"  [{pct:5.1f}%] Window {i+1}/{n_forecasts} → "
                f"Forecast {target_date.strftime('%Y-%m')}")
        
        try:
            forecast = run_forecast(
                train_data, tcodes,
                k_factors, factor_order,
                horizon=horizon,
                verbose=False
            )
            
            # Take the h-th forecast (last row)
            forecast_row = {'sasdate': target_date}
            for col in forecast.columns:
                forecast_row[col] = forecast[col].iloc[horizon - 1]
            
            all_forecasts.append(forecast_row)
            successful += 1
            
        except Exception as e:
            failed += 1
            continue
    
    result = pd.DataFrame(all_forecasts)
    
    if len(result) == 0:
        raise ValueError("All rolling windows failed.")
    
    try:
        result['sasdate'] = pd.to_datetime(result['sasdate']).dt.strftime('%-m/%-d/%Y')
    except ValueError:
        result['sasdate'] = pd.to_datetime(result['sasdate']).dt.strftime('%#m/%#d/%Y')
    
    if verbose:
        log(f"\n  Complete: {successful} forecasts, {failed} failed\n")
    
    return result


# =============================================================================
# BACKTEST (Simple holdout)
# =============================================================================

def run_backtest(
    levels: pd.DataFrame,
    tcodes: pd.Series,
    k_factors: int,
    factor_order: int,
    holdout: int,
    verbose: bool
) -> pd.DataFrame:
    """Run simple backtest with holdout."""
    
    train = levels.iloc[:-holdout]
    actual = levels.iloc[-holdout:]
    
    if verbose:
        log(f"Training on {len(train)} obs, testing on {holdout}")
    
    forecast = run_forecast(
        train, tcodes,
        k_factors, factor_order,
        holdout, verbose
    )
    forecast.index = actual.index
    
    df = forecast.reset_index()
    df = df.rename(columns={df.columns[0]: 'sasdate'})
    
    try:
        df['sasdate'] = df['sasdate'].dt.strftime('%-m/%-d/%Y')
    except ValueError:
        df['sasdate'] = df['sasdate'].dt.strftime('%#m/%#d/%Y')
    
    if verbose:
        log(f"\nBacktest complete: {len(df)} forecast rows")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fast DFM Forecast with Rolling Window Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple forecast (12 months ahead from end of data)
  python dfm_forecast_v2.py --data data.csv --horizon 12 --output forecast.csv
  
  # One-step-ahead rolling window
  python dfm_forecast_v2.py --data data.csv --rolling --train_window 120 --output rolling_1m.csv
  
  # 12-step-ahead rolling window
  python dfm_forecast_v2.py --data data.csv --rolling --horizon 12 --train_window 120 --output rolling_12m.csv
  
  # Simple backtest (hold out last 24 months)
  python dfm_forecast_v2.py --data data.csv --backtest 24 --output backtest.csv
        """
    )
    
    parser.add_argument("--data", required=True, help="Input CSV file path")
    parser.add_argument("--no-tcodes", action="store_true", help="Data has no transformation code row")
    parser.add_argument("--k_factors", type=int, default=2, help="Number of factors (default: 2)")
    parser.add_argument("--factor_order", type=int, default=1, help="VAR order for factors (default: 1)")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in months (default: 1)")
    parser.add_argument("--rolling", action="store_true", help="Use rolling window forecasting")
    parser.add_argument("--train_window", type=int, default=120, help="Rolling window training size in months (default: 120)")
    parser.add_argument("--backtest", type=int, default=None, help="Simple backtest with N holdout periods")
    parser.add_argument("--output", type=str, default="forecast.csv", help="Output CSV file path")
    parser.add_argument("--verbose", action="store_true", help="Print progress information")
    
    args = parser.parse_args()
    
    t0 = time.perf_counter()
    
    if args.verbose:
        log(f"\n{'='*60}")
        log("DFM FORECAST")
        log(f"{'='*60}")
        log(f"Loading {args.data}...")
    
    levels, tcodes = load_data(args.data, has_tcodes=not args.no_tcodes)
    
    if args.verbose:
        log(f"Data loaded: {len(levels)} rows x {len(levels.columns)} columns")
        log(f"Date range: {levels.index[0].strftime('%Y-%m-%d')} to {levels.index[-1].strftime('%Y-%m-%d')}")
    
    # Determine mode and run
    if args.rolling:
        # Rolling window mode
        if args.horizon == 1:
            result = run_rolling_one_step_ahead(
                levels, tcodes,
                args.k_factors, args.factor_order,
                args.train_window, args.verbose
            )
        else:
            result = run_rolling_multi_horizon(
                levels, tcodes,
                args.k_factors, args.factor_order,
                args.horizon, args.train_window,
                args.verbose
            )
        result.to_csv(args.output, index=False)
        
    elif args.backtest:
        # Simple backtest mode
        if args.verbose:
            log(f"\nMode: BACKTEST ({args.backtest} periods)")
        
        result = run_backtest(
            levels, tcodes,
            args.k_factors, args.factor_order,
            args.backtest, args.verbose
        )
        result.to_csv(args.output, index=False)
        
    else:
        # Standard forecast mode
        if args.verbose:
            log(f"\nMode: FORECAST ({args.horizon} periods ahead)")
        
        result = run_forecast(
            levels, tcodes,
            args.k_factors, args.factor_order,
            args.horizon, args.verbose
        )
        
        df = result.reset_index()
        df = df.rename(columns={df.columns[0]: 'sasdate'})
        try:
            df['sasdate'] = df['sasdate'].dt.strftime('%-m/%-d/%Y')
        except ValueError:
            df['sasdate'] = df['sasdate'].dt.strftime('%#m/%#d/%Y')
        df.to_csv(args.output, index=False)
    
    elapsed = time.perf_counter() - t0
    log(f"\nSaved to {args.output}")
    log(f"Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()