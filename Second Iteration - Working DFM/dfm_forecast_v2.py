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
# PIPELINE
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


def run_backtest(
    levels: pd.DataFrame,
    tcodes: pd.Series,
    k_factors: int,
    factor_order: int,
    holdout: int,
    verbose: bool
) -> pd.DataFrame:
    """
    Run backtest and return results in the same format as input data.
    Output: rows = dates, columns = series (with sasdate as first column)
    """
    
    train = levels.iloc[:-holdout]
    actual = levels.iloc[-holdout:]
    
    if verbose:
        log(f"Training on {len(train)} obs, testing on {holdout}")
    
    # Get forecast
    forecast = run_forecast(
        train, tcodes,
        k_factors, factor_order,
        holdout, verbose
    )
    forecast.index = actual.index
    
    # Format output to match original data structure
    # Reset index to make date a column named 'sasdate'
    df = forecast.reset_index()
    df = df.rename(columns={df.columns[0]: 'sasdate'})
    
    # Format dates to match original format (m/d/Y)
    # Use platform-appropriate format specifier
    try:
        # Try Unix-style first (Linux/Mac)
        df['sasdate'] = df['sasdate'].dt.strftime('%-m/%-d/%Y')
    except ValueError:
        # Fall back to Windows-style
        df['sasdate'] = df['sasdate'].dt.strftime('%#m/%#d/%Y')
    
    if verbose:
        log(f"\n{'='*50}")
        log("BACKTEST SUMMARY")
        log(f"{'='*50}")
        log(f"  Series: {len(forecast.columns)}")
        log(f"  Periods: {holdout}")
        log(f"{'='*50}\n")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fast DFM Forecast")
    parser.add_argument("--data", required=True, help="CSV file")
    parser.add_argument("--no-tcodes", action="store_true", help="No tcode row")
    parser.add_argument("--k_factors", type=int, default=2, help="Factors (default: 2)")
    parser.add_argument("--factor_order", type=int, default=1, help="VAR order (default: 1)")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon (default: 12)")
    parser.add_argument("--backtest", type=int, default=None, help="Backtest holdout periods")
    parser.add_argument("--output", type=str, default="forecast.csv", help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    t0 = time.perf_counter()
    
    if args.verbose:
        log(f"\nLoading {args.data}...")
    
    levels, tcodes = load_data(args.data, has_tcodes=not args.no_tcodes)
    
    if args.verbose:
        log(f"Data: {len(levels)} rows x {len(levels.columns)} columns")
    
    if args.backtest:
        # Backtest mode
        if args.verbose:
            log(f"\nMode: BACKTEST ({args.backtest} periods)")
        
        result = run_backtest(
            levels, tcodes,
            args.k_factors, args.factor_order,
            args.backtest, args.verbose
        )
        # Save without index since sasdate is now a column
        result.to_csv(args.output, index=False)
    else:
        # Forecast mode
        if args.verbose:
            log(f"\nMode: FORECAST ({args.horizon} periods)")
        
        result = run_forecast(
            levels, tcodes,
            args.k_factors, args.factor_order,
            args.horizon, args.verbose
        )
        # Format forecast output to match original data format
        df = result.reset_index()
        df = df.rename(columns={df.columns[0]: 'sasdate'})
        try:
            df['sasdate'] = df['sasdate'].dt.strftime('%-m/%-d/%Y')
        except ValueError:
            df['sasdate'] = df['sasdate'].dt.strftime('%#m/%#d/%Y')
        df.to_csv(args.output, index=False)
    
    elapsed = time.perf_counter() - t0
    log(f"Saved {args.output} ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()