"""
DFM Forecast Output Analyzer
Specifically designed for the output format of dfm_forecast_v2.py

The output CSV has:
- sasdate: Date column (YYYY-MM-DD format)
- Series columns: INDPRO, UNRATE, RPI, etc. (forecasted level values)

Usage:
    python3 analyze_forecast.py --forecast out.csv
    python3 analyze_forecast.py --forecast out.csv --historical 2014-12.csv --plot 

Basic analysis
python3 analyze_forecast.py --forecast out.csv

With historical comparison and plots
python3 analyze_forecast.py --forecast out.csv --historical 2014-12.csv --plot

Full analysis with report
python3 analyze_forecast.py --forecast out.csv --historical 2014-12.csv --plot --report --export_csv

Analyze specific series
python3 analyze_forecast.py --forecast out.csv --series "INDPRO,UNRATE,RPI"
"""

import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: Install matplotlib for visualization (pip install matplotlib)")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dfm_forecast(filepath: str, date_col: str = "sasdate") -> pd.DataFrame:
    """
    Load the DFM forecast output CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the forecast CSV (e.g., out.csv)
    date_col : str
        Name of the date column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and series as columns
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    
    # Convert all columns to numeric at once
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df


def load_historical_fredmd(filepath: str, date_col: str = "sasdate") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load historical FRED-MD data (with transform codes in first row).
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (levels DataFrame, transformation codes Series)
    """
    raw = pd.read_csv(filepath)
    
    # Check for transform row
    if str(raw.loc[0, date_col]).strip().lower() == "transform:":
        tcodes = raw.iloc[0].drop(date_col).astype(float)
        df = raw.iloc[1:].copy()
    else:
        tcodes = pd.Series()
        df = raw.copy()
    
    # Parse dates - try multiple formats
    df[date_col] = pd.to_datetime(df[date_col], format="%m/%d/%Y", errors="coerce")
    if df[date_col].isna().all():
        df[date_col] = pd.to_datetime(raw.iloc[1:][date_col], errors="coerce")
    
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    
    # Convert to numeric all at once
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df, tcodes


def strip_value_suffix(col_name: str) -> str:
    """Remove '_value' suffix from column name if present."""
    if col_name.endswith('_value'):
        return col_name[:-6]
    return col_name


def add_value_suffix(col_name: str) -> str:
    """Add '_value' suffix to column name if not present."""
    if not col_name.endswith('_value'):
        return f"{col_name}_value"
    return col_name


def get_series_mapping(forecast_df: pd.DataFrame, historical_df: pd.DataFrame) -> Dict[str, str]:
    """
    Create mapping between forecast columns (with _value) and historical columns (without).
    """
    mapping = {}
    for fcol in forecast_df.columns:
        base_name = strip_value_suffix(fcol)
        if base_name in historical_df.columns:
            mapping[fcol] = base_name
    return mapping


def get_common_series(forecast_df: pd.DataFrame, historical_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Get list of (forecast_col, historical_col) pairs for matching series.
    """
    pairs = []
    for fcol in forecast_df.columns:
        base_name = strip_value_suffix(fcol)
        if base_name in historical_df.columns:
            pairs.append((fcol, base_name))
    return pairs


# =============================================================================
# BASIC ANALYSIS
# =============================================================================

def summarize_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for each forecasted series.
    Returns DataFrame built efficiently without fragmentation.
    """
    records = []
    
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        
        first_val = s.iloc[0]
        last_val = s.iloc[-1]
        total_chg = last_val - first_val
        
        # Handle percentage change carefully
        if first_val != 0:
            pct_chg = (total_chg / abs(first_val) * 100)
        else:
            pct_chg = np.nan
        
        # Determine trend direction
        if pd.isna(pct_chg):
            trend = "N/A"
        elif pct_chg > 1:
            trend = "↑ Up"
        elif pct_chg < -1:
            trend = "↓ Down"
        else:
            trend = "→ Flat"
        
        # Get base name for display
        display_name = strip_value_suffix(col)
        
        records.append({
            "series": display_name,
            "start_date": s.index[0].strftime("%Y-%m-%d"),
            "end_date": s.index[-1].strftime("%Y-%m-%d"),
            "n_periods": len(s),
            "first_value": round(first_val, 4),
            "last_value": round(last_val, 4),
            "min": round(s.min(), 4),
            "max": round(s.max(), 4),
            "mean": round(s.mean(), 4),
            "std": round(s.std(), 4),
            "total_change": round(total_chg, 4),
            "pct_change": round(pct_chg, 2) if not pd.isna(pct_chg) else np.nan,
            "trend": trend
        })
    
    return pd.DataFrame(records)


def calculate_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate month-over-month and annualized growth rates.
    Handles negative values by using simple percent change instead of log returns.
    """
    records = []
    
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 2:
            continue
        
        display_name = strip_value_suffix(col)
        
        # Check if series has any non-positive values
        has_non_positive = (s <= 0).any()
        
        if has_non_positive:
            # Use simple percent change for series with non-positive values
            returns = s.pct_change().dropna() * 100
            method = "pct_change"
        else:
            # Use log returns for positive series (more accurate for growth rates)
            with np.errstate(invalid='ignore'):
                returns = np.log(s).diff().dropna() * 100
            method = "log_return"
        
        if len(returns) == 0:
            continue
        
        avg_monthly = returns.mean()
        std_monthly = returns.std()
        
        # Annualized
        avg_annual = avg_monthly * 12
        std_annual = std_monthly * np.sqrt(12)
        
        records.append({
            "series": display_name,
            "avg_monthly_growth_%": round(avg_monthly, 4),
            "avg_annual_growth_%": round(avg_annual, 4),
            "volatility_monthly_%": round(std_monthly, 4),
            "volatility_annual_%": round(std_annual, 4),
            "min_monthly_%": round(returns.min(), 4),
            "max_monthly_%": round(returns.max(), 4),
            "method": method
        })
    
    return pd.DataFrame(records)


def month_over_month_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a table of month-over-month changes.
    Built efficiently using dictionary approach.
    """
    result_dict: Dict[str, Any] = {}
    
    for col in df.columns:
        display_name = strip_value_suffix(col)
        result_dict[display_name] = df[col].values
        result_dict[f"{display_name}_chg"] = df[col].diff().values
        result_dict[f"{display_name}_pct"] = df[col].pct_change().values * 100
    
    return pd.DataFrame(result_dict, index=df.index)


# =============================================================================
# COMPARISON WITH HISTORICAL DATA
# =============================================================================

def compare_forecast_to_history(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    lookback_months: int = 24
) -> pd.DataFrame:
    """
    Compare forecast statistics to recent historical statistics.
    Properly handles column name matching between forecast (_value suffix) and historical data.
    """
    # Get matching series pairs
    series_pairs = get_common_series(forecast_df, historical_df)
    
    if len(series_pairs) == 0:
        print("Warning: No matching series found between forecast and historical data.")
        print(f"  Forecast columns (first 5): {list(forecast_df.columns)[:5]}")
        print(f"  Historical columns (first 5): {list(historical_df.columns)[:5]}")
        return pd.DataFrame()
    
    records = []
    
    for fcol, hcol in series_pairs:
        hist = historical_df[hcol].dropna()
        fcst = forecast_df[fcol].dropna()
        
        if len(hist) == 0 or len(fcst) == 0:
            continue
        
        # Get recent historical data
        hist_recent = hist.iloc[-lookback_months:] if len(hist) >= lookback_months else hist
        
        # Historical stats
        h_mean = hist_recent.mean()
        h_std = hist_recent.std()
        h_min = hist_recent.min()
        h_max = hist_recent.max()
        h_last = hist.iloc[-1]
        
        # Forecast stats
        f_mean = fcst.mean()
        f_std = fcst.std()
        f_min = fcst.min()
        f_max = fcst.max()
        f_first = fcst.iloc[0]
        f_last = fcst.iloc[-1]
        
        # Comparison metrics
        if h_last != 0:
            jump_at_start = ((f_first - h_last) / abs(h_last) * 100)
        else:
            jump_at_start = np.nan
            
        if h_mean != 0:
            mean_shift = ((f_mean - h_mean) / abs(h_mean) * 100)
        else:
            mean_shift = np.nan
        
        # Is forecast within historical range (with 10% buffer)?
        buffer = 0.1
        if h_min >= 0:
            in_range = (f_min >= h_min * (1 - buffer)) and (f_max <= h_max * (1 + buffer))
        else:
            # Handle negative values
            range_size = h_max - h_min
            in_range = (f_min >= h_min - range_size * buffer) and (f_max <= h_max + range_size * buffer)
        
        records.append({
            "series": hcol,
            "hist_last": round(h_last, 2),
            "fcst_first": round(f_first, 2),
            "jump_%": round(jump_at_start, 2) if not pd.isna(jump_at_start) else np.nan,
            "hist_mean": round(h_mean, 2),
            "fcst_mean": round(f_mean, 2),
            "mean_shift_%": round(mean_shift, 2) if not pd.isna(mean_shift) else np.nan,
            "hist_std": round(h_std, 2),
            "fcst_std": round(f_std, 2),
            "hist_range": f"[{h_min:.1f}, {h_max:.1f}]",
            "fcst_range": f"[{f_min:.1f}, {f_max:.1f}]",
            "in_hist_range": "Yes" if in_range else "No"
        })
    
    return pd.DataFrame(records)


def combine_history_and_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    series: str
) -> pd.DataFrame:
    """
    Combine historical and forecast data for a single series.
    Handles column name matching (with/without _value suffix).
    """
    # Find the correct column names
    hist_col = series if series in historical_df.columns else None
    fcst_col = None
    
    if f"{series}_value" in forecast_df.columns:
        fcst_col = f"{series}_value"
    elif series in forecast_df.columns:
        fcst_col = series
    
    if hist_col is None:
        raise ValueError(f"Series '{series}' not found in historical data")
    if fcst_col is None:
        raise ValueError(f"Series '{series}' not found in forecast data")
    
    # Build combined data efficiently
    hist_data = historical_df[hist_col].dropna()
    fcst_data = forecast_df[fcst_col].dropna()
    
    # Remove forecast overlap with historical
    last_hist = hist_data.index.max()
    fcst_data = fcst_data[fcst_data.index > last_hist]
    
    # Create combined DataFrame
    combined_records = []
    for idx, val in hist_data.items():
        combined_records.append({"date": idx, "value": val, "type": "historical"})
    for idx, val in fcst_data.items():
        combined_records.append({"date": idx, "value": val, "type": "forecast"})
    
    combined = pd.DataFrame(combined_records)
    combined = combined.set_index("date").sort_index()
    
    return combined


# =============================================================================
# FORECAST ACCURACY (when actuals are available)
# =============================================================================

def calculate_accuracy_metrics(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate forecast accuracy metrics when actual values are available.
    """
    series_pairs = get_common_series(forecast_df, actual_df)
    common_dates = forecast_df.index.intersection(actual_df.index)
    
    if len(common_dates) == 0:
        print("Warning: No overlapping dates between forecast and actual")
        return pd.DataFrame()
    
    records = []
    
    for fcol, acol in series_pairs:
        f = forecast_df.loc[common_dates, fcol]
        a = actual_df.loc[common_dates, acol]
        
        # Remove NaN pairs
        mask = ~(f.isna() | a.isna())
        f, a = f[mask], a[mask]
        
        if len(f) == 0:
            continue
        
        # Calculate errors
        error = f.values - a.values
        abs_error = np.abs(error)
        sq_error = error ** 2
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_error = np.where(a.values != 0, np.abs(error / a.values) * 100, np.nan)
        
        # Metrics
        me = np.mean(error)
        mae = np.mean(abs_error)
        rmse = np.sqrt(np.mean(sq_error))
        mape = np.nanmean(pct_error)
        corr = np.corrcoef(f.values, a.values)[0, 1] if len(f) > 1 else np.nan
        
        records.append({
            "series": acol,
            "n_obs": len(f),
            "ME": round(me, 4),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE_%": round(mape, 2) if not np.isnan(mape) else np.nan,
            "correlation": round(corr, 4) if not pd.isna(corr) else np.nan,
            "bias_direction": "over" if me > 0 else "under"
        })
    
    return pd.DataFrame(records)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_forecast_single(
    forecast_df: pd.DataFrame,
    series: str,
    historical_df: Optional[pd.DataFrame] = None,
    lookback: int = 60,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot a single series forecast with optional historical data.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Find correct column names
    fcst_col = f"{series}_value" if f"{series}_value" in forecast_df.columns else series
    
    # Plot historical if available
    if historical_df is not None and series in historical_df.columns:
        hist = historical_df[series].dropna()
        if len(hist) > lookback:
            hist = hist.iloc[-lookback:]
        ax.plot(hist.index, hist.values, 'b-', linewidth=2, label='Historical', alpha=0.8)
        
        # Mark transition point
        last_hist_date = hist.index[-1]
        ax.axvline(x=last_hist_date, color='gray', linestyle='--', alpha=0.5, label='Forecast Start')
    
    # Plot forecast
    if fcst_col in forecast_df.columns:
        fcst = forecast_df[fcst_col].dropna()
        ax.plot(fcst.index, fcst.values, 'r-', linewidth=2, label='Forecast', marker='o', markersize=4)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(title or f'{series}: Historical & Forecast', fontsize=13)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_forecast_grid(
    forecast_df: pd.DataFrame,
    series_list: List[str],
    historical_df: Optional[pd.DataFrame] = None,
    lookback: int = 60,
    ncols: int = 3,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot multiple series in a grid layout.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return
    
    n = len(series_list)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    
    # Handle different array shapes
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, series in enumerate(series_list):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        
        # Find correct column names
        fcst_col = f"{series}_value" if f"{series}_value" in forecast_df.columns else series
        
        # Plot historical
        if historical_df is not None and series in historical_df.columns:
            hist = historical_df[series].dropna()
            if len(hist) > lookback:
                hist = hist.iloc[-lookback:]
            ax.plot(hist.index, hist.values, 'b-', linewidth=1.5, alpha=0.7)
        
        # Plot forecast
        if fcst_col in forecast_df.columns:
            fcst = forecast_df[fcst_col].dropna()
            ax.plot(fcst.index, fcst.values, 'r-', linewidth=1.5, marker='o', markersize=2)
        
        ax.set_title(series, fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_normalized(
    forecast_df: pd.DataFrame,
    series_list: List[str],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot multiple forecasts normalized to 100 at start for comparison.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for series in series_list:
        # Find correct column name
        fcst_col = f"{series}_value" if f"{series}_value" in forecast_df.columns else series
        
        if fcst_col not in forecast_df.columns:
            continue
        
        s = forecast_df[fcst_col].dropna()
        if len(s) > 0 and s.iloc[0] != 0:
            normalized = s / s.iloc[0] * 100
            ax.plot(normalized.index, normalized.values, linewidth=2, label=series, marker='o', markersize=3)
    
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Index (100 = Forecast Start)', fontsize=11)
    ax.set_title('Forecast Comparison (Normalized)', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_growth_rates(
    forecast_df: pd.DataFrame,
    series_list: List[str],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot month-over-month growth rates.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for series in series_list:
        # Find correct column name
        fcst_col = f"{series}_value" if f"{series}_value" in forecast_df.columns else series
        
        if fcst_col not in forecast_df.columns:
            continue
        
        s = forecast_df[fcst_col].dropna()
        if len(s) > 1:
            growth = s.pct_change() * 100
            ax.plot(growth.index, growth.values, linewidth=2, label=series, marker='o', markersize=3)
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Month-over-Month Growth (%)', fontsize=11)
    ax.set_title('Forecasted Growth Rates', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# =============================================================================
# INTERACTIVE ANALYSIS FUNCTIONS
# =============================================================================

def quick_look(df: pd.DataFrame, n_series: int = 10) -> None:
    """
    Quick look at the forecast data.
    """
    print("\n" + "=" * 60)
    print("QUICK LOOK AT FORECAST DATA")
    print("=" * 60)
    
    print(f"\nDate range: {df.index.min()} to {df.index.max()}")
    print(f"Number of periods: {len(df)}")
    print(f"Number of series: {len(df.columns)}")
    
    print(f"\nFirst {n_series} series:")
    for col in list(df.columns)[:n_series]:
        series_name = strip_value_suffix(col)
        first = df[col].iloc[0]
        last = df[col].iloc[-1]
        if first != 0:
            pct = (last - first) / abs(first) * 100
            print(f"  {series_name:20s}: {first:12.2f} → {last:12.2f} ({pct:+.2f}%)")
        else:
            print(f"  {series_name:20s}: {first:12.2f} → {last:12.2f}")
    
    if len(df.columns) > n_series:
        print(f"  ... and {len(df.columns) - n_series} more series")


def analyze_series(
    df: pd.DataFrame,
    series: str,
    historical_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Detailed analysis of a single series.
    """
    # Find correct column name
    fcst_col = f"{series}_value" if f"{series}_value" in df.columns else series
    
    if fcst_col not in df.columns:
        print(f"Series '{series}' not found")
        print(f"Available series (first 10): {[strip_value_suffix(c) for c in list(df.columns)[:10]]}")
        return
    
    s = df[fcst_col].dropna()
    
    print("\n" + "=" * 60)
    print(f"ANALYSIS: {series}")
    print("=" * 60)
    
    print(f"\nForecast Period: {s.index.min()} to {s.index.max()}")
    print(f"Number of periods: {len(s)}")
    
    print(f"\nLevel Statistics:")
    print(f"  First value: {s.iloc[0]:,.4f}")
    print(f"  Last value:  {s.iloc[-1]:,.4f}")
    print(f"  Min:         {s.min():,.4f}")
    print(f"  Max:         {s.max():,.4f}")
    print(f"  Mean:        {s.mean():,.4f}")
    print(f"  Std Dev:     {s.std():,.4f}")
    
    total_chg = s.iloc[-1] - s.iloc[0]
    if s.iloc[0] != 0:
        pct_chg = total_chg / abs(s.iloc[0]) * 100
        print(f"\nChange over forecast horizon:")
        print(f"  Absolute: {total_chg:+,.4f}")
        print(f"  Percent:  {pct_chg:+.2f}%")
    else:
        print(f"\nChange over forecast horizon:")
        print(f"  Absolute: {total_chg:+,.4f}")
    
    if len(s) > 1:
        monthly_growth = s.pct_change().dropna() * 100
        print(f"\nMonthly Growth Rates:")
        print(f"  Mean:  {monthly_growth.mean():+.4f}%")
        print(f"  Std:   {monthly_growth.std():.4f}%")
        print(f"  Min:   {monthly_growth.min():+.4f}%")
        print(f"  Max:   {monthly_growth.max():+.4f}%")
    
    # Compare to historical if available
    if historical_df is not None and series in historical_df.columns:
        hist = historical_df[series].dropna()
        if len(hist) > 0:
            print(f"\nComparison to Historical (last 24 months):")
            hist_recent = hist.iloc[-24:] if len(hist) >= 24 else hist
            print(f"  Historical mean: {hist_recent.mean():,.4f}")
            print(f"  Forecast mean:   {s.mean():,.4f}")
            print(f"  Historical std:  {hist_recent.std():,.4f}")
            print(f"  Forecast std:    {s.std():,.4f}")
            print(f"  Last historical: {hist.iloc[-1]:,.4f}")
            print(f"  First forecast:  {s.iloc[0]:,.4f}")
    
    print("\nForecast Values:")
    print(s.to_string())


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_text_report(
    forecast_df: pd.DataFrame,
    historical_df: Optional[pd.DataFrame] = None,
    output_path: str = "forecast_report.txt"
) -> str:
    """
    Generate a text report summarizing the forecast analysis.
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("DFM FORECAST ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    
    # Overview
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Number of series: {len(forecast_df.columns)}")
    lines.append(f"Forecast start: {forecast_df.index.min().strftime('%Y-%m-%d')}")
    lines.append(f"Forecast end: {forecast_df.index.max().strftime('%Y-%m-%d')}")
    lines.append(f"Forecast horizon: {len(forecast_df)} months")
    lines.append("")
    
    # Series list
    lines.append("SERIES INCLUDED")
    lines.append("-" * 40)
    for i, col in enumerate(forecast_df.columns, 1):
        display_name = strip_value_suffix(col)
        lines.append(f"  {i:3d}. {display_name}")
    lines.append("")
    
    # Summary statistics
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 40)
    summary = summarize_forecast(forecast_df)
    lines.append(summary.to_string(index=False))
    lines.append("")
    
    # Growth rates
    lines.append("GROWTH RATE ANALYSIS")
    lines.append("-" * 40)
    growth = calculate_growth_rates(forecast_df)
    lines.append(growth.to_string(index=False))
    lines.append("")
    
    # Historical comparison
    if historical_df is not None:
        comparison = compare_forecast_to_history(forecast_df, historical_df)
        if len(comparison) > 0:
            lines.append("COMPARISON WITH HISTORICAL DATA (last 24 months)")
            lines.append("-" * 40)
            lines.append(comparison.to_string(index=False))
            lines.append("")
    
    # Key findings
    lines.append("KEY FINDINGS")
    lines.append("-" * 40)
    
    # Find biggest movers
    if len(summary) > 0:
        valid_pct = summary[summary['pct_change'].notna()]
        if len(valid_pct) > 0:
            biggest_gain = valid_pct.loc[valid_pct["pct_change"].idxmax()]
            biggest_loss = valid_pct.loc[valid_pct["pct_change"].idxmin()]
            
            lines.append(f"Largest increase: {biggest_gain['series']} ({biggest_gain['pct_change']:+.2f}%)")
            lines.append(f"Largest decrease: {biggest_loss['series']} ({biggest_loss['pct_change']:+.2f}%)")
            
            up_count = (valid_pct["pct_change"] > 1).sum()
            down_count = (valid_pct["pct_change"] < -1).sum()
            flat_count = len(valid_pct) - up_count - down_count
            
            lines.append(f"Trending up: {up_count} series")
            lines.append(f"Trending down: {down_count} series")
            lines.append(f"Relatively flat: {flat_count} series")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    # Write to file
    report_text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report_text)
    
    print(f"Report saved to: {output_path}")
    
    return report_text


def export_analysis_to_csv(
    forecast_df: pd.DataFrame,
    historical_df: Optional[pd.DataFrame] = None,
    output_dir: str = "analysis_output"
) -> None:
    """
    Export all analysis to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Raw forecasts
    forecast_df.to_csv(os.path.join(output_dir, "forecasts_raw.csv"))
    print(f"Saved: {os.path.join(output_dir, 'forecasts_raw.csv')}")
    
    # Summary statistics
    summarize_forecast(forecast_df).to_csv(
        os.path.join(output_dir, "summary_stats.csv"), index=False
    )
    print(f"Saved: {os.path.join(output_dir, 'summary_stats.csv')}")
    
    # Growth rates
    calculate_growth_rates(forecast_df).to_csv(
        os.path.join(output_dir, "growth_rates.csv"), index=False
    )
    print(f"Saved: {os.path.join(output_dir, 'growth_rates.csv')}")
    
    # Month-over-month
    month_over_month_table(forecast_df).to_csv(
        os.path.join(output_dir, "month_over_month.csv")
    )
    print(f"Saved: {os.path.join(output_dir, 'month_over_month.csv')}")
    
    # Historical comparison
    if historical_df is not None:
        comparison = compare_forecast_to_history(forecast_df, historical_df)
        if len(comparison) > 0:
            comparison.to_csv(
                os.path.join(output_dir, "historical_comparison.csv"), index=False
            )
            print(f"Saved: {os.path.join(output_dir, 'historical_comparison.csv')}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze DFM forecast output")
    parser.add_argument("--forecast", type=str, required=True, help="Path to forecast CSV")
    parser.add_argument("--historical", type=str, default=None, help="Path to historical FRED-MD CSV")
    parser.add_argument("--output_dir", type=str, default="analysis_output", help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--series", type=str, default=None, help="Comma-separated series to analyze (base names)")
    parser.add_argument("--report", action="store_true", help="Generate text report")
    parser.add_argument("--export_csv", action="store_true", help="Export all analysis to CSV files")
    parser.add_argument("--no_show", action="store_true", help="Don't display plots (just save)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading forecast from: {args.forecast}")
    forecast_df = load_dfm_forecast(args.forecast)
    print(f"  Loaded {len(forecast_df)} periods, {len(forecast_df.columns)} series")
    
    historical_df = None
    if args.historical:
        print(f"\nLoading historical from: {args.historical}")
        historical_df, _ = load_historical_fredmd(args.historical)
        print(f"  Loaded {len(historical_df)} periods, {len(historical_df.columns)} series")
        
        # Show matching info
        series_pairs = get_common_series(forecast_df, historical_df)
        print(f"  Matching series found: {len(series_pairs)}")
    
    # Quick look
    quick_look(forecast_df)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    summary = summarize_forecast(forecast_df)
    print(summary.to_string(index=False))
    
    # Growth rates
    print("\n" + "=" * 60)
    print("GROWTH RATE ANALYSIS")
    print("=" * 60)
    growth = calculate_growth_rates(forecast_df)
    print(growth.to_string(index=False))
    
    # Historical comparison
    if historical_df is not None:
        print("\n" + "=" * 60)
        print("COMPARISON WITH HISTORICAL DATA")
        print("=" * 60)
        comparison = compare_forecast_to_history(forecast_df, historical_df)
        if len(comparison) > 0:
            print(comparison.to_string(index=False))
        else:
            print("No matching series found for comparison.")
    
    # Specific series analysis
    if args.series:
        series_list = [s.strip() for s in args.series.split(",")]
        for s in series_list:
            analyze_series(forecast_df, s, historical_df)
    
    # Get series list for plotting (use base names)
    if args.series:
        plot_series = [s.strip() for s in args.series.split(",")]
    else:
        # Use first 9 series (base names)
        plot_series = [strip_value_suffix(c) for c in list(forecast_df.columns)[:9]]
    
    show_plots = not args.no_show
    
    # Plots
    if args.plot and HAS_MATPLOTLIB:
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)
        
        # Grid plot
        plot_forecast_grid(
            forecast_df, plot_series, historical_df,
            save_path=os.path.join(args.output_dir, "forecast_grid.png"),
            show=show_plots
        )
        
        # Normalized comparison
        plot_comparison_normalized(
            forecast_df, plot_series[:6],
            save_path=os.path.join(args.output_dir, "forecast_comparison.png"),
            show=show_plots
        )
        
        # Growth rates
        plot_growth_rates(
            forecast_df, plot_series[:6],
            save_path=os.path.join(args.output_dir, "growth_rates.png"),
            show=show_plots
        )
    
    # Generate report
    if args.report:
        generate_text_report(
            forecast_df, historical_df,
            output_path=os.path.join(args.output_dir, "forecast_report.txt")
        )
    
    # Export to CSV
    if args.export_csv:
        print("\n" + "=" * 60)
        print("EXPORTING TO CSV")
        print("=" * 60)
        export_analysis_to_csv(forecast_df, historical_df, args.output_dir)
    
    print("\n" + "=" * 60)
    print(f"Analysis complete. Output saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()