"""
Bayesian Dynamic Factor Model Forecasting for FRED-MD data

Usage:
    python forecast.py -i 2025-12-MD.csv -o forecasts.csv
    python forecast.py -i 2025-12-MD.csv -o forecasts.csv --forecast-horizon 12
    python forecast.py -i 2025-12-MD.csv -o forecasts.csv --start-date 2020-01-01
    python forecast.py -i 2025-12-MD.csv -o forecasts.csv --method bayesian
"""

import argparse
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(filepath):
    """Load FRED-MD data and extract transformation codes."""
    df = pd.read_csv(filepath)
    transform_codes = df.iloc[0].copy()
    df = df.iloc[1:].copy()
    df['sasdate'] = pd.to_datetime(df['sasdate'], format='%m/%d/%Y')
    df.set_index('sasdate', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df, transform_codes


def apply_transformations(df, transform_codes):
    """
    Apply FRED-MD transformations to make series stationary.
    
    Codes: 1=none, 2=diff, 3=diff2, 4=log, 5=log-diff, 6=log-diff2, 7=pct-diff
    """
    transformed = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        if col == 'sasdate':
            continue
        series = df[col].copy()
        try:
            code = int(float(transform_codes[col]))
        except (ValueError, KeyError):
            code = 1
        
        if code == 1:
            transformed[col] = series
        elif code == 2:
            transformed[col] = series.diff()
        elif code == 3:
            transformed[col] = series.diff().diff()
        elif code == 4:
            transformed[col] = np.log(series.replace(0, np.nan))
        elif code == 5:
            transformed[col] = np.log(series.replace(0, np.nan)).diff()
        elif code == 6:
            transformed[col] = np.log(series.replace(0, np.nan)).diff().diff()
        elif code == 7:
            pct_change = series / series.shift(1) - 1.0
            transformed[col] = pct_change.diff()
        else:
            transformed[col] = series
    
    return transformed


def reverse_transformation_single(forecast_val, x_t1, x_t2, code):
    """Reverse transformation for a single forecast value."""
    try:
        if pd.isna(forecast_val) or pd.isna(x_t1):
            return np.nan
        if code == 1:
            return forecast_val
        elif code == 2:
            return x_t1 + forecast_val
        elif code == 3:
            return 2 * x_t1 - x_t2 + forecast_val
        elif code == 4:
            return np.exp(forecast_val)
        elif code == 5:
            return x_t1 * np.exp(forecast_val) if x_t1 > 0 else x_t1
        elif code == 6:
            if x_t1 > 0 and x_t2 > 0:
                dlog_t1 = np.log(x_t1) - np.log(x_t2)
                return x_t1 * np.exp(dlog_t1 + forecast_val)
            return x_t1
        elif code == 7:
            if x_t1 != 0 and x_t2 != 0:
                pct_t1 = x_t1 / x_t2 - 1.0
                return x_t1 * (1 + pct_t1 + forecast_val)
            return x_t1
        else:
            return forecast_val
    except:
        return np.nan


def reverse_transformations(forecasts_transformed, df_original, transform_codes, forecast_dates):
    """Reverse transformations to original scale."""
    forecasts_original = pd.DataFrame(index=forecast_dates, columns=forecasts_transformed.columns)
    
    for col in forecasts_transformed.columns:
        if col not in df_original.columns:
            continue
        try:
            code = int(float(transform_codes[col]))
        except (ValueError, KeyError):
            code = 1
        
        forecast_vals = forecasts_transformed[col].values
        orig_vals = df_original[col].values
        
        for i, date in enumerate(forecast_dates):
            fc_val = forecast_vals[i]
            if i == 0:
                x_t1 = orig_vals[-1] if len(orig_vals) >= 1 else 0
                x_t2 = orig_vals[-2] if len(orig_vals) >= 2 else x_t1
            elif i == 1:
                x_t1 = forecasts_original[col].iloc[i-1]
                x_t2 = orig_vals[-1] if len(orig_vals) >= 1 else x_t1
            else:
                x_t1 = forecasts_original[col].iloc[i-1]
                x_t2 = forecasts_original[col].iloc[i-2]
            
            forecasts_original.loc[date, col] = reverse_transformation_single(fc_val, x_t1, x_t2, code)
    
    return forecasts_original.astype(float)


def prepare_data_for_modeling(df, min_obs_ratio=0.8):
    """Prepare data for factor modeling."""
    df = df.iloc[2:].copy()
    missing_ratio = df.isnull().sum() / len(df)
    valid_cols = missing_ratio[missing_ratio < (1 - min_obs_ratio)].index.tolist()
    df_clean = df[valid_cols].copy()
    df_clean = df_clean.ffill().bfill()
    df_clean = df_clean.dropna(axis=1, how='any')
    
    means = df_clean.mean()
    stds = df_clean.std().replace(0, 1)
    df_standardized = (df_clean - means) / stds
    
    return df_standardized, means, stds, df_clean.columns.tolist()


# =============================================================================
# 2. FACTOR MODELS
# =============================================================================

class EfficientFactorModel:
    """Fast factor model using SVD + OLS AR(1). No MCMC."""
    
    def __init__(self, n_factors=5):
        self.n_factors = n_factors
        
    def fit(self, Y):
        T, N = Y.shape
        K = min(self.n_factors, min(T, N) - 1)
        
        # SVD
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        self.factors = U[:, :K] * S[:K]
        self.loadings = Vt[:K, :].T
        
        # AR(1) for each factor
        self.ar_coefs = np.zeros(K)
        self.ar_intercepts = np.zeros(K)
        self.ar_sigmas = np.zeros(K)
        
        for k in range(K):
            f = self.factors[:, k]
            if len(f) < 3:
                continue
            X = np.column_stack([np.ones(len(f)-1), f[:-1]])
            y = f[1:]
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                self.ar_intercepts[k] = beta[0]
                self.ar_coefs[k] = np.clip(beta[1], -0.99, 0.99)
                self.ar_sigmas[k] = max((y - X @ beta).std(), 0.01)
            except:
                self.ar_coefs[k] = 0.9
                self.ar_sigmas[k] = 0.1
        
        Y_hat = self.factors @ self.loadings.T
        self.residual_std = np.maximum((Y - Y_hat).std(axis=0), 0.01)
        return self
    
    def forecast(self, h=1, n_samples=500):
        N = self.loadings.shape[0]
        K = self.factors.shape[1]
        forecasts_samples = np.zeros((n_samples, h, N))
        
        for i in range(n_samples):
            F_curr = self.factors[-1].copy()
            for step in range(h):
                F_next = self.ar_intercepts + self.ar_coefs * F_curr + self.ar_sigmas * np.random.randn(K)
                Y_fc = self.loadings @ F_next + self.residual_std * np.random.randn(N)
                forecasts_samples[i, step] = Y_fc
                F_curr = F_next
        
        return forecasts_samples.mean(axis=0), forecasts_samples.std(axis=0)


class BayesianFactorModel:
    """
    Bayesian Factor Model using PyMC.
    Uses a simpler formulation that avoids the graph complexity issue.
    """
    
    def __init__(self, n_factors=5):
        self.n_factors = n_factors
        
    def fit(self, Y, n_samples=1000, n_tune=500, cores=1):
        import pymc as pm
        
        T, N = Y.shape
        K = min(self.n_factors, min(T, N) - 1)
        
        print(f"    Fitting Bayesian Factor Model: T={T}, N={N}, K={K}")
        
        with pm.Model() as model:
            # Priors
            Lambda = pm.Normal('Lambda', mu=0, sigma=1, shape=(N, K))
            F = pm.Normal('F', mu=0, sigma=1, shape=(T, K))
            Psi = pm.HalfNormal('Psi', sigma=1, shape=N)
            
            # Likelihood
            mu = pm.math.dot(F, Lambda.T)
            pm.Normal('Y_obs', mu=mu, sigma=Psi, observed=Y)
            
            # Sample
            self.trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                cores=cores,
                return_inferencedata=True,
                progressbar=True,
                target_accept=0.9
            )
        
        # Extract posteriors
        self.loadings = self.trace.posterior['Lambda'].mean(dim=['chain', 'draw']).values
        self.factors = self.trace.posterior['F'].mean(dim=['chain', 'draw']).values
        self.Psi = self.trace.posterior['Psi'].mean(dim=['chain', 'draw']).values
        
        # Fit AR(1) to factors (frequentist for simplicity)
        self.ar_coefs = np.zeros(K)
        self.ar_intercepts = np.zeros(K)
        self.ar_sigmas = np.zeros(K)
        
        for k in range(K):
            f = self.factors[:, k]
            X = np.column_stack([np.ones(len(f)-1), f[:-1]])
            y = f[1:]
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                self.ar_intercepts[k] = beta[0]
                self.ar_coefs[k] = np.clip(beta[1], -0.99, 0.99)
                self.ar_sigmas[k] = max((y - X @ beta).std(), 0.01)
            except:
                self.ar_coefs[k] = 0.9
                self.ar_sigmas[k] = 0.1
        
        return self
    
    def forecast(self, h=1, n_samples=500):
        N = self.loadings.shape[0]
        K = self.factors.shape[1]
        
        # Get posterior samples
        Lambda_samples = self.trace.posterior['Lambda'].values.reshape(-1, N, K)
        Psi_samples = self.trace.posterior['Psi'].values.reshape(-1, N)
        n_posterior = Lambda_samples.shape[0]
        
        forecasts_samples = np.zeros((n_samples, h, N))
        
        for i in range(n_samples):
            idx = np.random.randint(n_posterior)
            Lambda = Lambda_samples[idx]
            Psi = Psi_samples[idx]
            
            F_curr = self.factors[-1].copy()
            for step in range(h):
                F_next = self.ar_intercepts + self.ar_coefs * F_curr + self.ar_sigmas * np.random.randn(K)
                Y_fc = Lambda @ F_next + Psi * np.random.randn(N)
                forecasts_samples[i, step] = Y_fc
                F_curr = F_next
        
        return forecasts_samples.mean(axis=0), forecasts_samples.std(axis=0)


class BayesianARFactorModel:
    """
    Bayesian Factor Model with Bayesian AR(1) on factors.
    Two-step approach to avoid graph complexity.
    """
    
    def __init__(self, n_factors=5):
        self.n_factors = n_factors
        
    def fit(self, Y, n_samples=1000, n_tune=500, cores=1):
        import pymc as pm
        
        T, N = Y.shape
        K = min(self.n_factors, min(T, N) - 1)
        
        print(f"    Fitting Bayesian AR Factor Model: T={T}, N={N}, K={K}")
        
        # Step 1: Extract factors using Bayesian PCA
        print("    Step 1: Bayesian factor extraction...")
        with pm.Model():
            Lambda = pm.Normal('Lambda', mu=0, sigma=1, shape=(N, K))
            F = pm.Normal('F', mu=0, sigma=1, shape=(T, K))
            Psi = pm.HalfNormal('Psi', sigma=1, shape=N)
            mu = pm.math.dot(F, Lambda.T)
            pm.Normal('Y_obs', mu=mu, sigma=Psi, observed=Y)
            
            self.trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                cores=cores,
                return_inferencedata=True,
                progressbar=True,
                target_accept=0.9
            )
        
        self.loadings = self.trace.posterior['Lambda'].mean(dim=['chain', 'draw']).values
        self.factors = self.trace.posterior['F'].mean(dim=['chain', 'draw']).values
        self.Psi = self.trace.posterior['Psi'].mean(dim=['chain', 'draw']).values
        
        # Step 2: Fit Bayesian AR(1) to each factor
        print("    Step 2: Bayesian AR(1) for factors...")
        self.ar_traces = []
        self.ar_coefs = np.zeros(K)
        self.ar_intercepts = np.zeros(K)
        self.ar_sigmas = np.zeros(K)
        
        for k in range(K):
            f = self.factors[:, k]
            f_lag = f[:-1]
            f_now = f[1:]
            
            with pm.Model():
                c = pm.Normal('c', mu=0, sigma=1)
                phi = pm.Uniform('phi', lower=-0.99, upper=0.99)
                sigma = pm.HalfNormal('sigma', sigma=1)
                
                mu_ar = c + phi * f_lag
                pm.Normal('f_obs', mu=mu_ar, sigma=sigma, observed=f_now)
                
                ar_trace = pm.sample(
                    draws=500,
                    tune=250,
                    cores=1,
                    return_inferencedata=True,
                    progressbar=False
                )
            
            self.ar_traces.append(ar_trace)
            self.ar_coefs[k] = float(ar_trace.posterior['phi'].mean().values)
            self.ar_intercepts[k] = float(ar_trace.posterior['c'].mean().values)
            self.ar_sigmas[k] = float(ar_trace.posterior['sigma'].mean().values)
        
        return self
    
    def forecast(self, h=1, n_samples=500):
        N = self.loadings.shape[0]
        K = self.factors.shape[1]
        
        Lambda_samples = self.trace.posterior['Lambda'].values.reshape(-1, N, K)
        Psi_samples = self.trace.posterior['Psi'].values.reshape(-1, N)
        n_posterior = Lambda_samples.shape[0]
        
        forecasts_samples = np.zeros((n_samples, h, N))
        
        for i in range(n_samples):
            idx = np.random.randint(n_posterior)
            Lambda = Lambda_samples[idx]
            Psi = Psi_samples[idx]
            
            F_curr = self.factors[-1].copy()
            
            for step in range(h):
                F_next = np.zeros(K)
                for k in range(K):
                    # Sample from AR posterior
                    ar_post = self.ar_traces[k].posterior
                    ar_size = ar_post['phi'].values.size
                    ar_idx = np.random.randint(ar_size)
                    
                    phi = ar_post['phi'].values.flatten()[ar_idx]
                    c = ar_post['c'].values.flatten()[ar_idx]
                    sigma = ar_post['sigma'].values.flatten()[ar_idx]
                    
                    F_next[k] = c + phi * F_curr[k] + sigma * np.random.randn()
                
                Y_fc = Lambda @ F_next + Psi * np.random.randn(N)
                forecasts_samples[i, step] = Y_fc
                F_curr = F_next
        
        return forecasts_samples.mean(axis=0), forecasts_samples.std(axis=0)


# =============================================================================
# 3. MAIN FORECASTING FUNCTION
# =============================================================================

def run_forecasting(df_standardized, means, stds, df_original, transform_codes,
                    n_factors=5, initial_window=120, forecast_horizon=0,
                    start_date=None, method='efficient', n_samples=1000,
                    n_tune=500, cores=1, verbose=True):
    """
    Run rolling window forecasting with optional future forecasts.
    """
    T = len(df_standardized)
    columns = df_standardized.columns
    N = len(columns)
    
    # Determine start index
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        start_idx = df_standardized.index.get_indexer([start_date], method='bfill')[0]
        if start_idx < initial_window:
            print(f"Warning: Adjusting start to index {initial_window}")
            start_idx = initial_window
    else:
        start_idx = initial_window
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"FORECASTING CONFIGURATION")
        print(f"{'='*60}")
        print(f"  Method: {method}")
        print(f"  Factors: {n_factors}, Variables: {N}, Observations: {T}")
        print(f"  Initial window: {initial_window}")
        print(f"  Rolling forecast periods: {T - start_idx}")
        print(f"  Future forecast horizon: {forecast_horizon}")
        print(f"{'='*60}\n")
    
    forecast_dates = []
    forecast_values = []
    forecast_stds_list = []
    
    # ===================
    # ROLLING FORECASTS
    # ===================
    if verbose:
        print("PHASE 1: Rolling Window Forecasts")
        print("-" * 40)
    
    for t in range(start_idx, T):
        forecast_date = df_standardized.index[t]
        
        if verbose:
            progress = t - start_idx + 1
            total = T - start_idx
            print(f"  [{progress}/{total}] {forecast_date.strftime('%Y-%m')}...", end=" ", flush=True)
        
        Y_train = df_standardized.iloc[:t].values
        
        try:
            if method == 'efficient':
                model = EfficientFactorModel(n_factors=n_factors)
                model.fit(Y_train)
            elif method == 'bayesian':
                model = BayesianFactorModel(n_factors=n_factors)
                model.fit(Y_train, n_samples=n_samples, n_tune=n_tune, cores=cores)
            elif method == 'bayesian-ar':
                model = BayesianARFactorModel(n_factors=n_factors)
                model.fit(Y_train, n_samples=n_samples, n_tune=n_tune, cores=cores)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            fc_mean, fc_std = model.forecast(h=1, n_samples=300)
            forecast_dates.append(forecast_date)
            forecast_values.append(fc_mean[0])
            forecast_stds_list.append(fc_std[0])
            
            if verbose:
                print("Done")
                
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            fallback = df_standardized.iloc[t-1].values * 0.95
            forecast_dates.append(forecast_date)
            forecast_values.append(fallback)
            forecast_stds_list.append(np.ones(N))
    
    # ===================
    # FUTURE FORECASTS
    # ===================
    if forecast_horizon > 0:
        if verbose:
            print(f"\nPHASE 2: Future Forecasts ({forecast_horizon} months)")
            print("-" * 40)
            print(f"  Training on full dataset (T={T})...")
        
        Y_all = df_standardized.values
        
        try:
            if method == 'efficient':
                model = EfficientFactorModel(n_factors=n_factors)
                model.fit(Y_all)
            elif method == 'bayesian':
                model = BayesianFactorModel(n_factors=n_factors)
                model.fit(Y_all, n_samples=n_samples, n_tune=n_tune, cores=cores)
            elif method == 'bayesian-ar':
                model = BayesianARFactorModel(n_factors=n_factors)
                model.fit(Y_all, n_samples=n_samples, n_tune=n_tune, cores=cores)
            
            if verbose:
                print(f"  Generating {forecast_horizon}-step forecasts...")
            
            fc_mean, fc_std = model.forecast(h=forecast_horizon, n_samples=500)
            
            last_date = df_standardized.index[-1]
            for h in range(forecast_horizon):
                future_date = last_date + relativedelta(months=h+1)
                forecast_dates.append(future_date)
                forecast_values.append(fc_mean[h])
                forecast_stds_list.append(fc_std[h])
                if verbose:
                    print(f"    {future_date.strftime('%Y-%m')}: Done")
                    
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            last_date = df_standardized.index[-1]
            last_val = df_standardized.iloc[-1].values
            for h in range(forecast_horizon):
                future_date = last_date + relativedelta(months=h+1)
                forecast_dates.append(future_date)
                forecast_values.append(last_val * (0.9 ** (h+1)))
                forecast_stds_list.append(np.ones(N) * (1 + 0.1*h))
    
    # ===================
    # CREATE OUTPUT
    # ===================
    if verbose:
        print(f"\nPHASE 3: Processing Results")
        print("-" * 40)
    
    forecasts_standardized = pd.DataFrame(forecast_values, index=forecast_dates, columns=columns)
    forecast_stds_df = pd.DataFrame(forecast_stds_list, index=forecast_dates, columns=columns)
    forecasts_transformed = forecasts_standardized * stds + means
    
    if verbose:
        print("  Reversing transformations...")
    
    df_orig_aligned = df_original[columns].copy()
    forecasts_original = reverse_transformations(
        forecasts_transformed, df_orig_aligned, transform_codes, pd.DatetimeIndex(forecast_dates)
    )
    
    if verbose:
        print(f"  Total forecasts: {len(forecasts_original)}")
        print(f"  Date range: {forecast_dates[0]} to {forecast_dates[-1]}")
    
    return forecasts_original, forecasts_transformed, forecasts_standardized, forecast_stds_df


# =============================================================================
# 4. MAIN
# =============================================================================

def main(args):
    """Main entry point."""
    print("=" * 70)
    print("BAYESIAN DYNAMIC FACTOR MODEL FORECASTING")
    print("=" * 70)
    
    # Load
    print(f"\n[1/4] Loading: {args.input}")
    df_raw, transform_codes = load_and_preprocess_data(args.input)
    print(f"  Shape: {df_raw.shape}, Range: {df_raw.index[0]} to {df_raw.index[-1]}")
    df_original = df_raw.copy()
    
    # Transform
    print("\n[2/4] Applying transformations...")
    df_transformed = apply_transformations(df_raw, transform_codes)
    
    # Prepare
    print("\n[3/4] Preparing data...")
    df_standardized, means, stds, valid_cols = prepare_data_for_modeling(
        df_transformed, min_obs_ratio=args.min_obs_ratio
    )
    print(f"  Clean shape: {df_standardized.shape}, Variables: {len(valid_cols)}")
    
    # Forecast
    print("\n[4/4] Running forecasts...")
    forecasts_original, forecasts_transformed, forecasts_standardized, forecast_stds = run_forecasting(
        df_standardized, means, stds, df_original, transform_codes,
        n_factors=args.n_factors,
        initial_window=args.initial_window,
        forecast_horizon=args.forecast_horizon,
        start_date=args.start_date,
        method=args.method,
        n_samples=args.n_samples,
        n_tune=args.n_tune,
        cores=args.cores,
        verbose=not args.quiet
    )
    
    # Save
    print(f"\nSaving to: {args.output}")
    output_df = pd.DataFrame(index=forecasts_original.index, columns=df_raw.columns)
    for col in forecasts_original.columns:
        if col in output_df.columns:
            output_df[col] = forecasts_original[col]
    
    try:
        output_df.index = output_df.index.strftime('%-m/%-d/%Y')
    except ValueError:
        output_df.index = output_df.index.strftime('%#m/%#d/%Y')
    output_df.index.name = 'sasdate'
    
    transform_row = transform_codes[output_df.columns].to_frame().T
    transform_row.index = ['Transform:']
    final_output = pd.concat([transform_row, output_df])
    final_output.to_csv(args.output)
    
    if args.save_std:
        std_path = args.output.replace('.csv', '_std.csv')
        try:
            forecast_stds.index = forecast_stds.index.strftime('%-m/%-d/%Y')
        except ValueError:
            forecast_stds.index = forecast_stds.index.strftime('%#m/%#d/%Y')
        forecast_stds.index.name = 'sasdate'
        forecast_stds.to_csv(std_path)
        print(f"  Std saved to: {std_path}")
    
    if args.save_transformed:
        trans_path = args.output.replace('.csv', '_transformed.csv')
        try:
            forecasts_transformed.index = forecasts_transformed.index.strftime('%-m/%-d/%Y')
        except ValueError:
            forecasts_transformed.index = forecasts_transformed.index.strftime('%#m/%#d/%Y')
        forecasts_transformed.index.name = 'sasdate'
        forecasts_transformed.to_csv(trans_path)
        print(f"  Transformed saved to: {trans_path}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    return final_output, forecasts_original, forecasts_transformed


# =============================================================================
# 5. ARGUMENT PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Bayesian Dynamic Factor Model Forecasting for FRED-MD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast rolling forecasts (SVD + OLS)
  python forecast.py -i 2025-12-MD.csv -o forecasts.csv

  # Rolling + 12 months future
  python forecast.py -i 2025-12-MD.csv -o forecasts.csv -f 12

  # Start from 2020
  python forecast.py -i 2025-12-MD.csv -o forecasts.csv --start-date 2020-01-01

  # Bayesian method
  python forecast.py -i 2025-12-MD.csv -o forecasts.csv --method bayesian

  # Bayesian with Bayesian AR on factors
  python forecast.py -i 2025-12-MD.csv -o forecasts.csv --method bayesian-ar

Methods:
  efficient   - SVD + OLS AR(1). Fast, no MCMC.
  bayesian    - Bayesian PCA + frequentist AR(1). Medium.
  bayesian-ar - Bayesian PCA + Bayesian AR(1). Slowest, full uncertainty.
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True, help='Input CSV')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output CSV')
    parser.add_argument('--n-factors', '-k', type=int, default=5, help='Factors (default: 5)')
    parser.add_argument('--initial-window', '-w', type=int, default=120, help='Window (default: 120)')
    parser.add_argument('--forecast-horizon', '-f', type=int, default=0, help='Future months (default: 0)')
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--method', '-m', type=str, default='efficient',
                        choices=['efficient', 'bayesian', 'bayesian-ar'], help='Method')
    parser.add_argument('--n-samples', '-s', type=int, default=1000, help='MCMC samples')
    parser.add_argument('--n-tune', '-t', type=int, default=500, help='MCMC tune')
    parser.add_argument('--cores', '-c', type=int, default=1, help='CPU cores')
    parser.add_argument('--min-obs-ratio', type=float, default=0.8, help='Min obs ratio')
    parser.add_argument('--save-std', action='store_true', help='Save std devs')
    parser.add_argument('--save-transformed', action='store_true', help='Save transformed')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)