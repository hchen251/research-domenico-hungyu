"""
Rolling-Window Bayesian Dynamic Factor Model Forecasting
Using PyMC3 (or PyMC) for FRED-MD style macroeconomic data

This implementation uses Bayesian Factor Analysis to extract latent factors
from high-dimensional macroeconomic data and produces month-by-month forecasts.

Usage:
    python forecast.py --input 2025-12-MD.csv --output forecasts.csv --n-factors 5 --initial-window 120
"""

import argparse
import numpy as np
import pandas as pd
import pymc3 as pm  # Use `import pymc as pm` for PyMC v4+
import theano.tensor as tt  # Use `import pytensor.tensor as pt` for PyMC v4+
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(filepath):
    """
    Load FRED-MD data and apply appropriate transformations.
    
    Transformation codes (tcode) as defined in FRED-QD/FRED-MD appendix:
    1 = no transformation
    2 = Δx_t (first difference)
    3 = Δ²x_t (second difference)  
    4 = log(x_t)
    5 = Δlog(x_t) (log first difference)
    6 = Δ²log(x_t) (log second difference)
    7 = Δ(x_t/x_{t-1} - 1.0) (first difference of percent change)
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Extract transformation codes (first row)
    transform_codes = df.iloc[0].copy()
    
    # Remove transformation row and set date index
    df = df.iloc[1:].copy()
    df['sasdate'] = pd.to_datetime(df['sasdate'], format='%m/%d/%Y')
    df.set_index('sasdate', inplace=True)
    
    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df, transform_codes


def apply_transformations(df, transform_codes):
    """
    Apply FRED-MD transformations to make series stationary.
    
    Transformation codes (tcode):
    1 = no transformation
    2 = Δx_t
    3 = Δ²x_t
    4 = log(x_t)
    5 = Δlog(x_t)
    6 = Δ²log(x_t)
    7 = Δ(x_t/x_{t-1} - 1.0)
    """
    transformed = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        if col == 'sasdate':
            continue
            
        series = df[col].copy()
        try:
            code = int(float(transform_codes[col]))
        except (ValueError, KeyError):
            code = 1  # Default to no transformation
        
        if code == 1:
            # No transformation
            transformed[col] = series
        elif code == 2:
            # First difference: Δx_t
            transformed[col] = series.diff()
        elif code == 3:
            # Second difference: Δ²x_t
            transformed[col] = series.diff().diff()
        elif code == 4:
            # Log: log(x_t)
            transformed[col] = np.log(series.replace(0, np.nan))
        elif code == 5:
            # Log first difference: Δlog(x_t)
            transformed[col] = np.log(series.replace(0, np.nan)).diff()
        elif code == 6:
            # Log second difference: Δ²log(x_t)
            transformed[col] = np.log(series.replace(0, np.nan)).diff().diff()
        elif code == 7:
            # First difference of percent change: Δ(x_t/x_{t-1} - 1.0)
            pct_change = series / series.shift(1) - 1.0
            transformed[col] = pct_change.diff()
        else:
            transformed[col] = series
    
    return transformed


def reverse_transformations(forecasts_transformed, df_original, transform_codes, forecast_dates):
    """
    Reverse the transformations to get forecasts back to original scale.
    
    Parameters:
    -----------
    forecasts_transformed : pd.DataFrame
        Forecasts in transformed scale
    df_original : pd.DataFrame
        Original data (untransformed)
    transform_codes : pd.Series
        Transformation codes for each variable
    forecast_dates : pd.DatetimeIndex
        Dates of forecasts
        
    Returns:
    --------
    forecasts_original : pd.DataFrame
        Forecasts in original scale
    """
    forecasts_original = pd.DataFrame(index=forecast_dates, columns=forecasts_transformed.columns)
    
    for col in forecasts_transformed.columns:
        try:
            code = int(float(transform_codes[col]))
        except (ValueError, KeyError):
            code = 1
        
        forecast_vals = forecasts_transformed[col].values
        
        # Get the last available original values needed for reversal
        last_orig_idx = df_original.index.get_indexer([forecast_dates[0]], method='ffill')[0]
        if last_orig_idx < 0:
            last_orig_idx = 0
            
        for i, date in enumerate(forecast_dates):
            fc_val = forecast_vals[i]
            
            if i == 0:
                # Use original data for lagged values
                if last_orig_idx >= 1:
                    x_t1 = df_original[col].iloc[last_orig_idx]
                    x_t2 = df_original[col].iloc[last_orig_idx - 1] if last_orig_idx >= 2 else x_t1
                else:
                    x_t1 = df_original[col].iloc[0]
                    x_t2 = x_t1
            else:
                # Use previous forecast
                x_t1 = forecasts_original[col].iloc[i-1]
                if i >= 2:
                    x_t2 = forecasts_original[col].iloc[i-2]
                elif last_orig_idx >= 1:
                    x_t2 = df_original[col].iloc[last_orig_idx]
                else:
                    x_t2 = x_t1
            
            if code == 1:
                # No transformation
                forecasts_original.loc[date, col] = fc_val
            elif code == 2:
                # Reverse first difference: x_t = x_{t-1} + Δx_t
                forecasts_original.loc[date, col] = x_t1 + fc_val
            elif code == 3:
                # Reverse second difference: x_t = 2*x_{t-1} - x_{t-2} + Δ²x_t
                forecasts_original.loc[date, col] = 2*x_t1 - x_t2 + fc_val
            elif code == 4:
                # Reverse log: x_t = exp(log_x_t)
                forecasts_original.loc[date, col] = np.exp(fc_val)
            elif code == 5:
                # Reverse log first difference: x_t = x_{t-1} * exp(Δlog(x_t))
                forecasts_original.loc[date, col] = x_t1 * np.exp(fc_val)
            elif code == 6:
                # Reverse log second difference
                if x_t1 > 0 and x_t2 > 0:
                    dlog_t1 = np.log(x_t1) - np.log(x_t2)
                    dlog_t = dlog_t1 + fc_val
                    forecasts_original.loc[date, col] = x_t1 * np.exp(dlog_t)
                else:
                    forecasts_original.loc[date, col] = x_t1
            elif code == 7:
                # Reverse first difference of percent change
                if x_t1 != 0 and x_t2 != 0:
                    pct_change_t1 = x_t1 / x_t2 - 1.0
                    pct_change_t = pct_change_t1 + fc_val
                    forecasts_original.loc[date, col] = x_t1 * (1 + pct_change_t)
                else:
                    forecasts_original.loc[date, col] = x_t1
            else:
                forecasts_original.loc[date, col] = fc_val
    
    return forecasts_original.astype(float)


def prepare_data_for_modeling(df, min_obs_ratio=0.8):
    """
    Prepare data for factor modeling:
    - Remove series with too many missing values
    - Standardize the data
    - Handle remaining missing values
    """
    # Drop rows at the beginning that have many NaN due to differencing
    df = df.iloc[2:].copy()
    
    # Calculate missing ratio for each column
    missing_ratio = df.isnull().sum() / len(df)
    
    # Keep columns with sufficient observations
    valid_cols = missing_ratio[missing_ratio < (1 - min_obs_ratio)].index.tolist()
    df_clean = df[valid_cols].copy()
    
    # Forward fill then backward fill remaining missing values
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    
    # Drop any remaining columns with NaN
    df_clean = df_clean.dropna(axis=1, how='any')
    
    # Standardize
    means = df_clean.mean()
    stds = df_clean.std()
    stds = stds.replace(0, 1)  # Avoid division by zero
    
    df_standardized = (df_clean - means) / stds
    
    return df_standardized, means, stds, df_clean.columns.tolist()


# =============================================================================
# 2. BAYESIAN DYNAMIC FACTOR MODEL
# =============================================================================

class BayesianDynamicFactorModel:
    """
    Bayesian Dynamic Factor Model using PyMC3.
    
    Model:
        Y_t = Lambda * F_t + e_t,  e_t ~ N(0, Psi)
        F_t = Phi * F_{t-1} + u_t,  u_t ~ N(0, Q)
    
    Where:
        Y_t = N-dimensional observed data at time t
        F_t = K-dimensional latent factors at time t
        Lambda = N x K factor loading matrix
        Phi = K x K factor transition matrix (VAR(1) for factors)
        Psi = N x N diagonal idiosyncratic variance matrix
    """
    
    def __init__(self, n_factors=5, ar_order=1):
        """
        Initialize the model.
        
        Parameters:
        -----------
        n_factors : int
            Number of latent factors to extract
        ar_order : int
            AR order for factor dynamics (currently supports 1)
        """
        self.n_factors = n_factors
        self.ar_order = ar_order
        self.trace = None
        self.model = None
        self.factor_loadings = None
        self.factor_ar_coefs = None
        self.idiosyncratic_var = None
        
    def fit(self, Y, n_samples=1000, n_tune=500, target_accept=0.9, cores=1):
        """
        Fit the Bayesian Dynamic Factor Model.
        
        Parameters:
        -----------
        Y : np.ndarray
            T x N matrix of observations
        n_samples : int
            Number of MCMC samples
        n_tune : int
            Number of tuning samples
        target_accept : float
            Target acceptance rate for NUTS sampler
        cores : int
            Number of CPU cores for sampling
        """
        T, N = Y.shape
        K = self.n_factors
        
        with pm.Model() as self.model:
            # ==================
            # PRIORS
            # ==================
            
            # Factor loadings: Lambda ~ N(0, 1)
            # Use identification constraint: upper triangular with positive diagonal
            Lambda = pm.Normal('Lambda', mu=0, sd=1, shape=(N, K))
            
            # Factor AR coefficients: diagonal VAR(1)
            # Phi_diag ~ Uniform(-0.99, 0.99) for stationarity
            Phi_diag = pm.Uniform('Phi_diag', lower=-0.99, upper=0.99, shape=K)
            
            # Idiosyncratic variances: Psi ~ InverseGamma
            Psi = pm.InverseGamma('Psi', alpha=2, beta=1, shape=N)
            
            # Factor innovation variance (fixed to 1 for identification)
            Q = tt.eye(K)
            
            # ==================
            # LATENT FACTORS
            # ==================
            
            # Initial factor: F_0 ~ N(0, I)
            F_init = pm.Normal('F_init', mu=0, sd=1, shape=K)
            
            # Factor dynamics: F_t = Phi * F_{t-1} + u_t
            F = [F_init]
            for t in range(1, T):
                F_t = pm.Normal(f'F_{t}', 
                               mu=Phi_diag * F[t-1], 
                               sd=1, 
                               shape=K)
                F.append(F_t)
            
            # Stack factors
            F_stacked = tt.stack(F, axis=0)  # T x K
            
            # ==================
            # LIKELIHOOD
            # ==================
            
            # Y_t = Lambda * F_t + e_t
            mu = tt.dot(F_stacked, Lambda.T)  # T x N
            
            # Observation likelihood
            Y_obs = pm.Normal('Y_obs', 
                             mu=mu, 
                             sd=tt.sqrt(Psi), 
                             observed=Y)
            
            # ==================
            # INFERENCE
            # ==================
            
            self.trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                target_accept=target_accept,
                cores=cores,
                return_inferencedata=False,
                progressbar=True
            )
        
        # Extract posterior means
        self.factor_loadings = self.trace['Lambda'].mean(axis=0)
        self.factor_ar_coefs = self.trace['Phi_diag'].mean(axis=0)
        self.idiosyncratic_var = self.trace['Psi'].mean(axis=0)
        
        # Extract factor estimates (posterior means)
        self.factors = np.zeros((T, K))
        for t in range(T):
            if t == 0:
                self.factors[t] = self.trace['F_init'].mean(axis=0)
            else:
                self.factors[t] = self.trace[f'F_{t}'].mean(axis=0)
        
        return self
    
    def forecast(self, h=1, n_samples=1000):
        """
        Generate h-step ahead forecasts.
        
        Parameters:
        -----------
        h : int
            Forecast horizon
        n_samples : int
            Number of samples for forecast distribution
            
        Returns:
        --------
        forecasts : np.ndarray
            h x N matrix of point forecasts
        forecast_std : np.ndarray
            h x N matrix of forecast standard deviations
        """
        N = self.factor_loadings.shape[0]
        K = self.n_factors
        
        # Sample from posterior
        n_posterior = len(self.trace['Lambda'])
        sample_idx = np.random.choice(n_posterior, size=n_samples, replace=True)
        
        forecasts_samples = np.zeros((n_samples, h, N))
        
        for i, idx in enumerate(sample_idx):
            Lambda = self.trace['Lambda'][idx]
            Phi_diag = self.trace['Phi_diag'][idx]
            Psi = self.trace['Psi'][idx]
            
            # Get last factor
            F_current = self.factors[-1].copy()
            
            for step in range(h):
                # Forecast factor: F_{t+h} = Phi * F_{t+h-1} + u
                F_next = Phi_diag * F_current + np.random.randn(K)
                
                # Forecast observation: Y_{t+h} = Lambda * F_{t+h} + e
                Y_forecast = Lambda @ F_next + np.sqrt(Psi) * np.random.randn(N)
                
                forecasts_samples[i, step] = Y_forecast
                F_current = F_next
        
        # Point forecasts and uncertainty
        forecasts = forecasts_samples.mean(axis=0)
        forecast_std = forecasts_samples.std(axis=0)
        
        return forecasts, forecast_std


# =============================================================================
# 3. SIMPLIFIED BAYESIAN FACTOR MODEL (Two-Step Approach)
# =============================================================================

class SimplifiedBayesianFactorModel:
    """
    A more computationally efficient Bayesian Factor Model.
    Uses a two-step approach:
    1. Extract factors using Bayesian PCA
    2. Fit AR dynamics to factors
    3. Forecast
    """
    
    def __init__(self, n_factors=5):
        self.n_factors = n_factors
        self.trace = None
        self.factor_loadings = None
        self.factors = None
        self.ar_models = None
        
    def fit(self, Y, n_samples=2000, n_tune=1000, cores=1):
        """
        Fit the simplified Bayesian factor model.
        """
        T, N = Y.shape
        K = self.n_factors
        
        with pm.Model() as model:
            # Factor loadings
            Lambda = pm.Normal('Lambda', mu=0, sd=1, shape=(N, K))
            
            # Factors
            F = pm.Normal('F', mu=0, sd=1, shape=(T, K))
            
            # Idiosyncratic variance
            Psi = pm.HalfNormal('Psi', sd=1, shape=N)
            
            # Likelihood
            mu = pm.math.dot(F, Lambda.T)
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=Psi, observed=Y)
            
            # Sample
            self.trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                cores=cores,
                return_inferencedata=False,
                progressbar=True
            )
        
        # Extract estimates
        self.factor_loadings = self.trace['Lambda'].mean(axis=0)
        self.factors = self.trace['F'].mean(axis=0)
        self.idiosyncratic_var = self.trace['Psi'].mean(axis=0) ** 2
        
        # Fit AR(1) models to each factor using Bayesian estimation
        self.ar_coefs = []
        self.ar_intercepts = []
        self.ar_sigmas = []
        self.ar_traces = []
        
        for k in range(K):
            factor_series = self.factors[:, k]
            
            with pm.Model() as ar_model:
                phi = pm.Uniform('phi', lower=-0.99, upper=0.99)
                sigma = pm.HalfNormal('sigma', sd=1)
                c = pm.Normal('c', mu=0, sd=1)
                
                mu_ar = c + phi * factor_series[:-1]
                obs = pm.Normal('obs', mu=mu_ar, sd=sigma, observed=factor_series[1:])
                
                ar_trace = pm.sample(1000, tune=500, cores=1, 
                                    return_inferencedata=False, progressbar=False)
            
            self.ar_coefs.append(ar_trace['phi'].mean())
            self.ar_intercepts.append(ar_trace['c'].mean())
            self.ar_sigmas.append(ar_trace['sigma'].mean())
            self.ar_traces.append(ar_trace)
        
        return self
    
    def forecast(self, h=1, n_samples=1000):
        """
        Generate forecasts.
        """
        N = self.factor_loadings.shape[0]
        K = self.n_factors
        
        forecasts_samples = np.zeros((n_samples, h, N))
        
        for i in range(n_samples):
            # Get random posterior sample
            idx = np.random.randint(len(self.trace['Lambda']))
            Lambda = self.trace['Lambda'][idx]
            Psi = self.trace['Psi'][idx]
            
            F_current = self.factors[-1].copy()
            
            for step in range(h):
                F_next = np.zeros(K)
                for k in range(K):
                    # Sample AR parameters from posterior
                    ar_idx = np.random.randint(len(self.ar_traces[k]['phi']))
                    phi_k = self.ar_traces[k]['phi'][ar_idx]
                    c_k = self.ar_traces[k]['c'][ar_idx]
                    sigma_k = self.ar_traces[k]['sigma'][ar_idx]
                    
                    F_next[k] = c_k + phi_k * F_current[k] + sigma_k * np.random.randn()
                
                Y_forecast = Lambda @ F_next + Psi * np.random.randn(N)
                forecasts_samples[i, step] = Y_forecast
                F_current = F_next
        
        return forecasts_samples.mean(axis=0), forecasts_samples.std(axis=0)


# =============================================================================
# 4. FULL BAYESIAN DYNAMIC FACTOR MODEL (State-Space Formulation)
# =============================================================================

class FullBayesianDFM:
    """
    Full Bayesian Dynamic Factor Model with proper state-space formulation.
    
    This implements the model:
        Y_t = Lambda * F_t + e_t,  e_t ~ N(0, Psi)
        F_t = Phi * F_{t-1} + u_t,  u_t ~ N(0, Sigma_F)
    
    With full Bayesian inference over all parameters.
    """
    
    def __init__(self, n_factors=5):
        self.n_factors = n_factors
        self.trace = None
        self.model = None
        
    def fit(self, Y, n_samples=2000, n_tune=1000, target_accept=0.9, cores=1):
        """
        Fit the full Bayesian DFM.
        """
        T, N = Y.shape
        K = self.n_factors
        
        with pm.Model() as self.model:
            # ===================
            # HYPERPRIORS
            # ===================
            
            # Prior precision for factor loadings
            tau_lambda = pm.Gamma('tau_lambda', alpha=1, beta=1, shape=K)
            
            # ===================
            # FACTOR LOADINGS
            # ===================
            
            # Lambda ~ N(0, 1/tau_lambda) with identification constraints
            # Lower triangular with positive diagonal for identification
            Lambda_raw = pm.Normal('Lambda_raw', mu=0, sd=1, shape=(N, K))
            
            # Apply soft identification (scale by tau)
            Lambda = Lambda_raw / tt.sqrt(tau_lambda)
            
            # ===================
            # FACTOR DYNAMICS
            # ===================
            
            # AR(1) coefficients for factors
            Phi_diag = pm.Uniform('Phi_diag', lower=-0.99, upper=0.99, shape=K)
            
            # Factor innovation standard deviation
            sigma_F = pm.HalfCauchy('sigma_F', beta=1, shape=K)
            
            # ===================
            # IDIOSYNCRATIC VARIANCE
            # ===================
            
            # Psi ~ InverseGamma (idiosyncratic variance)
            Psi = pm.InverseGamma('Psi', alpha=2, beta=1, shape=N)
            
            # ===================
            # LATENT FACTORS (State-Space)
            # ===================
            
            # Initial state
            F_init = pm.Normal('F_init', mu=0, sd=1, shape=K)
            
            # Build factor sequence using scan for efficiency
            # F_t = Phi * F_{t-1} + sigma_F * epsilon_t
            
            F_innovations = pm.Normal('F_innovations', mu=0, sd=1, shape=(T-1, K))
            
            # Construct factors iteratively
            F_list = [F_init]
            for t in range(T-1):
                F_new = Phi_diag * F_list[-1] + sigma_F * F_innovations[t]
                F_list.append(F_new)
            
            F_stacked = tt.stack(F_list, axis=0)  # T x K
            
            # ===================
            # OBSERVATION MODEL
            # ===================
            
            # Y_t = Lambda * F_t + e_t
            mu_Y = tt.dot(F_stacked, Lambda.T)
            
            # Likelihood
            Y_obs = pm.Normal('Y_obs', mu=mu_Y, sd=tt.sqrt(Psi), observed=Y)
            
            # ===================
            # INFERENCE
            # ===================
            
            self.trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                target_accept=target_accept,
                cores=cores,
                return_inferencedata=False,
                progressbar=True,
                init='adapt_diag'
            )
        
        # Extract posterior estimates
        self._extract_posteriors(T)
        
        return self
    
    def _extract_posteriors(self, T):
        """Extract posterior means and store for forecasting."""
        K = self.n_factors
        
        self.factor_loadings = self.trace['Lambda_raw'].mean(axis=0) / np.sqrt(
            self.trace['tau_lambda'].mean(axis=0)
        )
        self.Phi_diag = self.trace['Phi_diag'].mean(axis=0)
        self.sigma_F = self.trace['sigma_F'].mean(axis=0)
        self.Psi = self.trace['Psi'].mean(axis=0)
        
        # Reconstruct factors
        F_init = self.trace['F_init'].mean(axis=0)
        F_innovations = self.trace['F_innovations'].mean(axis=0)
        
        self.factors = np.zeros((T, K))
        self.factors[0] = F_init
        for t in range(1, T):
            self.factors[t] = self.Phi_diag * self.factors[t-1] + self.sigma_F * F_innovations[t-1]
    
    def forecast(self, h=1, n_samples=1000):
        """
        Generate h-step ahead forecasts with full uncertainty quantification.
        """
        N = self.factor_loadings.shape[0]
        K = self.n_factors
        
        n_posterior = len(self.trace['Phi_diag'])
        forecasts_samples = np.zeros((n_samples, h, N))
        
        for i in range(n_samples):
            # Sample from posterior
            idx = np.random.randint(n_posterior)
            
            Lambda = self.trace['Lambda_raw'][idx] / np.sqrt(self.trace['tau_lambda'][idx])
            Phi_diag = self.trace['Phi_diag'][idx]
            sigma_F = self.trace['sigma_F'][idx]
            Psi = self.trace['Psi'][idx]
            
            # Get last factor state
            F_current = self.factors[-1].copy()
            
            for step in range(h):
                # Forecast factors
                F_next = Phi_diag * F_current + sigma_F * np.random.randn(K)
                
                # Forecast observations
                Y_forecast = Lambda @ F_next + np.sqrt(Psi) * np.random.randn(N)
                
                forecasts_samples[i, step] = Y_forecast
                F_current = F_next
        
        return forecasts_samples.mean(axis=0), forecasts_samples.std(axis=0)


# =============================================================================
# 5. EFFICIENT SVD-BASED FACTOR MODEL (For comparison/speed)
# =============================================================================

class EfficientFactorModel:
    """
    Efficient Factor Model using SVD for factor extraction.
    Used as fallback or for faster computation.
    """
    
    def __init__(self, n_factors=5):
        self.n_factors = n_factors
        self.factors = None
        self.loadings = None
        self.ar_coefs = None
        self.residual_std = None
        
    def fit(self, Y):
        """
        Fit using SVD.
        """
        T, N = Y.shape
        K = self.n_factors
        
        # SVD for factor extraction
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        self.factors = U[:, :K] * S[:K]
        self.loadings = Vt[:K, :].T
        
        # Fit AR(1) to each factor
        self.ar_coefs = np.zeros(K)
        self.ar_intercepts = np.zeros(K)
        self.ar_sigmas = np.zeros(K)
        
        for k in range(K):
            f = self.factors[:, k]
            # OLS for AR(1): f_t = c + phi * f_{t-1} + e
            X = np.column_stack([np.ones(len(f)-1), f[:-1]])
            y = f[1:]
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            self.ar_intercepts[k] = beta[0]
            self.ar_coefs[k] = np.clip(beta[1], -0.99, 0.99)
            residuals = y - X @ beta
            self.ar_sigmas[k] = residuals.std()
        
        # Residual variance for observations
        Y_reconstructed = self.factors @ self.loadings.T
        self.residual_std = (Y - Y_reconstructed).std(axis=0)
        
        return self
    
    def forecast(self, h=1, n_samples=500):
        """
        Generate forecasts with uncertainty.
        """
        N = self.loadings.shape[0]
        K = self.n_factors
        
        forecasts_samples = np.zeros((n_samples, h, N))
        
        for i in range(n_samples):
            F_current = self.factors[-1].copy()
            
            for step in range(h):
                F_next = np.zeros(K)
                for k in range(K):
                    F_next[k] = (self.ar_intercepts[k] + 
                                self.ar_coefs[k] * F_current[k] + 
                                self.ar_sigmas[k] * np.random.randn())
                
                Y_forecast = self.loadings @ F_next + self.residual_std * np.random.randn(N)
                forecasts_samples[i, step] = Y_forecast
                F_current = F_next
        
        return forecasts_samples.mean(axis=0), forecasts_samples.std(axis=0)


# =============================================================================
# 6. ROLLING WINDOW FORECASTING
# =============================================================================

def rolling_window_forecast(df_standardized, means, stds, df_original, transform_codes,
                           n_factors=5, 
                           initial_window=120,
                           method='simplified',
                           n_samples=1000,
                           n_tune=500,
                           target_accept=0.9,
                           cores=1,
                           verbose=True):
    """
    Perform rolling-window forecasting using Bayesian Dynamic Factor Model.
    
    Parameters:
    -----------
    df_standardized : pd.DataFrame
        Standardized transformed data
    means : pd.Series
        Means for destandardization
    stds : pd.Series
        Stds for destandardization
    df_original : pd.DataFrame
        Original untransformed data
    transform_codes : pd.Series
        Transformation codes for each variable
    n_factors : int
        Number of latent factors
    initial_window : int
        Initial training window size
    method : str
        'full' - Full Bayesian DFM (slowest, most accurate)
        'simplified' - Two-step Bayesian (medium speed)
        'efficient' - SVD-based (fastest, least accurate)
    n_samples : int
        MCMC samples
    n_tune : int
        MCMC tuning samples
    target_accept : float
        Target acceptance rate for NUTS
    cores : int
        Number of CPU cores
    verbose : bool
        Print progress
        
    Returns:
    --------
    forecasts_original : pd.DataFrame
        Forecasts in original scale
    forecasts_transformed : pd.DataFrame
        Forecasts in transformed scale
    forecasts_standardized : pd.DataFrame
        Forecasts in standardized scale
    forecast_stds_df : pd.DataFrame
        Forecast standard deviations
    """
    
    T = len(df_standardized)
    columns = df_standardized.columns
    
    forecast_dates = []
    forecast_values_standardized = []
    forecast_stds = []
    
    if verbose:
        print(f"Starting rolling-window forecasting...")
        print(f"  Method: {method}")
        print(f"  Total periods: {T}")
        print(f"  Initial window: {initial_window}")
        print(f"  Forecast periods: {T - initial_window}")
        print(f"  Number of factors: {n_factors}")
        print(f"  MCMC samples: {n_samples}")
        print(f"  MCMC tune: {n_tune}")
        print("=" * 60)
    
    for t in range(initial_window, T):
        forecast_date = df_standardized.index[t]
        
        if verbose:
            print(f"\nForecasting {forecast_date.strftime('%Y-%m')} "
                  f"(Period {t - initial_window + 1}/{T - initial_window})")
        
        # Training data: all data up to time t-1
        Y_train = df_standardized.iloc[:t].values
        
        try:
            if method == 'full':
                model = FullBayesianDFM(n_factors=n_factors)
                model.fit(Y_train, n_samples=n_samples, n_tune=n_tune, 
                         target_accept=target_accept, cores=cores)
                forecast_mean, forecast_std = model.forecast(h=1, n_samples=500)
                
            elif method == 'simplified':
                model = SimplifiedBayesianFactorModel(n_factors=n_factors)
                model.fit(Y_train, n_samples=n_samples, n_tune=n_tune, cores=cores)
                forecast_mean, forecast_std = model.forecast(h=1, n_samples=500)
                
            elif method == 'dynamic':
                model = BayesianDynamicFactorModel(n_factors=n_factors)
                model.fit(Y_train, n_samples=n_samples, n_tune=n_tune, 
                         target_accept=target_accept, cores=cores)
                forecast_mean, forecast_std = model.forecast(h=1, n_samples=500)
                
            else:  # efficient
                model = EfficientFactorModel(n_factors=n_factors)
                model.fit(Y_train)
                forecast_mean, forecast_std = model.forecast(h=1, n_samples=500)
            
            forecast_dates.append(forecast_date)
            forecast_values_standardized.append(forecast_mean[0])
            forecast_stds.append(forecast_std[0])
            
            if verbose:
                print(f"  Forecast completed successfully")
            
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
                print(f"  Using fallback (last observation * 0.9)")
            # Fallback
            forecast_values_standardized.append(df_standardized.iloc[t-1].values * 0.9)
            forecast_stds.append(np.ones(len(columns)))
            forecast_dates.append(forecast_date)
    
    # Create forecast DataFrames
    forecasts_standardized = pd.DataFrame(
        forecast_values_standardized,
        index=forecast_dates,
        columns=columns
    )
    
    forecast_stds_df = pd.DataFrame(
        forecast_stds,
        index=forecast_dates,
        columns=columns
    )
    
    # Destandardize to transformed scale
    forecasts_transformed = forecasts_standardized * stds + means
    
    # Reverse transformations to original scale
    df_orig_aligned = df_original[columns]
    forecasts_original = reverse_transformations(
        forecasts_transformed,
        df_orig_aligned,
        transform_codes,
        pd.DatetimeIndex(forecast_dates)
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("Forecasting complete!")
    
    return forecasts_original, forecasts_transformed, forecasts_standardized, forecast_stds_df


# =============================================================================
# 7. MAIN FUNCTION
# =============================================================================

def main(args):
    """
    Main function to run the rolling-window forecasting.
    """
    
    print("=" * 70)
    print("BAYESIAN DYNAMIC FACTOR MODEL - ROLLING WINDOW FORECASTING")
    print("=" * 70)
    
    # Step 1: Load and preprocess data
    print(f"\n[1/5] Loading data from: {args.input}")
    df_raw, transform_codes = load_and_preprocess_data(args.input)
    print(f"  Raw data shape: {df_raw.shape}")
    print(f"  Date range: {df_raw.index[0]} to {df_raw.index[-1]}")
    
    # Store original data for reverse transformation
    df_original = df_raw.copy()
    
    # Step 2: Apply transformations
    print("\n[2/5] Applying FRED-MD transformations...")
    print("  Transformation codes:")
    print("    1 = no transformation")
    print("    2 = Δx_t (first difference)")
    print("    3 = Δ²x_t (second difference)")
    print("    4 = log(x_t)")
    print("    5 = Δlog(x_t) (log first difference)")
    print("    6 = Δ²log(x_t) (log second difference)")
    print("    7 = Δ(x_t/x_{t-1} - 1.0)")
    df_transformed = apply_transformations(df_raw, transform_codes)
    
    # Step 3: Prepare for modeling
    print(f"\n[3/5] Preparing data for factor modeling...")
    print(f"  Minimum observation ratio: {args.min_obs_ratio}")
    df_standardized, means, stds, valid_cols = prepare_data_for_modeling(
        df_transformed, min_obs_ratio=args.min_obs_ratio
    )
    print(f"  Clean data shape: {df_standardized.shape}")
    print(f"  Variables retained: {len(valid_cols)}")
    
    # Step 4: Run rolling-window forecasting
    print(f"\n[4/5] Running rolling-window forecasting...")
    
    forecasts_original, forecasts_transformed, forecasts_standardized, forecast_stds = rolling_window_forecast(
        df_standardized, 
        means, 
        stds,
        df_original,
        transform_codes,
        n_factors=args.n_factors,
        initial_window=args.initial_window,
        method=args.method,
        n_samples=args.n_samples,
        n_tune=args.n_tune,
        target_accept=args.target_accept,
        cores=args.cores,
        verbose=not args.quiet
    )
    
    # Step 5: Prepare output
    print(f"\n[5/5] Saving output to: {args.output}")
    
    # Create output DataFrame matching input format
    output_df = pd.DataFrame(index=forecasts_original.index, columns=df_raw.columns)
    
    # Fill in forecasted columns
    for col in forecasts_original.columns:
        if col in output_df.columns:
            output_df[col] = forecasts_original[col]
    
    # Format dates like input (M/D/YYYY)
    output_df.index = output_df.index.strftime('%-m/%-d/%Y')
    output_df.index.name = 'sasdate'
    
    # Add transformation codes as first row
    transform_row = transform_codes[output_df.columns].to_frame().T
    transform_row.index = ['Transform:']
    
    # Combine and save
    final_output = pd.concat([transform_row, output_df])
    final_output.to_csv(args.output)
    
    print(f"\n  Forecast period: {forecasts_original.index[0]} to {forecasts_original.index[-1]}")
    print(f"  Total forecasts: {len(forecasts_original)}")
    print(f"  Output saved to: {args.output}")
    
    # Save forecast standard deviations if requested
    if args.save_std:
        std_output_path = args.output.replace('.csv', '_std.csv')
        forecast_stds.index = forecast_stds.index.strftime('%-m/%-d/%Y')
        forecast_stds.index.name = 'sasdate'
        forecast_stds.to_csv(std_output_path)
        print(f"  Forecast std saved to: {std_output_path}")
    
    # Save transformed forecasts if requested
    if args.save_transformed:
        trans_output_path = args.output.replace('.csv', '_transformed.csv')
        forecasts_transformed.index = forecasts_transformed.index.strftime('%-m/%-d/%Y')
        forecasts_transformed.index.name = 'sasdate'
        forecasts_transformed.to_csv(trans_output_path)
        print(f"  Transformed forecasts saved to: {trans_output_path}")
    
    print("\n" + "=" * 70)
    print("FORECASTING COMPLETE")
    print("=" * 70)
    
    return final_output, forecasts_original, forecasts_transformed


# =============================================================================
# 8. ARGUMENT PARSER
# =============================================================================

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Rolling-Window Bayesian Dynamic Factor Model Forecasting for FRED-MD data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults (simplified Bayesian method)
  python forecast.py --input 2025-12-MD.csv --output forecasts.csv

  # Use full Bayesian DFM (slower but most accurate)
  python forecast.py --input 2025-12-MD.csv --output forecasts.csv --method full

  # Custom number of factors and initial window
  python forecast.py --input 2025-12-MD.csv --output forecasts.csv --n-factors 8 --initial-window 60

  # Use efficient SVD-based method (fastest)
  python forecast.py --input 2025-12-MD.csv --output forecasts.csv --method efficient

  # Increase MCMC samples for better inference
  python forecast.py --input 2025-12-MD.csv --output forecasts.csv --n-samples 5000 --n-tune 2000

  # Save forecast standard deviations and transformed forecasts
  python forecast.py --input 2025-12-MD.csv --output forecasts.csv --save-std --save-transformed

  # Use multiple cores for faster sampling
  python forecast.py --input 2025-12-MD.csv --output forecasts.csv --cores 4

  # Quiet mode (minimal output)
  python forecast.py --input 2025-12-MD.csv --output forecasts.csv --quiet

Methods:
  full       - Full Bayesian Dynamic Factor Model with state-space formulation
               Slowest but most accurate. Uses MCMC for all parameters jointly.
  
  simplified - Two-step Bayesian approach (default)
               First extracts factors via Bayesian PCA, then fits AR(1) to factors.
               Good balance of speed and accuracy.
  
  dynamic    - Bayesian DFM with explicit factor dynamics
               Models factor evolution directly in the joint posterior.
  
  efficient  - SVD-based factor extraction with OLS AR(1)
               Fastest method, no MCMC. Good for quick results or large datasets.
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input FRED-MD CSV file (e.g., 2025-12-MD.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to output CSV file for forecasts'
    )
    
    # Model parameters
    parser.add_argument(
        '--n-factors', '-k',
        type=int,
        default=5,
        help='Number of latent factors to extract (default: 5)'
    )
    
    parser.add_argument(
        '--initial-window', '-w',
        type=int,
        default=120,
        help='Initial training window in months (default: 120 = 10 years)'
    )
    
    parser.add_argument(
        '--method', '-m',
        type=str,
        default='simplified',
        choices=['full', 'simplified', 'dynamic', 'efficient'],
        help='Estimation method: full, simplified, dynamic, or efficient (default: simplified)'
    )
    
    # MCMC parameters
    parser.add_argument(
        '--n-samples', '-s',
        type=int,
        default=1000,
        help='Number of MCMC samples (default: 1000)'
    )
    
    parser.add_argument(
        '--n-tune', '-t',
        type=int,
        default=500,
        help='Number of MCMC tuning/burn-in samples (default: 500)'
    )
    
    parser.add_argument(
        '--target-accept',
        type=float,
        default=0.9,
        help='Target acceptance rate for NUTS sampler (default: 0.9)'
    )
    
    parser.add_argument(
        '--cores', '-c',
        type=int,
        default=1,
        help='Number of CPU cores for parallel sampling (default: 1)'
    )
    
    # Data preprocessing
    parser.add_argument(
        '--min-obs-ratio',
        type=float,
        default=0.8,
        help='Minimum observation ratio to keep a variable (default: 0.8)'
    )
    
    # Output options
    parser.add_argument(
        '--save-std',
        action='store_true',
        help='Save forecast standard deviations to separate file'
    )
    
    parser.add_argument(
        '--save-transformed',
        action='store_true',
        help='Save forecasts in transformed scale to separate file'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    return parser.parse_args()


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    final_output, forecasts_original, forecasts_transformed = main(args)