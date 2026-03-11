# Outputs:
#   forecasts.csv          Mean point estimates (same layout as input file)
#   forecasts_p5.csv       5th percentile of predictive density
#   forecasts_p95.csv      95th percentile of predictive density
#   results.db             SQLite database with posterior draws + run metadata (if --db is set)
#
# Basic rolling forecast (24 months, Bayesian):
#   python3 code.py --input 2026-01-MD.csv --output forecasts.csv --method bayesian --horizon 24
#
# With more simulations:
#   python3 code.py --input 2026-01-MD.csv --output forecasts.csv --method bayesian --horizon 24 --simulations 2000
#
# With SQLite database:
#   python3 code.py --input 2026-01-MD.csv --output forecasts.csv --method bayesian --horizon 24 --db results.db
#
# With all tuning options:
#   python3 code.py --input 2026-01-MD.csv --output forecasts.csv --method bayesian --horizon 24
#       --simulations 2000 --n-factors 5 --n-samples 1000 --n-tune 500 --cores 4 --db results.db
#
# Bayesian-AR variant (fully Bayesian AR(1) on factors, slower):
#   python3 code.py --input 2026-01-MD.csv --output forecasts.csv --method bayesian-ar --horizon 24
#
# Fast non-Bayesian run (SVD factor model, no MCMC):
#   python3 code.py --input 2026-01-MD.csv --output forecasts.csv --method efficient --horizon 12
#
# Fill ragged missing data at end of series:
#   python3 code.py --input 2026-01-MD.csv --output filled.csv --missing
#
# Suppress all progress output:
#   python3 code.py --input 2026-01-MD.csv --output forecasts.csv --method bayesian --horizon 24 --quiet

import argparse
import os
import sqlite3
import tempfile
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT    DEFAULT (datetime('now')),
            input_file  TEXT,
            method      TEXT,
            n_factors   INTEGER,
            n_samples   INTEGER,
            n_tune      INTEGER,
            simulations INTEGER,
            horizon     INTEGER,
            start_date  TEXT
        );
        CREATE TABLE IF NOT EXISTS traces (
            trace_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id       INTEGER REFERENCES runs(run_id),
            horizon_step INTEGER,
            model_step   TEXT,
            trace_blob   BLOB
        );
        CREATE TABLE IF NOT EXISTS forecasts (
            run_id          INTEGER REFERENCES runs(run_id),
            forecast_date   TEXT,
            variable_name   TEXT,
            mean_value      REAL,
            p5_value        REAL,
            p95_value       REAL
        );
        CREATE INDEX IF NOT EXISTS idx_forecasts_run  ON forecasts(run_id);
        CREATE INDEX IF NOT EXISTS idx_forecasts_date ON forecasts(forecast_date);
        CREATE INDEX IF NOT EXISTS idx_traces_run     ON traces(run_id);
    """)
    conn.commit()
    return conn


def register_run(conn, args):
    cur = conn.execute(
        """INSERT INTO runs
           (input_file, method, n_factors, n_samples, n_tune, simulations, horizon, start_date)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            args.input,
            args.method,
            args.n_factors,
            args.n_samples,
            args.n_tune,
            args.simulations,
            getattr(args, 'horizon', 0),
            args.start_date,
        )
    )
    conn.commit()
    return cur.lastrowid


def save_trace_to_db(conn, run_id, horizon_step, model_step, trace):
    import arviz as az
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp = f.name
    try:
        az.to_netcdf(trace, tmp)
        with open(tmp, "rb") as f:
            blob = f.read()
    finally:
        os.unlink(tmp)
    conn.execute(
        "INSERT INTO traces (run_id, horizon_step, model_step, trace_blob) VALUES (?,?,?,?)",
        (run_id, horizon_step, model_step, blob)
    )
    conn.commit()


def load_trace_from_db(conn, run_id, horizon_step, model_step):
    import arviz as az
    row = conn.execute(
        "SELECT trace_blob FROM traces WHERE run_id=? AND horizon_step=? AND model_step=?",
        (run_id, horizon_step, model_step)
    ).fetchone()
    if row is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        f.write(row[0])
        tmp = f.name
    try:
        return az.from_netcdf(tmp)
    finally:
        os.unlink(tmp)


def save_forecasts_to_db(conn, run_id, forecast_dates, mean_df, p5_df, p95_df):
    rows = []
    for date in forecast_dates:
        date_str = str(date.date()) if hasattr(date, 'date') else str(date)
        for var in mean_df.columns:
            m   = float(mean_df.loc[date, var])  if pd.notna(mean_df.loc[date, var])  else None
            p5  = float(p5_df.loc[date, var])    if pd.notna(p5_df.loc[date, var])    else None
            p95 = float(p95_df.loc[date, var])   if pd.notna(p95_df.loc[date, var])   else None
            rows.append((run_id, date_str, var, m, p5, p95))
    conn.executemany(
        "INSERT INTO forecasts (run_id,forecast_date,variable_name,mean_value,p5_value,p95_value) VALUES (?,?,?,?,?,?)",
        rows
    )
    conn.commit()


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    transform_codes = df.iloc[0].copy()
    df = df.iloc[1:].copy()
    df['sasdate'] = pd.to_datetime(df['sasdate'], format='%m/%d/%Y')
    df.set_index('sasdate', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df, transform_codes


def apply_transformations(df, transform_codes):
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


class EfficientFactorModel:
    def __init__(self, n_factors=5):
        self.n_factors = n_factors

    def fit(self, Y):
        T, N = Y.shape
        K = min(self.n_factors, min(T, N) - 1)
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        self.factors = U[:, :K] * S[:K]
        self.loadings = Vt[:K, :].T
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

    def forecast_samples(self, h=1, n_samples=1000):
        N = self.loadings.shape[0]
        K = self.factors.shape[1]
        out = np.zeros((n_samples, h, N))
        for i in range(n_samples):
            F_curr = self.factors[-1].copy()
            for step in range(h):
                F_next = self.ar_intercepts + self.ar_coefs * F_curr + self.ar_sigmas * np.random.randn(K)
                out[i, step] = self.loadings @ F_next + self.residual_std * np.random.randn(N)
                F_curr = F_next
        return out

    def forecast(self, h=1, n_samples=1000):
        s = self.forecast_samples(h, n_samples)
        return s.mean(axis=0), s.std(axis=0)


class BayesianFactorModel:
    def __init__(self, n_factors=5):
        self.n_factors = n_factors

    def fit(self, Y, n_samples=1000, n_tune=500, cores=1):
        import pymc as pm
        T, N = Y.shape
        K = min(self.n_factors, min(T, N) - 1)
        print(f"    Fitting Bayesian Factor Model: T={T}, N={N}, K={K}")
        with pm.Model():
            Lambda = pm.Normal('Lambda', mu=0, sigma=1, shape=(N, K))
            F      = pm.Normal('F',      mu=0, sigma=1, shape=(T, K))
            Psi    = pm.HalfNormal('Psi', sigma=1, shape=N)
            mu     = pm.math.dot(F, Lambda.T)
            pm.Normal('Y_obs', mu=mu, sigma=Psi, observed=Y)
            self.trace = pm.sample(
                draws=n_samples, tune=n_tune, cores=cores,
                return_inferencedata=True, progressbar=True, target_accept=0.9
            )
        self.loadings = self.trace.posterior['Lambda'].mean(dim=['chain','draw']).values
        self.factors  = self.trace.posterior['F'].mean(dim=['chain','draw']).values
        self.Psi      = self.trace.posterior['Psi'].mean(dim=['chain','draw']).values
        self._fit_ar()
        return self

    def _fit_ar(self):
        K = self.factors.shape[1]
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

    def forecast_samples(self, h=1, n_samples=1000):
        N = self.loadings.shape[0]
        K = self.factors.shape[1]
        Lambda_post = self.trace.posterior['Lambda'].values.reshape(-1, N, K)
        Psi_post    = self.trace.posterior['Psi'].values.reshape(-1, N)
        n_post      = Lambda_post.shape[0]
        out = np.zeros((n_samples, h, N))
        for i in range(n_samples):
            idx    = np.random.randint(n_post)
            Lambda = Lambda_post[idx]
            Psi    = Psi_post[idx]
            F_curr = self.factors[-1].copy()
            for step in range(h):
                F_next       = self.ar_intercepts + self.ar_coefs * F_curr + self.ar_sigmas * np.random.randn(K)
                out[i, step] = Lambda @ F_next + Psi * np.random.randn(N)
                F_curr       = F_next
        return out

    def forecast(self, h=1, n_samples=1000):
        s = self.forecast_samples(h, n_samples)
        return s.mean(axis=0), s.std(axis=0)


class BayesianARFactorModel:
    def __init__(self, n_factors=5):
        self.n_factors = n_factors

    def fit(self, Y, n_samples=1000, n_tune=500, cores=1):
        import pymc as pm
        T, N = Y.shape
        K = min(self.n_factors, min(T, N) - 1)
        print(f"    Fitting Bayesian AR Factor Model: T={T}, N={N}, K={K}")
        print("    Step 1: factor extraction...")
        with pm.Model():
            Lambda = pm.Normal('Lambda', mu=0, sigma=1, shape=(N, K))
            F      = pm.Normal('F',      mu=0, sigma=1, shape=(T, K))
            Psi    = pm.HalfNormal('Psi', sigma=1, shape=N)
            mu     = pm.math.dot(F, Lambda.T)
            pm.Normal('Y_obs', mu=mu, sigma=Psi, observed=Y)
            self.trace = pm.sample(
                draws=n_samples, tune=n_tune, cores=cores,
                return_inferencedata=True, progressbar=True, target_accept=0.9
            )
        self.loadings = self.trace.posterior['Lambda'].mean(dim=['chain','draw']).values
        self.factors  = self.trace.posterior['F'].mean(dim=['chain','draw']).values
        self.Psi      = self.trace.posterior['Psi'].mean(dim=['chain','draw']).values
        print("    Step 2: Bayesian AR(1) for each factor...")
        self.ar_traces = []
        self.ar_coefs = np.zeros(K)
        self.ar_intercepts = np.zeros(K)
        self.ar_sigmas = np.zeros(K)
        for k in range(K):
            f = self.factors[:, k]
            with pm.Model():
                c   = pm.Normal('c',   mu=0, sigma=1)
                phi = pm.Uniform('phi', lower=-0.99, upper=0.99)
                sig = pm.HalfNormal('sigma', sigma=1)
                pm.Normal('f_obs', mu=c + phi*f[:-1], sigma=sig, observed=f[1:])
                ar_trace = pm.sample(500, tune=250, cores=1,
                                     return_inferencedata=True, progressbar=False)
            self.ar_traces.append(ar_trace)
            self.ar_coefs[k]      = float(ar_trace.posterior['phi'].mean().values)
            self.ar_intercepts[k] = float(ar_trace.posterior['c'].mean().values)
            self.ar_sigmas[k]     = float(ar_trace.posterior['sigma'].mean().values)
        return self

    def forecast_samples(self, h=1, n_samples=1000):
        N = self.loadings.shape[0]
        K = self.factors.shape[1]
        Lambda_post = self.trace.posterior['Lambda'].values.reshape(-1, N, K)
        Psi_post    = self.trace.posterior['Psi'].values.reshape(-1, N)
        n_post      = Lambda_post.shape[0]
        out = np.zeros((n_samples, h, N))
        for i in range(n_samples):
            idx    = np.random.randint(n_post)
            Lambda = Lambda_post[idx]
            Psi    = Psi_post[idx]
            F_curr = self.factors[-1].copy()
            for step in range(h):
                F_next = np.zeros(K)
                for k in range(K):
                    ar_post = self.ar_traces[k].posterior
                    sz      = ar_post['phi'].values.size
                    ai      = np.random.randint(sz)
                    phi_s   = ar_post['phi'].values.flatten()[ai]
                    c_s     = ar_post['c'].values.flatten()[ai]
                    sig_s   = ar_post['sigma'].values.flatten()[ai]
                    F_next[k] = c_s + phi_s * F_curr[k] + sig_s * np.random.randn()
                out[i, step] = Lambda @ F_next + Psi * np.random.randn(N)
                F_curr = F_next
        return out

    def forecast(self, h=1, n_samples=1000):
        s = self.forecast_samples(h, n_samples)
        return s.mean(axis=0), s.std(axis=0)


def run_rolling_horizon(
    df_raw, transform_codes,
    horizon=24,
    method='bayesian',
    n_factors=5,
    n_samples=1000,
    n_tune=500,
    cores=1,
    simulations=1000,
    min_obs_ratio=0.8,
    db_conn=None,
    run_id=None,
    verbose=True,
):
    if verbose:
        print(f"\n{'='*70}")
        print(f"ROLLING HORIZON FORECAST  ({horizon} months, {simulations} simulations)")
        print(f"Method: {method}  |  Factors: {n_factors}")
        print(f"{'='*70}")

    df_working   = df_raw.copy()
    last_date    = df_working.index[-1]
    all_dates    = []
    mean_rows    = []
    p5_rows      = []
    p95_rows     = []
    step_samples = {}

    for step in range(horizon):
        forecast_date = last_date + relativedelta(months=step + 1)
        if verbose:
            print(f"\n  Step {step+1}/{horizon}  →  {forecast_date.strftime('%Y-%m')}")

        df_transformed = apply_transformations(df_working, transform_codes)
        df_std, means, stds, valid_cols = prepare_data_for_modeling(
            df_transformed, min_obs_ratio=min_obs_ratio
        )
        Y_train = df_std.values

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

            if db_conn is not None and run_id is not None and hasattr(model, 'trace'):
                save_trace_to_db(db_conn, run_id, step, 'factor_model', model.trace)
                if hasattr(model, 'ar_traces'):
                    for k, ar_tr in enumerate(model.ar_traces):
                        save_trace_to_db(db_conn, run_id, step, f'ar_factor_{k}', ar_tr)

            raw_samples    = model.forecast_samples(h=1, n_samples=simulations)
            draws_std      = raw_samples[:, 0, :]
            draws_trans    = draws_std * stds[valid_cols].values + means[valid_cols].values
            draws_original = np.full((simulations, len(df_raw.columns)), np.nan)
            all_cols       = list(df_raw.columns)

            for vi, col in enumerate(valid_cols):
                col_idx = all_cols.index(col) if col in all_cols else None
                if col_idx is None:
                    continue
                try:
                    code = int(float(transform_codes[col]))
                except (ValueError, KeyError):
                    code = 1
                orig_vals = df_working[col].dropna().values
                x_t1 = orig_vals[-1] if len(orig_vals) >= 1 else np.nan
                x_t2 = orig_vals[-2] if len(orig_vals) >= 2 else x_t1
                for sim_i in range(simulations):
                    draws_original[sim_i, col_idx] = reverse_transformation_single(
                        draws_trans[sim_i, vi], x_t1, x_t2, code
                    )

            mean_row = np.nanmean(draws_original, axis=0)
            p5_row   = np.nanpercentile(draws_original, 5,  axis=0)
            p95_row  = np.nanpercentile(draws_original, 95, axis=0)

            if verbose:
                n_valid = np.sum(~np.isnan(mean_row))
                print(f"    Forecast complete  ({n_valid}/{len(all_cols)} variables)")

        except Exception as e:
            if verbose:
                print(f"    Error: {e}  — using fallback")
            mean_row = p5_row = p95_row = np.full(len(df_raw.columns), np.nan)

        all_dates.append(forecast_date)
        mean_rows.append(mean_row)
        p5_rows.append(p5_row)
        p95_rows.append(p95_row)
        step_samples[str(forecast_date.date())] = draws_original

        new_row    = pd.Series(mean_row, index=df_raw.columns, name=forecast_date)
        df_working = pd.concat([df_working, new_row.to_frame().T])

    cols    = list(df_raw.columns)
    mean_df = pd.DataFrame(mean_rows, index=all_dates, columns=cols)
    p5_df   = pd.DataFrame(p5_rows,  index=all_dates, columns=cols)
    p95_df  = pd.DataFrame(p95_rows, index=all_dates, columns=cols)

    if db_conn is not None and run_id is not None:
        if verbose:
            print("\n  Saving forecasts to database...")
        save_forecasts_to_db(db_conn, run_id, all_dates, mean_df, p5_df, p95_df)

    return mean_df, p5_df, p95_df, step_samples


def format_date_index(df):
    try:
        df.index = df.index.strftime('%-m/%-d/%Y')
    except ValueError:
        df.index = df.index.strftime('%#m/%#d/%Y')
    df.index.name = 'sasdate'
    return df


def build_output_csv(df_raw, transform_codes, forecast_df):
    out = pd.DataFrame(index=forecast_df.index, columns=df_raw.columns)
    for col in forecast_df.columns:
        if col in out.columns:
            out[col] = forecast_df[col]
    out = format_date_index(out)
    transform_row = transform_codes[out.columns].to_frame().T
    transform_row.index = ['Transform:']
    return pd.concat([transform_row, out])


def identify_missing_data(df):
    missing_info = {}
    for col in df.columns:
        series    = df[col]
        valid_idx = series.last_valid_index()
        if valid_idx is not None:
            last_valid_pos = df.index.get_loc(valid_idx)
            if last_valid_pos < len(df) - 1:
                missing_info[col] = {
                    'last_valid_idx':    valid_idx,
                    'last_valid_pos':    last_valid_pos,
                    'missing_start_pos': last_valid_pos + 1,
                    'n_missing':         len(df) - 1 - last_valid_pos
                }
    return missing_info


def fill_missing_data_rolling(df_original, transform_codes, n_factors=5, method='efficient',
                               n_samples=1000, n_tune=500, cores=1, min_obs_ratio=0.8, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print("FILLING MISSING DATA (Rolling Basis)")
        print("="*60)
    missing_info = identify_missing_data(df_original)
    if not missing_info:
        if verbose:
            print("  No missing data found.")
        return df_original.copy()
    max_missing       = max(info['n_missing'] for info in missing_info.values())
    all_dates         = df_original.index
    last_complete_pos = len(df_original) - max_missing - 1
    df_filled         = df_original.copy()
    for step in range(max_missing):
        current_pos      = last_complete_pos + step + 1
        forecast_date    = all_dates[current_pos]
        cols_to_forecast = [col for col, info in missing_info.items()
                            if info['missing_start_pos'] <= current_pos]
        if not cols_to_forecast:
            continue
        df_train       = df_filled.iloc[:current_pos].copy()
        df_transformed = apply_transformations(df_train, transform_codes)
        df_std, means, stds, valid_cols = prepare_data_for_modeling(df_transformed, min_obs_ratio)
        if len(df_std) < 10:
            continue
        try:
            if method == 'efficient':
                model = EfficientFactorModel(n_factors=n_factors)
                model.fit(df_std.values)
            elif method == 'bayesian':
                model = BayesianFactorModel(n_factors=n_factors)
                model.fit(df_std.values, n_samples=n_samples, n_tune=n_tune, cores=cores)
            elif method == 'bayesian-ar':
                model = BayesianARFactorModel(n_factors=n_factors)
                model.fit(df_std.values, n_samples=n_samples, n_tune=n_tune, cores=cores)
            fc_mean, _ = model.forecast(h=1, n_samples=300)
            fc_trans   = pd.DataFrame([fc_mean], index=[forecast_date], columns=valid_cols)
            fc_orig    = fc_trans * stds[valid_cols].values + means[valid_cols].values
            for col in cols_to_forecast:
                if col not in valid_cols:
                    continue
                try:
                    code = int(float(transform_codes[col]))
                except (ValueError, KeyError):
                    code = 1
                fc_val    = fc_orig.loc[forecast_date, col]
                orig_vals = df_filled[col].iloc[:current_pos].dropna()
                x_t1      = orig_vals.iloc[-1] if len(orig_vals) >= 1 else np.nan
                x_t2      = orig_vals.iloc[-2] if len(orig_vals) >= 2 else x_t1
                df_filled.loc[forecast_date, col] = reverse_transformation_single(fc_val, x_t1, x_t2, code)
        except Exception as e:
            if verbose:
                print(f"    Error at step {step}: {e}")
    return df_filled


def main(args):
    print("=" * 70)
    print("BAYESIAN DYNAMIC FACTOR MODEL FORECASTING")
    print("=" * 70)

    print(f"\n[1] Loading: {args.input}")
    df_raw, transform_codes = load_and_preprocess_data(args.input)
    last_date = df_raw.index[-1]
    print(f"  Shape: {df_raw.shape}  |  Range: {df_raw.index[0].date()} → {last_date.date()}")

    db_conn = run_id = None
    if args.db:
        print(f"\n[DB] Connecting to: {args.db}")
        db_conn = init_db(args.db)
        run_id  = register_run(db_conn, args)
        print(f"  Run ID: {run_id}")

    if args.missing:
        print("\n[2] Filling missing data...")
        df_filled = fill_missing_data_rolling(
            df_raw, transform_codes,
            n_factors=args.n_factors, method=args.method,
            n_samples=args.n_samples, n_tune=args.n_tune,
            cores=args.cores, min_obs_ratio=args.min_obs_ratio,
            verbose=not args.quiet
        )
        final = build_output_csv(df_raw, transform_codes, df_filled)
        final.to_csv(args.output)
        print(f"\nSaved → {args.output}")
        if db_conn:
            db_conn.close()
        return

    if args.horizon > 0:
        print(f"\n[2] Rolling horizon forecast: {args.horizon} months from {last_date.date()}")
        mean_df, p5_df, p95_df, step_samples = run_rolling_horizon(
            df_raw, transform_codes,
            horizon=args.horizon,
            method=args.method,
            n_factors=args.n_factors,
            n_samples=args.n_samples,
            n_tune=args.n_tune,
            cores=args.cores,
            simulations=args.simulations,
            min_obs_ratio=args.min_obs_ratio,
            db_conn=db_conn,
            run_id=run_id,
            verbose=not args.quiet,
        )

        build_output_csv(df_raw, transform_codes, mean_df).to_csv(args.output)
        print(f"\nMean forecasts    → {args.output}")

        p5_path  = args.output.replace('.csv', '_p5.csv')
        p95_path = args.output.replace('.csv', '_p95.csv')
        build_output_csv(df_raw, transform_codes, p5_df).to_csv(p5_path)
        build_output_csv(df_raw, transform_codes, p95_df).to_csv(p95_path)
        print(f"5th  percentile   → {p5_path}")
        print(f"95th percentile   → {p95_path}")

        print("\n" + "="*70)
        print("COMPLETE")
        print("="*70)
        if db_conn:
            db_conn.close()
        return mean_df, p5_df, p95_df

    print("\nNo --horizon specified. Use --horizon N to run rolling forecasts.")
    if db_conn:
        db_conn.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Bayesian Dynamic Factor Model — Rolling Horizon Forecaster',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input',            '-i', required=True,  help='Input CSV (FRED-MD format)')
    parser.add_argument('--output',           '-o', required=True,  help='Output CSV (mean forecasts)')
    parser.add_argument('--horizon',                type=int,   default=0,           help='Months to forecast ahead (default: 0)')
    parser.add_argument('--simulations',            type=int,   default=1000,        help='Predictive simulations (default: 1000)')
    parser.add_argument('--db',                     type=str,   default=None,        help='SQLite database path')
    parser.add_argument('--missing',                action='store_true',             help='Fill ragged missing data')
    parser.add_argument('--n-factors',        '-k', type=int,   default=5,           help='Latent factors (default: 5)')
    parser.add_argument('--initial-window',   '-w', type=int,   default=120,         help='Initial training window (default: 120)')
    parser.add_argument('--start-date',             type=str,   default=None,        help='Start date YYYY-MM-DD')
    parser.add_argument('--method',           '-m', type=str,   default='efficient',
                        choices=['efficient','bayesian','bayesian-ar'],              help='Model method')
    parser.add_argument('--n-samples',        '-s', type=int,   default=1000,        help='MCMC draws (default: 1000)')
    parser.add_argument('--n-tune',           '-t', type=int,   default=500,         help='MCMC tuning steps (default: 500)')
    parser.add_argument('--cores',            '-c', type=int,   default=1,           help='CPU cores (default: 1)')
    parser.add_argument('--min-obs-ratio',          type=float, default=0.8,         help='Min obs ratio (default: 0.8)')
    parser.add_argument('--save-std',               action='store_true',             help='Save std-dev CSV')
    parser.add_argument('--save-transformed',       action='store_true',             help='Save transformed-space CSV')
    parser.add_argument('--quiet',            '-q', action='store_true',             help='Suppress progress output')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)