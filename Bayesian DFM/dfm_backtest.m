function dfm_backtest(varargin)
% DFM_BACKTEST  Pseudo-out-of-sample rolling backtest for the Bayesian DFM
%
% Companion to dfm_forecast.m.  For each horizon h in {1,3,12,24,60} months,
% steps through backtest origins from start_date (default 2000-01) to the
% last usable date, fits a Dynamic Factor Model on all data available at
% each origin, generates n_draws forecast paths h steps ahead, then records
% both summary statistics and n_spaghetti sampled paths for spaghetti-plot
% visualisation.
%
% Memory strategy — draws are processed ONE VARIABLE AT A TIME so peak RAM
% is proportional to  n_draws × h  rather than  n_draws × h × N_variables.
% The raw draw matrix from forecast_samples (n_draws × h × N_valid) is the
% only large allocation per origin; it is cleared before the next origin.
%
% USAGE
%   dfm_backtest('2026-02-MD.csv', 'backtest_out/')
%   dfm_backtest('2026-02-MD.csv', 'out/', 'method', 'efficient', ...
%                'n_draws', 10000, 'n_spaghetti', 200)
%
% REQUIRED (positional)
%   arg 1   FRED-MD format input CSV
%   arg 2   Output directory  (created automatically if absent)
%
% PARAMETERS (name-value)
%   'horizons'       Months-ahead horizons         [1 3 12 24 60]
%   'start_date'     First backtest origin         ['2000-01-01']
%   'n_draws'        Forecast draws per origin     [10000]
%   'n_spaghetti'    Draw paths saved per origin   [200]
%   'method'         'efficient' | 'bayesian'      ['efficient']
%   'n_factors'      Latent factors                [5]
%   'n_samples'      MCMC draws  (bayesian only)   [500]
%   'n_tune'         MCMC burn-in (bayesian only)  [250]
%   'min_obs_ratio'  Min valid-obs fraction / col  [0.8]
%   'quiet'          Suppress progress output      [false]
%
% OUTPUT FILES  (one pair per horizon H, written to output_dir)
%
%   backtest_h{HH}_stats.csv
%     origin_date, forecast_date, variable,
%     mean, p5, p50, p95, realized
%     — one row per (origin × forecast_step × variable)
%
%   backtest_h{HH}_draws.csv
%     origin_date, forecast_date, variable,
%     d001, d002, …, d{n_spaghetti}
%     — one row per (origin × forecast_step × variable)
%     — each d### column is one draw path value in ORIGINAL LEVELS
%
% ORIGIN-STEP SCHEDULE
%   h =  1  →  origins every  1 month
%   h =  3  →  origins every  3 months
%   h = 12  →  origins every 12 months
%   h = 24  →  origins every 12 months
%   h = 60  →  origins every 12 months
%
% MULTI-STEP REVERSE TRANSFORM
%   For differencing codes (2,3,5,6,7) the level at forecast step s is
%   computed from the FORECASTED level at step s-1, so each draw has its
%   own recursive trajectory.  Level codes (1,4) are step-independent.
%
% NOTE ON SPEED
%   For large n_draws (≥ 5000) and many origins (h = 1, ~300 origins),
%   the 'efficient' method is strongly recommended.  The Bayesian method
%   with n_draws = 10000 is best suited to h ≥ 12 with fewer origins.
%
% FRED-MD Transformation Codes:
%   1: x_t            (level)
%   2: Δx_t           (first difference)
%   3: Δ²x_t          (second difference)
%   4: log(x_t)       (log level)
%   5: Δlog(x_t)      (log first difference)
%   6: Δ²log(x_t)     (log second difference)
%   7: Δ(x_t/x_{t-1} - 1)  (first difference of percent change)

% =========================================================================
%  PARSE INPUTS
% =========================================================================
p = inputParser;
addRequired(p,  'input');
addRequired(p,  'output_dir');
addParameter(p, 'horizons',      [1 3 12 24 60]);
addParameter(p, 'start_date',    '2000-01-01');
addParameter(p, 'n_draws',       10000);
addParameter(p, 'n_spaghetti',   200);
addParameter(p, 'method',        'efficient');
addParameter(p, 'n_factors',     5);
addParameter(p, 'n_samples',     500);
addParameter(p, 'n_tune',        250);
addParameter(p, 'min_obs_ratio', 0.8);
addParameter(p, 'quiet',         false);

if nargin >= 2 && (ischar(varargin{1}) || isstring(varargin{1})) ...
        && ~contains(varargin{1}, '=') && ~startsWith(varargin{1}, '-')
    parse(p, varargin{1}, varargin{2}, varargin{3:end});
else
    parse(p, varargin{:});
end
args    = p.Results;
verbose = ~args.quiet;

% =========================================================================
%  [1] LOAD DATA
% =========================================================================
if verbose
    fprintf('%s\n', repmat('=',1,70));
    fprintf('DFM BACKTEST\n');
    fprintf('%s\n', repmat('=',1,70));
    fprintf('\n[1] Loading: %s\n', args.input);
end

[Y_raw, var_names, transform_codes, dates] = load_and_preprocess(args.input);

if verbose
    fprintf('    Shape: %d x %d  |  Range: %s -> %s\n', ...
        size(Y_raw,1), length(var_names), fmt_date(dates(1)), fmt_date(dates(end)));
end

% =========================================================================
%  [2] CREATE OUTPUT DIRECTORY
% =========================================================================
if ~isfolder(args.output_dir)
    mkdir(args.output_dir);
    if verbose
        fprintf('\n[2] Created output directory: %s\n', args.output_dir);
    end
end

% =========================================================================
%  [3] RUN BACKTEST FOR EACH HORIZON
% =========================================================================
start_dt = datetime(args.start_date, 'InputFormat', 'yyyy-MM-dd');

for h = args.horizons(:)'
    if verbose
        fprintf('\n%s\n', repmat('=',1,70));
        fprintf('HORIZON  h = %d month(s)\n', h);
        fprintf('%s\n', repmat('=',1,70));
    end
    run_horizon_backtest(Y_raw, var_names, transform_codes, dates, ...
                         h, start_dt, args, verbose);
end

if verbose
    fprintf('\n%s\n', repmat('=',1,70));
    fprintf('BACKTEST COMPLETE\n');
    fprintf('%s\n', repmat('=',1,70));
end
end


% =========================================================================
%  HORIZON BACKTEST RUNNER
%  One call per horizon h.  Opens output files, loops over origins,
%  streams results variable-by-variable, then closes files.
% =========================================================================
function run_horizon_backtest(Y_raw, var_names, transform_codes, dates, ...
                              h, start_dt, args, verbose)

N_all       = size(Y_raw, 2);
last_date   = dates(end);
n_draws     = args.n_draws;
n_spaghetti = min(args.n_spaghetti, n_draws);

% --- Origin-step schedule ------------------------------------------------
if     h == 1,  ostep = 1;
elseif h <= 3,  ostep = 3;
else,           ostep = 12;     % annual for h = 12, 24, 60
end

% --- Build origin list ---------------------------------------------------
% Minimum training window: 3 years of monthly data
min_train    = 36;
first_avail  = dates(min_train + 1);
origin_start = max(start_dt, first_avail);
last_allowed = last_date - calmonths(h);

if origin_start > last_allowed
    fprintf('    h=%d: date range too short after burn-in. Skipping.\n', h);
    return
end

origin_dates = datetime.empty(0,1);
target = origin_start;
while target <= last_allowed
    [~, idx] = min(abs(dates - target));
    d = dates(idx);
    if (isempty(origin_dates) || d ~= origin_dates(end)) && d <= last_allowed
        origin_dates(end+1,1) = d; %#ok<AGROW>
    end
    target = target + calmonths(ostep);
end
n_origins = length(origin_dates);

if n_origins == 0
    fprintf('    h=%d: no valid origins found. Skipping.\n', h);
    return
end

if verbose
    fprintf('\n  Origins     : %d  (step = every %d month(s))\n', n_origins, ostep);
    fprintf('  Range       : %s -> %s\n', fmt_date(origin_dates(1)), fmt_date(origin_dates(end)));
    fprintf('  Draws       : %d  |  Spaghetti paths saved: %d\n', n_draws, n_spaghetti);
    fprintf('  Method      : %s  |  Factors: %d\n\n', args.method, args.n_factors);
end

% --- Open output CSV files -----------------------------------------------
out_base   = fullfile(args.output_dir, sprintf('backtest_h%02d', h));
stats_path = [out_base '_stats.csv'];
draws_path = [out_base '_draws.csv'];

fid_s = fopen(stats_path, 'w');
fid_d = fopen(draws_path, 'w');
if fid_s < 0, error('Cannot open for writing: %s', stats_path); end
if fid_d < 0, error('Cannot open for writing: %s', draws_path); end

% Stats header
fprintf(fid_s, 'origin_date,forecast_date,variable,mean,p5,p50,p95,realized\n');

% Draws header: fixed columns + one column per draw path
draw_hdr = 'origin_date,forecast_date,variable';
for di = 1:n_spaghetti
    draw_hdr = [draw_hdr, sprintf(',d%03d', di)]; %#ok<AGROW>
end
fprintf(fid_d, '%s\n', draw_hdr);

% Pre-build fprintf format string for draw rows (computed once)
draw_fmt = ['%s,%s,%s', repmat(',%.6g', 1, n_spaghetti), '\n'];

% Evenly-spaced indices into the n_draws draws for spaghetti sampling
draw_sel = round(linspace(1, n_draws, n_spaghetti));

% --- Main origin loop ----------------------------------------------------
n_ok = 0;

for oi = 1:n_origins
    origin   = origin_dates(oi);
    t_origin = find(dates == origin, 1);
    if isempty(t_origin) || t_origin <= min_train, continue; end

    Y_train = Y_raw(1:t_origin, :);

    if verbose
        pct = floor(100 * oi / n_origins);
        fprintf('  [%3d%%]  origin %d/%d  -> %s\n', pct, oi, n_origins, fmt_date(origin));
    end

    % --- Fit model and draw paths in standardised transformed space -------
    try
        [out_std, mn, sd, valid_idx] = fit_model_and_draw( ...
            Y_train, transform_codes, h, ...
            args.method, args.n_factors, n_draws, ...
            args.n_samples, args.n_tune, args.min_obs_ratio);
        % out_std : n_draws x h x N_valid
    catch ME
        if verbose
            fprintf('    SKIP (fit failed): %s\n', ME.message);
        end
        continue
    end

    % --- Realized values from the held-out data --------------------------
    realized = NaN(h, N_all);
    for s = 1:h
        t_fut = find(dates == origin + calmonths(s), 1);
        if ~isempty(t_fut) && t_fut <= size(Y_raw, 1)
            realized(s, :) = Y_raw(t_fut, :);
        end
    end

    % --- Stream: one variable at a time ----------------------------------
    for vi = 1:length(valid_idx)
        col_idx = valid_idx(vi);
        vname   = var_names{col_idx};

        % Unstandardise this variable's draws: n_draws x h
        draws_trans = squeeze(out_std(:, :, vi)) * sd(vi) + mn(vi);
        if h == 1
            draws_trans = reshape(draws_trans, n_draws, 1);
        end

        % Reverse-transform recursively to original levels: n_draws x h
        paths_orig = reverse_transform_one_var( ...
            draws_trans, Y_train(:, col_idx), transform_codes(col_idx), h);

        % Summary statistics across draws (1 x h each)
        p5_v  = prctile_base(paths_orig,  5, 1);
        p50_v = prctile_base(paths_orig, 50, 1);
        p95_v = prctile_base(paths_orig, 95, 1);
        mn_v  = mean(paths_orig, 1, 'omitnan');

        % Spaghetti: n_spaghetti x h  (evenly sampled from draws)
        spag = paths_orig(draw_sel, :);

        % Write one row per forecast step
        for s = 1:h
            fc_date = origin + calmonths(s);
            ori_str = fmt_date(origin);
            fc_str  = fmt_date(fc_date);

            % Stats row
            fprintf(fid_s, '%s,%s,%s,%.6g,%.6g,%.6g,%.6g,%.6g\n', ...
                ori_str, fc_str, vname, ...
                mn_v(s), p5_v(s), p50_v(s), p95_v(s), realized(s, col_idx));

            % Draw row: n_spaghetti values for this step
            fprintf(fid_d, draw_fmt, ori_str, fc_str, vname, spag(:, s)');
        end
    end

    % Free the large draw array before the next origin
    clear out_std draws_trans paths_orig spag

    n_ok = n_ok + 1;
end

fclose(fid_s);
fclose(fid_d);

if verbose
    fprintf('\n  Stats  -> %s\n', stats_path);
    fprintf('  Draws  -> %s\n',  draws_path);
    fprintf('  Origins completed: %d / %d\n', n_ok, n_origins);
end
end


% =========================================================================
%  FIT MODEL AND DRAW
%  Returns draws in standardised, FRED-MD-transformed space.
%  out_std : n_draws x h x N_valid
% =========================================================================
function [out_std, mn, sd, valid_idx] = fit_model_and_draw( ...
    Y_train, transform_codes, h, method, n_factors, ...
    n_draws, n_samples, n_tune, min_obs_ratio)

% Transform to FRED-MD space
[Y_trans, ~] = apply_transformations(Y_train, transform_codes);

% Standardise and filter low-coverage columns
[Y_std, mn, sd, valid_idx] = prepare_data_for_modeling(Y_trans, min_obs_ratio);

if isempty(Y_std) || size(Y_std, 1) < 10
    error('Fewer than 10 usable observations after preprocessing.');
end

% Fit factor model
switch lower(method)
    case 'efficient'
        mdl = fit_efficient(Y_std, n_factors);
    case {'bayesian', 'bayesian-ar'}
        mdl = fit_bayesian(Y_std, n_factors, n_samples, n_tune, false);
    otherwise
        error('Unknown method: %s', method);
end

% Draw h-step forecast paths: n_draws x h x N_valid
out_std = forecast_samples(mdl, h, n_draws);
end


% =========================================================================
%  SINGLE-VARIABLE MULTI-STEP REVERSE TRANSFORM
%
%  Converts a (n_draws x h) matrix of draw paths from unstandardised
%  FRED-MD transformed space to original level space.  For differencing
%  codes (2, 3, 5, 6, 7) the level at step s uses the FORECASTED level
%  at step s-1 as context, so each draw gets its own recursive trajectory.
%
%  draws_trans : n_draws x h  (unstandardised, FRED-MD transformed)
%  col_history : T x 1        (original levels for this variable)
%  code        : integer 1-7
%  h           : forecast horizon
%
%  Returns  paths_orig : n_draws x h  (original levels)
% =========================================================================
function paths_orig = reverse_transform_one_var(draws_trans, col_history, code, h)

n_draws = size(draws_trans, 1);

valid_hist = col_history(~isnan(col_history));
if isempty(valid_hist)
    paths_orig = NaN(n_draws, h);
    return
end

% Seed the recursion from the last two observed levels
lev_t0  = valid_hist(end);
lev_tm1 = valid_hist(max(end-1, 1));

lev_prev  = repmat(lev_t0,  n_draws, 1);   % n_draws x 1
lev_prev2 = repmat(lev_tm1, n_draws, 1);   % n_draws x 1

% Historical stats for safety clipping
hmu    = mean(valid_hist);
hsig   = std(valid_hist);
do_clip = hsig > 0;

paths_orig = zeros(n_draws, h);

for s = 1:h
    v = draws_trans(:, s);          % n_draws x 1

    switch code
        case 1   % level: no transformation
            lev_s = v;

        case 2   % Δx_t  →  x_t = x_{t-1} + Δx_t
            lev_s = lev_prev + v;

        case 3   % Δ²x_t  →  x_t = 2·x_{t-1} - x_{t-2} + Δ²x_t
            lev_s = 2*lev_prev - lev_prev2 + v;

        case 4   % log(x_t)  →  x_t = exp(v)
            lev_s = exp(v);

        case 5   % Δlog(x_t)  →  x_t = x_{t-1} · exp(Δlog)
            lev_s = lev_prev .* exp(v);
            lev_s = max(lev_s, 1e-10);

        case 6   % Δ²log(x_t)
            % Δlog(x_{t-1}) = log(x_{t-1}) - log(x_{t-2})
            % Δlog(x_t)     = Δlog(x_{t-1}) + Δ²log(x_t)
            % x_t           = x_{t-1} · exp(Δlog(x_t))
            dlog_prev = log(max(lev_prev, 1e-10)) - log(max(lev_prev2, 1e-10));
            lev_s     = lev_prev .* exp(dlog_prev + v);
            lev_s     = max(lev_s, 1e-10);

        case 7   % Δ(x_t/x_{t-1} - 1)
            % pct_t = pct_{t-1} + Δpct_t
            % x_t   = x_{t-1} · (1 + pct_t)
            pct_prev = lev_prev ./ max(lev_prev2, 1e-10) - 1;
            lev_s    = lev_prev .* (1 + pct_prev + v);

        otherwise
            lev_s = v;
    end

    % Safety clip: ±20 historical standard deviations
    if do_clip
        lev_s = min(max(lev_s, hmu - 20*hsig), hmu + 20*hsig);
    end

    paths_orig(:, s) = lev_s;

    % Shift context for next step
    lev_prev2 = lev_prev;
    lev_prev  = lev_s;
end
end


% =========================================================================
%  TOOLBOX-FREE PERCENTILE
%  Replaces Statistics Toolbox prctile(X, pct, dim).
%  Works on arrays of any dimensionality.
% =========================================================================
function p = prctile_base(X, pct, dim)
if nargin < 3, dim = 1; end
n   = size(X, dim);
Xs  = sort(X, dim, 'ascend');
% Hazen formula: same as Statistics Toolbox prctile
frac = pct / 100 * n - 0.5;
lo   = max(floor(frac), 0);
hi   = min(ceil(frac),  n - 1);
w    = frac - lo;
idx_lo           = repmat({':'}, 1, ndims(X));
idx_hi           = repmat({':'}, 1, ndims(X));
idx_lo{dim}      = lo + 1;
idx_hi{dim}      = hi + 1;
p = (1 - w) .* Xs(idx_lo{:}) + w .* Xs(idx_hi{:});
end


% =========================================================================
%  BELOW: ALL SHARED HELPERS
%  Identical to dfm_forecast.m (post-fixes).  Keep the two files in sync.
% =========================================================================

% ---- Data loading -------------------------------------------------------
function [Y, var_names, transform_codes, dates] = load_and_preprocess(filepath)
raw = readtable(filepath, 'ReadVariableNames', true);

tc_row = raw(1, :);
raw    = raw(2:end, :);

date_col = raw{:, 1};
if iscell(date_col)
    dates = datetime(date_col, 'InputFormat', 'M/d/yyyy');
else
    dates = date_col;
end

all_names = raw.Properties.VariableNames;
var_names = all_names(2:end);
N         = length(var_names);

transform_codes = zeros(1, N);
for j = 1:N
    v = tc_row{1, j+1};
    if iscell(v),         v = v{1}; end
    if ischar(v),         transform_codes(j) = str2double(v);
    elseif isnumeric(v),  transform_codes(j) = v;
    end
    if isnan(transform_codes(j)), transform_codes(j) = 1; end
end

Y = NaN(height(raw), N);
for j = 1:N
    col = raw{:, j+1};
    if iscell(col)
        col = cellfun(@(x) str2double_safe(x), col);
    end
    Y(:, j) = double(col);
end
end

% ---- FRED-MD forward transformations (codes 1-7) ------------------------
function [Y_out, dates_out] = apply_transformations(Y, codes)
[T, N] = size(Y);
Y_out  = NaN(T, N);

for j = 1:N
    x = Y(:, j);
    c = codes(j);
    switch c
        case 1
            Y_out(:, j) = x;
        case 2
            Y_out(2:T, j) = x(2:T) - x(1:T-1);
        case 3
            dx = x(2:T) - x(1:T-1);
            Y_out(3:T, j) = dx(2:end) - dx(1:end-1);
        case 4
            x(x <= 0) = NaN;
            Y_out(:, j) = log(x);
        case 5
            x(x <= 0) = NaN;
            lx = log(x);
            Y_out(2:T, j) = lx(2:T) - lx(1:T-1);
        case 6
            x(x <= 0) = NaN;
            lx  = log(x);
            dlx = lx(2:T) - lx(1:T-1);
            Y_out(3:T, j) = dlx(2:end) - dlx(1:end-1);
        case 7
            pct = x(2:T) ./ x(1:T-1) - 1;
            Y_out(3:T, j) = pct(2:end) - pct(1:end-1);
        otherwise
            Y_out(:, j) = x;
    end
end

Y_out     = Y_out(3:end, :);   % drop first 2 rows lost to differencing
dates_out = [];
end

% ---- Standardise and filter low-coverage columns ------------------------
function [Y_std, means, stds, valid_idx] = prepare_data_for_modeling(Y, min_obs_ratio)
first_valid_row = find(any(~isnan(Y), 2), 1, 'first');
if isempty(first_valid_row)
    Y_std = []; means = []; stds = []; valid_idx = [];
    return
end
Y = Y(first_valid_row:end, :);
T = size(Y, 1);

miss_ratio = sum(isnan(Y), 1) / T;
valid_idx  = find(miss_ratio < (1 - min_obs_ratio));
Y_clean    = Y(:, valid_idx);

Y_clean = interp_time_limit(Y_clean, 3);
Y_clean = ffill(Y_clean);
Y_clean = bfill(Y_clean);

all_valid = all(~isnan(Y_clean), 1);
Y_clean   = Y_clean(:, all_valid);
valid_idx = valid_idx(all_valid);

if isempty(Y_clean)
    Y_std = []; means = []; stds = []; valid_idx = [];
    return
end

means = mean(Y_clean, 1, 'omitnan');
stds  = std(Y_clean,  0, 1, 'omitnan');
stds(stds == 0) = 1;
Y_std = (Y_clean - means) ./ stds;
end

% ---- Efficient (SVD) factor model ---------------------------------------
function mdl = fit_efficient(Y, n_factors)
[T, N] = size(Y);
K = min(n_factors, min(T, N) - 1);

[U, S, V]    = svd(Y, 'econ');
mdl.factors  = U(:, 1:K) * S(1:K, 1:K);
mdl.loadings = V(:, 1:K);
mdl.K        = K;
mdl.N        = N;
mdl.type     = 'efficient';

mdl.ar_coefs      = zeros(1, K);
mdl.ar_intercepts = zeros(1, K);
mdl.ar_sigmas     = zeros(1, K);

for k = 1:K
    f = mdl.factors(:, k);
    X = [ones(T-1, 1), f(1:end-1)];
    y = f(2:end);
    b = X \ y;
    mdl.ar_intercepts(k) = b(1);
    mdl.ar_coefs(k)      = max(min(b(2), 0.99), -0.99);
    resid = y - X * b;
    mdl.ar_sigmas(k)     = max(std(resid), 0.01);
end

Y_hat            = mdl.factors * mdl.loadings';
mdl.residual_std = max(std(Y - Y_hat, 0, 1), 0.01);
end

% ---- Bayesian (Gibbs) factor model --------------------------------------
function mdl = fit_bayesian(Y, n_factors, n_samples, n_tune, verbose)
[T, N] = size(Y);
K      = min(n_factors, min(T, N) - 1);
n_total = n_samples + n_tune;

if verbose
    fprintf('    Bayesian DFM: T=%d, N=%d, K=%d  |  MCMC=%d (%d burn-in)\n', ...
        T, N, K, n_total, n_tune);
end

[U, S, V] = svd(Y, 'econ');
F      = U(:, 1:K) * S(1:K, 1:K);
Lambda = V(:, 1:K);
Psi    = 0.5 * ones(N, 1);
Phi    = zeros(K, 1);
Sigma2 = ones(K, 1);

Lambda_store = zeros(n_samples, N, K);
Psi_store    = zeros(n_samples, N);
Phi_store    = zeros(n_samples, K);
Sigma2_store = zeros(n_samples, K);
F_store      = zeros(n_samples, T, K);

for i = 1:n_total
    F             = sample_factors(Y, Lambda, Psi, Phi, Sigma2);
    Lambda        = sample_loadings(Y, F, Psi);
    Psi           = sample_psi(Y, F, Lambda);
    [Phi, Sigma2] = sample_ar(F);

    if i > n_tune
        s = i - n_tune;
        Lambda_store(s, :, :) = Lambda;
        Psi_store(s, :)       = Psi';
        Phi_store(s, :)       = Phi';
        Sigma2_store(s, :)    = Sigma2';
        F_store(s, :, :)      = F;
    end

    if verbose && mod(i, 500) == 0
        phase = 'BURN-IN'; if i > n_tune, phase = 'SAMPLING'; end
        fprintf('      [%3d%%] iter %d/%d [%s]\n', ...
            round(100*i/n_total), i, n_total, phase);
    end
end

mdl.Lambda_store  = Lambda_store;
mdl.Psi_store     = Psi_store;
mdl.Phi_store     = Phi_store;
mdl.Sigma2_store  = Sigma2_store;
mdl.F_store       = F_store;
mdl.loadings      = squeeze(mean(Lambda_store, 1));
mdl.factors       = squeeze(mean(F_store,      1));
mdl.Psi           = mean(Psi_store,    1)';
mdl.ar_coefs      = mean(Phi_store,    1)';
mdl.ar_sigmas     = sqrt(mean(Sigma2_store, 1))';
mdl.ar_intercepts = zeros(K, 1);
mdl.K             = K;
mdl.N             = N;
mdl.type          = 'bayesian';
end

% ---- Forecast draws (standardised transformed space) --------------------
function out = forecast_samples(mdl, h, n_sims)
K   = mdl.K;
N   = mdl.N;
out = zeros(n_sims, h, N);

if strcmp(mdl.type, 'efficient')
    ar_c    = mdl.ar_intercepts(:);
    ar_phi  = mdl.ar_coefs(:);
    ar_sig  = mdl.ar_sigmas(:);
    res_std = mdl.residual_std(:)';
    F_last  = mdl.factors(end, :)';
    F_prev  = F_last;
    if size(mdl.factors, 1) >= 2
        F_prev = mdl.factors(end-1, :)';
    end

    for i = 1:n_sims
        F_curr = F_last;
        F_lag  = F_prev;
        trend  = zeros(K, 1);
        vol    = ar_sig .^ 2;

        for s = 1:h
            trend        = 0.95 * trend + 0.05 * F_curr;
            momentum     = 0.25 * (F_curr - F_lag);
            mean_rev     = -0.10 * (F_curr - trend);
            cycle_phase  = 2 * pi * s / 48;
            cycle        = 0.15 * sin(cycle_phase + rand(K,1)*0.5) .* sign(F_curr);

            if s > 1
                shock_prev = (F_curr - F_lag) .^ 2;
                vol = 0.05*ar_sig.^2 + 0.10*shock_prev + 0.85*vol;
            end

            regime_shock = zeros(K, 1);
            if rand() < 0.08
                regime_shock = randn(K, 1) .* ar_sig * 0.6;
            end

            decay        = exp(-0.015 * s);
            F_stat       = ar_phi .* (F_curr - trend);
            innovation   = sqrt(vol) .* randn(K, 1);

            F_next = trend + decay*F_stat + momentum + mean_rev + ...
                     cycle + regime_shock + innovation;
            F_next = max(min(F_next, 6), -6);

            out(i, s, :) = (mdl.loadings * F_next)' + res_std .* randn(1, N);

            F_lag  = F_curr;
            F_curr = F_next;
        end
    end

else  % Bayesian
    n_post = size(mdl.Lambda_store, 1);

    for i = 1:n_sims
        post_idx = randi(n_post);

        Lambda = squeeze(mdl.Lambda_store(post_idx, :, :));  % N x K
        Psi    = mdl.Psi_store(post_idx, :)';                % N x 1
        Phi    = mdl.Phi_store(post_idx, :)';                % K x 1
        Sigma2 = mdl.Sigma2_store(post_idx, :)';             % K x 1

        if isfield(mdl, 'F_store')
            F_hist = squeeze(mdl.F_store(post_idx, :, :));   % T x K
        else
            F_hist = mdl.factors;
        end

        T_hist   = size(F_hist, 1);
        lookback = min(12, T_hist);
        F_recent = F_hist(end-lookback+1:end, :);
        F_curr   = F_hist(end, :)';
        F_lag    = F_curr;
        if T_hist >= 2, F_lag = F_hist(end-1, :)'; end

        trend    = zeros(K, 1);
        for k = 1:K
            t_idx = (1:lookback)';
            y_k   = F_recent(:, k);
            if sum(~isnan(y_k)) > 3
                pp      = polyfit(t_idx, y_k, 1);
                trend(k) = pp(1) * lookback;
            end
        end

        vol_base = Sigma2;
        if lookback > 2
            rc       = diff(F_recent);
            rv       = var(rc, 0, 1)';
            vol_base = 0.5*Sigma2 + 0.5*rv;
        end

        F_path = zeros(h, K);
        vol_t  = vol_base;

        for s = 1:h
            trend_mom = 0.3 * trend;
            trend = 0.92*trend + 0.08*(F_curr - trend) + 0.05*trend_mom;

            cyc_bus  = 0.20 * sin(2*pi*s/48  + post_idx*0.10);
            cyc_sea  = 0.10 * sin(2*pi*s/12  + post_idx*0.20);
            cyc_long = 0.08 * sin(2*pi*s/96  + post_idx*0.15);
            cyc_tot  = (cyc_bus + cyc_sea + cyc_long) .* ones(K, 1);
            cyc_tot  = cyc_tot .* (1 + 0.2*tanh(F_curr));

            momentum = 0.30 * (F_curr - F_lag);
            deviation  = F_curr - trend;
            mean_rev = -0.12 * deviation .* (1 + 0.5*abs(deviation));

            if s > 1
                shock_t = (F_curr - F_path(s-1,:)') .^ 2;
                vol_t   = 0.03*vol_base + 0.15*shock_t + 0.82*vol_t;
            end
            if s > 3
                rv_recent = mean(var(F_path(max(1,s-3):s-1,:), 0, 1));
                if rv_recent > 2*mean(vol_base), vol_t = vol_t * 1.5; end
            end
            vol_t = max(min(vol_t, 10*vol_base), 0.1*vol_base);

            jump_prob    = 0.06 + 0.04*min(mean(abs(deviation)), 2);
            regime_shock = zeros(K, 1);
            if rand() < jump_prob
                js = sqrt(vol_t) .* randn(K,1) * 0.8;
                if rand() < 0.6, js = -sign(deviation) .* abs(js); end
                regime_shock = js;
            end

            if rand() < 0.05, vol_t = vol_t * (1 + abs(randn())*0.5); end

            decay_h     = exp(-0.018 * s);
            Phi_eff     = max(min(Phi * decay_h, 0.98), -0.98);
            F_stat      = Phi_eff .* (F_curr - trend);
            innovation  = sqrt(vol_t) .* randn(K, 1);
            if rand() < 0.3, innovation = innovation - 0.2*sqrt(vol_t); end

            F_next = trend + F_stat + momentum + mean_rev + ...
                     cyc_tot + regime_shock + innovation;
            for k = 1:K
                if abs(F_next(k)) > 5
                    F_next(k) = 5 * tanh(F_next(k) / 5);
                end
            end

            F_path(s, :) = F_next';
            F_lag  = F_curr;
            F_curr = F_next;
        end

        Y_fc = (Lambda * F_path')';   % h x N
        for n_idx = 1:N
            psi_n = Psi(n_idx);
            for s = 1:h
                vs = 1 + 0.3*(s/h);
                if rand() < 0.05
                    idio = sqrt(psi_n) * randn() * 2;
                else
                    idio = sqrt(psi_n * vs) * randn();
                end
                Y_fc(s, n_idx) = Y_fc(s, n_idx) + idio;
            end
        end

        out(i, :, :) = Y_fc;
    end
end
end

% ---- Gibbs steps --------------------------------------------------------
function F = sample_factors(Y, Lambda, Psi, Phi, Sigma2)
[T, ~]     = size(Y);
K          = length(Phi);
Phi_mat    = diag(Phi);
Sigma2_mat = diag(Sigma2);
Psi_inv    = 1 ./ max(Psi, 1e-8);
eye_K      = eye(K);

LPsiInvL = Lambda' * (Lambda .* Psi_inv);
LPsiInvY = Lambda' * bsxfun(@times, Y', Psi_inv);

m_filt = zeros(T, K);
P_filt = zeros(K, K, T);
P_pred = diag(Sigma2 ./ max(1 - Phi.^2, 1e-6));
m_pred = zeros(K, 1);

for t = 1:T
    P_pred_reg = P_pred + eye_K * 1e-8;
    P_pred_inv = P_pred_reg \ eye_K;
    Omega_post = P_pred_inv + LPsiInvL;
    P_post     = Omega_post \ eye_K;
    P_post     = 0.5 * (P_post + P_post');
    m_post     = P_post * (P_pred_inv * m_pred + LPsiInvY(:, t));
    m_filt(t, :)     = m_post';
    P_filt(:, :, t)  = P_post;
    if t < T
        m_pred = Phi .* m_post;
        P_pred = Phi_mat * P_post * Phi_mat' + Sigma2_mat;
    end
end

F       = zeros(T, K);
F(T, :) = mvn_sample(m_filt(T,:)', P_filt(:,:,T))';

for t = T-1:-1:1
    P_t          = P_filt(:,:,t);
    P_pred_tp1   = Phi_mat * P_t * Phi_mat' + Sigma2_mat + eye_K*1e-8;
    G            = P_t * Phi_mat' / P_pred_tp1;
    m_smooth     = m_filt(t,:)' + G * (F(t+1,:)' - Phi .* m_filt(t,:)');
    P_smooth     = P_t - G * Phi_mat * P_t;
    P_smooth     = 0.5 * (P_smooth + P_smooth');
    F(t, :)      = mvn_sample(m_smooth, P_smooth)';
end
end

function Lambda = sample_loadings(Y, F, Psi)
[~, K] = size(F);
N      = length(Psi);
FtF    = F' * F;
FtY    = F' * Y;
eye_K  = eye(K);
Lambda = zeros(N, K);
for i = 1:N
    psi_i        = max(Psi(i), 1e-8);
    V            = (FtF / psi_i + eye_K) \ eye_K;
    mu           = V * FtY(:, i) / psi_i;
    Lambda(i, :) = mvn_sample(mu, V)';
end
end

function Psi = sample_psi(Y, F, Lambda)
[T, N] = size(Y);
resid  = Y - F * Lambda';
a0     = 3;  b0 = 2;
Psi    = zeros(N, 1);
for i = 1:N
    a_post = a0 + T/2;
    b_post = b0 + 0.5 * sum(resid(:,i).^2);
    Psi(i) = b_post / gamma_rnd(a_post, 1);
end
Psi = min(max(Psi, 1e-6), 1e4);
end

function [Phi, Sigma2] = sample_ar(F)
[T, K]  = size(F);
Phi     = zeros(K, 1);
Sigma2  = ones(K, 1);
a0 = 3;  b0 = 1;
for k = 1:K
    f_lag    = F(1:end-1, k);
    f_cur    = F(2:end,   k);
    sigma2   = Sigma2(k);
    f_lag_sq = f_lag' * f_lag;
    V_phi    = 1 / (f_lag_sq / sigma2 + 1);
    mu_phi   = V_phi * (f_lag' * f_cur) / sigma2;
    Phi(k)   = sample_truncated_normal(mu_phi, sqrt(V_phi), -0.99, 0.99);
    resid_ar  = f_cur - Phi(k) * f_lag;
    a_post    = a0 + (T-1)/2;
    b_post    = b0 + 0.5 * (resid_ar' * resid_ar);
    Sigma2(k) = b_post / gamma_rnd(a_post, 1);
end
Sigma2 = min(max(Sigma2, 1e-6), 1e4);
end

% ---- Probability utilities ----------------------------------------------
function x = mvn_sample(mu, P)
n = length(mu);
P = P + eye(n) * 1e-8;
try
    L = chol(P, 'lower');
    x = mu + L * randn(size(mu));
catch
    [V, D] = eig(P);
    D = max(real(diag(D)), 1e-8);
    x = mu + V * (sqrt(D) .* randn(n, 1));
end
end

function x = sample_truncated_normal(mu, sigma, lo, hi)
a = (lo - mu) / sigma;
b = (hi - mu) / sigma;
Phi_a = 0.5 * (1 + erf(a / sqrt(2)));
Phi_b = 0.5 * (1 + erf(b / sqrt(2)));
if Phi_b - Phi_a < 1e-10
    x = (lo + hi) / 2;
    return
end
u = Phi_a + rand() * (Phi_b - Phi_a);
u = max(min(u, 1 - 1e-10), 1e-10);
x = mu + sigma * sqrt(2) * erfinv(2*u - 1);
x = max(lo + 1e-6, min(hi - 1e-6, x));
end

function x = gamma_rnd(a, b)
if a < 1
    x = gamma_rnd(a + 1, b) * rand()^(1/a);
    return
end
d = a - 1/3;
c = 1 / sqrt(9 * d);
while true
    z = randn();
    v = (1 + c*z)^3;
    if v <= 0, continue; end
    u = rand();
    if u < 1 - 0.0331 * z^4
        x = d * v * b; return
    end
    if log(u) < 0.5*z^2 + d*(1 - v + log(v))
        x = d * v * b; return
    end
end
end

% ---- Gap-filling utilities ----------------------------------------------
function Y = interp_time_limit(Y, max_gap)
[~, N] = size(Y);
for j = 1:N
    col = Y(:, j);
    idx = find(~isnan(col));
    if length(idx) < 2, continue; end
    for ii = 1:length(idx)-1
        gap = idx(ii+1) - idx(ii) - 1;
        if gap > 0 && gap <= max_gap
            t1 = idx(ii);  t2 = idx(ii+1);
            v1 = col(t1);  v2 = col(t2);
            for t = t1+1:t2-1
                col(t) = v1 + (v2-v1) * (t-t1)/(t2-t1);
            end
        end
    end
    Y(:, j) = col;
end
end

function Y = ffill(Y)
[T, N] = size(Y);
for j = 1:N
    for t = 2:T
        if isnan(Y(t,j)) && ~isnan(Y(t-1,j))
            Y(t,j) = Y(t-1,j);
        end
    end
end
end

function Y = bfill(Y)
[T, N] = size(Y);
for j = 1:N
    for t = T-1:-1:1
        if isnan(Y(t,j)) && ~isnan(Y(t+1,j))
            Y(t,j) = Y(t+1,j);
        end
    end
end
end

% ---- Formatting ---------------------------------------------------------
function s = fmt_date(d)
s = sprintf('%d/%d/%d', month(d), day(d), year(d));
end

function v = str2double_safe(x)
if ischar(x) || isstring(x),  v = str2double(x);
elseif isnumeric(x),           v = double(x);
else,                          v = NaN;
end
end
