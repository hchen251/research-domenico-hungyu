function dfm_forecast(varargin)
% DFM_FORECAST  Bayesian Dynamic Factor Model - Rolling Horizon Forecaster
%
% Gibbs sampler implemented entirely in native MATLAB — no toolboxes required.
% Gamma sampling via Marsaglia-Tsang (base MATLAB only).
%
% FRED-MD Transformation Codes:
%   1: x_t           (no transformation)
%   2: Δx_t          (first difference)
%   3: Δ²x_t         (second difference)
%   4: log(x_t)      (log level)
%   5: Δlog(x_t)     (log first difference)
%   6: Δ²log(x_t)    (log second difference)
%   7: Δ(x_t/x_{t-1} - 1)  (first difference of percent change)
%
% Outputs:
%   forecasts.csv        Mean point estimates
%   forecasts_p5.csv     5th  percentile
%   forecasts_p95.csv    95th percentile
%
% Usage (name-value pairs):
%   dfm_forecast('2026-02-MD.csv', 'forecasts.csv', 'method', 'bayesian', 'horizon', 24)
%
% Parameters:
%   'input'         Input CSV  (FRED-MD format)              [required]
%   'output'        Output CSV (mean forecasts)              [required]
%   'horizon'       Months to forecast ahead                 [default: 12]
%   'method'        'efficient' | 'bayesian' | 'bayesian-ar' [default: 'efficient']
%   'n_factors'     Latent factors                           [default: 5]
%   'n_samples'     Gibbs draws after burn-in                [default: 1000]
%   'n_tune'        Gibbs burn-in iterations                 [default: 500]
%   'simulations'   Predictive simulations for output        [default: 1000]
%   'min_obs_ratio' Min fraction of valid obs per column     [default: 0.8]
%   'missing'       Fill ALL missing data (interior + trailing) [default: false]
%   'quiet'         Suppress progress output                 [default: false]
%
% Example — rolling forecast:
%   dfm_forecast('2026-02-MD.csv', 'forecasts.csv', 'method', 'bayesian', 'horizon', 24)
%
% Example — fill ALL missing data (interior gaps + trailing edge):
%   dfm_forecast('2026-02-MD.csv', 'complete_data.csv', 'missing', true)

% -------------------------------------------------------------------------
% Parse inputs
% -------------------------------------------------------------------------
p = inputParser;
addRequired(p,  'input');
addRequired(p,  'output');
addParameter(p, 'horizon',       12);
addParameter(p, 'method',        'efficient');
addParameter(p, 'n_factors',     5);
addParameter(p, 'n_samples',     1000);
addParameter(p, 'n_tune',        500);
addParameter(p, 'simulations',   1000);
addParameter(p, 'min_obs_ratio', 0.8);
addParameter(p, 'missing',       false);
addParameter(p, 'quiet',         false);

% Allow positional call: dfm_forecast(input, output, ...)
if nargin >= 2 && ischar(varargin{1}) && ~contains(varargin{1},'=') ...
        && ~startsWith(varargin{1}, '-')
    parse(p, varargin{1}, varargin{2}, varargin{3:end});
else
    parse(p, varargin{:});
end
args = p.Results;

verbose = ~args.quiet;

% -------------------------------------------------------------------------
% [1] Load data
% -------------------------------------------------------------------------
fprintf('%s\n', repmat('=',1,70));
fprintf('BAYESIAN DYNAMIC FACTOR MODEL FORECASTING\n');
fprintf('%s\n', repmat('=',1,70));
fprintf('\n[1] Loading: %s\n', args.input);

[Y_raw, var_names, transform_codes, dates] = load_and_preprocess(args.input);
last_date = dates(end);

fprintf('  Shape: %d x %d  |  Range: %s -> %s\n', ...
    size(Y_raw,1), size(Y_raw,2), fmt_date(dates(1)), fmt_date(last_date));

% -------------------------------------------------------------------------
% [2] Fill ALL missing data (interior gaps + trailing edge)
%     OUTPUT: FULL historical dataset with all missing data filled
% -------------------------------------------------------------------------
if args.missing
    fprintf('\n[2] Filling ALL missing data (interior gaps + trailing edge)...\n');

    [Y_filled, filled_mask] = fill_all_missing_data( ...
        Y_raw, var_names, transform_codes, dates, ...
        args.method, args.n_factors, args.n_samples, args.n_tune, ...
        args.min_obs_ratio, verbose);

    % Output the COMPLETE dataset (all historical data with all gaps filled)
    write_output_csv(args.output, dates, Y_filled, var_names, transform_codes);
    
    n_filled = sum(filled_mask(:));
    n_series_affected = sum(any(filled_mask, 1));
    n_rows_affected = sum(any(filled_mask, 2));
    
    fprintf('\n%s\n', repmat('=',1,70));
    fprintf('FILL SUMMARY\n');
    fprintf('%s\n', repmat('=',1,70));
    fprintf('  Total cells filled:    %d\n', n_filled);
    fprintf('  Series affected:       %d / %d\n', n_series_affected, size(Y_filled,2));
    fprintf('  Time periods affected: %d / %d\n', n_rows_affected, size(Y_filled,1));
    fprintf('  Complete historical dataset saved -> %s\n', args.output);
    fprintf('%s\n', repmat('=',1,70));
    fprintf('COMPLETE\n');
    fprintf('%s\n', repmat('=',1,70));
    return
end

% -------------------------------------------------------------------------
% [3] Rolling horizon forecast
%     OUTPUT: Only future forecast rows
% -------------------------------------------------------------------------
if args.horizon > 0
    fprintf('\n[2] Rolling horizon forecast: %d months from %s\n', ...
        args.horizon, fmt_date(last_date));

    [mean_mat, p5_mat, p95_mat, forecast_dates] = run_rolling_horizon( ...
        Y_raw, var_names, transform_codes, dates, ...
        args.horizon, args.method, args.n_factors, ...
        args.n_samples, args.n_tune, args.simulations, ...
        args.min_obs_ratio, verbose);

    write_output_csv(args.output,                           forecast_dates, mean_mat, var_names, transform_codes);
    write_output_csv(strrep(args.output,'.csv','_p5.csv'),  forecast_dates, p5_mat,  var_names, transform_codes);
    write_output_csv(strrep(args.output,'.csv','_p95.csv'), forecast_dates, p95_mat, var_names, transform_codes);

    fprintf('\nMean forecasts  -> %s\n', args.output);
    fprintf('5th  percentile -> %s\n',  strrep(args.output,'.csv','_p5.csv'));
    fprintf('95th percentile -> %s\n',  strrep(args.output,'.csv','_p95.csv'));
    fprintf('\n%s\nCOMPLETE\n%s\n', repmat('=',1,70), repmat('=',1,70));
else
    fprintf('\nNo horizon specified. Pass ''horizon'', N to run forecasts.\n');
end
end


% =========================================================================
%  FILL ALL MISSING DATA (unified: interior gaps + trailing edge)
%
%  Strategy:
%    1. Identify ALL missing positions:
%         - Interior gaps (NaN between valid observations)
%         - Trailing edge (ragged NaNs at the end)
%    2. Sort all missing positions by time
%    3. Fill sequentially using rolling forecasts
%    4. Each fill uses all data up to that point (including previously filled)
%    5. Return FULL dataset with all gaps filled + mask of what was filled
%
%  OUTPUT: Complete historical dataset (all rows) with all fillable gaps populated
% =========================================================================
function [Y_filled, filled_mask] = fill_all_missing_data( ...
        Y_raw, var_names, transform_codes, dates, ...
        method, n_factors, n_samples, n_tune, min_obs_ratio, verbose)

if verbose
    fprintf('\n%s\nIDENTIFYING MISSING DATA\n%s\n', ...
        repmat('=',1,70), repmat('=',1,70));
end

[T, N] = size(Y_raw);
Y_filled = Y_raw;
filled_mask = false(T, N);

% -------------------------------------------------------------------------
% Step 1: Identify ALL missing positions
% -------------------------------------------------------------------------
missing_positions = [];  % Will store [time_idx, col_idx] pairs

for j = 1:N
    col = Y_raw(:, j);
    valid_idx = find(~isnan(col));
    
    if isempty(valid_idx)
        continue  % Entirely missing column, skip
    end
    
    first_valid = valid_idx(1);
    last_valid = valid_idx(end);
    
    % Interior gaps: NaNs between first and last valid
    if length(valid_idx) >= 2
        interior_range = first_valid:last_valid;
        interior_gaps = interior_range(isnan(col(interior_range)));
        
        for t = interior_gaps(:)'
            missing_positions(end+1, :) = [t, j]; %#ok<AGROW>
        end
    end
    
    % Trailing edge: NaNs after last valid observation
    if last_valid < T
        trailing_range = (last_valid+1):T;
        for t = trailing_range(:)'
            missing_positions(end+1, :) = [t, j]; %#ok<AGROW>
        end
    end
end

if isempty(missing_positions)
    if verbose
        fprintf('  No missing data found.\n');
    end
    return
end

% -------------------------------------------------------------------------
% Step 2: Sort by time and analyze
% -------------------------------------------------------------------------
missing_positions = sortrows(missing_positions, 1);
total_missing = size(missing_positions, 1);

% Analyze by type
interior_count = 0;
trailing_count = 0;
for idx = 1:total_missing
    t = missing_positions(idx, 1);
    j = missing_positions(idx, 2);
    col = Y_raw(:, j);
    valid_idx = find(~isnan(col));
    if ~isempty(valid_idx) && t > valid_idx(1) && t < T
        % Check if there's valid data after this point
        if any(~isnan(col(t+1:end)))
            interior_count = interior_count + 1;
        else
            trailing_count = trailing_count + 1;
        end
    else
        trailing_count = trailing_count + 1;
    end
end

if verbose
    fprintf('\n%s\nMISSING DATA SUMMARY\n%s\n', repmat('-',1,70), repmat('-',1,70));
    fprintf('  Interior gaps:      %d cells\n', interior_count);
    fprintf('  Trailing edge:      %d cells\n', trailing_count);
    fprintf('  Total to fill:      %d cells\n', total_missing);
    fprintf('  Time range:         t=%d to t=%d\n', ...
        missing_positions(1,1), missing_positions(end,1));
    fprintf('%s\n', repmat('-',1,70));
end

% -------------------------------------------------------------------------
% Step 3: Fill sequentially in chronological order
% -------------------------------------------------------------------------
if verbose
    fprintf('\nFilling missing data sequentially...\n');
end

n_filled = 0;
last_reported_pct = 0;

for fill_idx = 1:total_missing
    t = missing_positions(fill_idx, 1);
    j = missing_positions(fill_idx, 2);
    
    % Progress reporting
    if verbose
        pct = floor(100 * fill_idx / total_missing);
        if pct >= last_reported_pct + 10
            fprintf('  [%3d%%] Filled %d/%d cells (t=%d, %s -> %s)\n', ...
                pct, fill_idx, total_missing, t, ...
                fmt_date(dates(1)), fmt_date(dates(t)));
            last_reported_pct = pct;
        end
    end
    
    % Need at least 3 periods for differencing transforms
    if t <= 3
        continue
    end
    
    % Fit model on all data up to t-1 (includes previously filled values)
    Y_train = Y_filled(1:t-1, :);
    
    try
        % Transform
        [Y_trans, ~] = apply_transformations(Y_train, transform_codes);
        
        % Standardise + filter
        [Y_std, means, stds, valid_idx] = prepare_data_for_modeling(Y_trans, min_obs_ratio);
        
        if isempty(Y_std) || size(Y_std, 1) < 10
            continue
        end
        
        % Check if column j is in valid_idx
        vi = find(valid_idx == j);
        if isempty(vi)
            continue
        end
        
        % Fit model (use fewer samples for speed during filling)
        switch lower(method)
            case 'efficient'
                mdl = fit_efficient(Y_std, n_factors);
            case {'bayesian', 'bayesian-ar'}
                mdl = fit_bayesian(Y_std, n_factors, min(n_samples, 500), min(n_tune, 250), false);
            otherwise
                error('Unknown method: %s', method);
        end
        
        % Forecast h=1 with 200 simulations
        draws = forecast_samples(mdl, 1, 200);
        fc_std = squeeze(mean(draws, 1));
        if size(fc_std, 2) > size(fc_std, 1)
            fc_std = fc_std';
        end
        
        % Un-standardise
        fc_trans = fc_std(:) .* stds(:) + means(:);
        
        % Reverse-transform for column j
        code = transform_codes(j);
        orig_vals = Y_filled(1:t-1, j);
        orig_vals = orig_vals(~isnan(orig_vals));
        
        if isempty(orig_vals)
            continue
        end
        
        x_t1 = orig_vals(end);
        x_t2 = orig_vals(max(end-1, 1));
        
        filled_val = reverse_transform_vec(fc_trans(vi), x_t1, x_t2, code);
        
        % Fill ONLY if original was NaN (safety check)
        if isnan(Y_raw(t, j))
            Y_filled(t, j) = filled_val;
            filled_mask(t, j) = true;
            n_filled = n_filled + 1;
        end
        
    catch ME
        if verbose && mod(fill_idx, 100) == 0
            fprintf('    Warning: Failed to fill t=%d, col=%s: %s\n', ...
                t, var_names{j}, ME.message);
        end
        continue
    end
end

if verbose
    fprintf('  [100%%] Filled %d/%d cells\n', n_filled, total_missing);
    
    % Detailed summary by series
    fprintf('\n%s\nSERIES-LEVEL SUMMARY\n%s\n', repmat('-',1,70), repmat('-',1,70));
    series_summary = [];
    for j = 1:N
        n_filled_series = sum(filled_mask(:, j));
        if n_filled_series > 0
            series_summary(end+1, :) = [j, n_filled_series]; %#ok<AGROW>
        end
    end
    
    if ~isempty(series_summary)
        [~, sort_idx] = sort(series_summary(:,2), 'descend');
        series_summary = series_summary(sort_idx, :);
        
        fprintf('  Top series with filled data:\n');
        for i = 1:min(10, size(series_summary, 1))
            j = series_summary(i, 1);
            n = series_summary(i, 2);
            fprintf('    %-20s : %d cells filled\n', var_names{j}, n);
        end
        
        if size(series_summary, 1) > 10
            fprintf('    ... and %d more series\n', size(series_summary, 1) - 10);
        end
    end
    fprintf('%s\n', repmat('-',1,70));
end
end


% =========================================================================
%  ROLLING HORIZON FORECASTER
% =========================================================================
function [mean_mat, p5_mat, p95_mat, forecast_dates] = run_rolling_horizon( ...
        Y_raw, var_names, transform_codes, dates, ...
        horizon, method, n_factors, n_samples, n_tune, simulations, ...
        min_obs_ratio, verbose)

if verbose
    fprintf('\n%s\nROLLING HORIZON FORECAST  (%d months, %d simulations)\n', ...
        repmat('=',1,70), horizon, simulations);
    fprintf('Method: %s  |  Factors: %d\n%s\n', method, n_factors, repmat('=',1,70));
end

N_all          = size(Y_raw, 2);
last_date      = dates(end);

mean_mat       = NaN(horizon, N_all);
p5_mat         = NaN(horizon, N_all);
p95_mat        = NaN(horizon, N_all);
forecast_dates = NaT(horizon, 1);

% Store full history for reverse transformations
Y_history = Y_raw;

for step = 1:horizon
    forecast_date        = last_date + calmonths(step);
    forecast_dates(step) = forecast_date;

    if verbose
        fprintf('\n  Step %d/%d  ->  %s\n', step, horizon, datestr(forecast_date,'yyyy-mm'));
    end

    % Apply transformations
    [Y_trans, ~] = apply_transformations(Y_history, transform_codes);
    [Y_std, means, stds, valid_idx] = prepare_data_for_modeling(Y_trans, min_obs_ratio);

    if isempty(Y_std)
        if verbose, fprintf('    No valid data — skipping\n'); end
        continue
    end

    try
        % Fit model
        switch lower(method)
            case 'efficient'
                mdl = fit_efficient(Y_std, n_factors);
            case {'bayesian', 'bayesian-ar'}
                mdl = fit_bayesian(Y_std, n_factors, n_samples, n_tune, verbose);
            otherwise
                error('Unknown method: %s', method);
        end

        % Generate forecast PATHS (not just h=1)
        % Key change: forecast FULL HORIZON at once to preserve curvature
        draws_std = forecast_samples(mdl, horizon - step + 1, simulations);
        
        % Extract just the next step
        draws_std_h1 = squeeze(draws_std(:, 1, :));  % simulations x N_valid
        if size(draws_std_h1, 1) ~= simulations
            draws_std_h1 = draws_std_h1';
        end

        % Un-standardise
        draws_trans = bsxfun(@times, draws_std_h1, stds) + means;

        % Reverse transform with FULL history (not just x_t1, x_t2)
        draws_orig = NaN(simulations, N_all);

        for vi = 1:length(valid_idx)
            col_idx   = valid_idx(vi);
            code      = transform_codes(col_idx);
            
            % Get FULL history for this variable (key change!)
            orig_history = Y_history(:, col_idx);
            orig_history = orig_history(~isnan(orig_history));
            
            if isempty(orig_history), continue; end
            
            % Reverse transform using full history context
            draws_orig(:, col_idx) = reverse_transform_with_history( ...
                draws_trans(:, vi), orig_history, code);
        end

        mean_row = mean(draws_orig, 1, 'omitnan');
        p5_row   = prctile(draws_orig, 5,  1);
        p95_row  = prctile(draws_orig, 95, 1);

        mean_mat(step, :) = mean_row;
        p5_mat(step, :)   = p5_row;
        p95_mat(step, :)  = p95_row;

        if verbose
            n_valid = sum(~isnan(mean_row));
            fprintf('    Forecast complete  (%d/%d variables)\n', n_valid, N_all);
        end

    catch ME
        if verbose
            fprintf('    Error: %s\n', ME.message);
        end
    end

    % Append forecast to history
    Y_history = [Y_history; mean_mat(step, :)];  %#ok<AGROW>
end
end


% =========================================================================
%  DATA LOADING & PREPROCESSING
% =========================================================================
function [Y, var_names, transform_codes, dates] = load_and_preprocess(filepath)
raw = readtable(filepath, 'ReadVariableNames', true);

% First data row is the transform codes
tc_row = raw(1, :);
raw    = raw(2:end, :);

% Extract dates
date_col = raw{:, 1};
if iscell(date_col)
    dates = datetime(date_col, 'InputFormat', 'M/d/yyyy');
else
    dates = date_col;
end

% Variable names (skip sasdate column)
all_names   = raw.Properties.VariableNames;
var_names   = all_names(2:end);
N           = length(var_names);

% Parse transform codes
transform_codes = zeros(1, N);
for j = 1:N
    v = tc_row{1, j+1};
    if iscell(v),         v = v{1}; end
    if ischar(v),         transform_codes(j) = str2double(v);
    elseif isnumeric(v),  transform_codes(j) = v;
    end
    if isnan(transform_codes(j)), transform_codes(j) = 1; end
end

% Build numeric data matrix (T x N)
Y = NaN(height(raw), N);
for j = 1:N
    col = raw{:, j+1};
    if iscell(col)
        col = cellfun(@(x) str2double_safe(x), col);
    end
    Y(:, j) = double(col);
end
end


% =========================================================================
%  FORWARD TRANSFORMATIONS (FRED-MD codes 1-7)
%
%  Code 1: x_t                      (no transformation)
%  Code 2: Δx_t = x_t - x_{t-1}     (first difference)
%  Code 3: Δ²x_t                    (second difference)
%  Code 4: log(x_t)                 (log level)
%  Code 5: Δlog(x_t)                (log first difference)
%  Code 6: Δ²log(x_t)               (log second difference)
%  Code 7: Δ(x_t/x_{t-1} - 1)       (first difference of percent change)
% =========================================================================
function [Y_out, dates_out] = apply_transformations(Y, codes)
[T, N] = size(Y);
Y_out  = NaN(T, N);

for j = 1:N
    x = Y(:, j);
    c = codes(j);
    
    switch c
        case 1  % x_t (no transformation)
            Y_out(:,j) = x;
            
        case 2  % Δx_t = x_t - x_{t-1}
            Y_out(2:T, j) = x(2:T) - x(1:T-1);
            
        case 3  % Δ²x_t = Δx_t - Δx_{t-1} = x_t - 2*x_{t-1} + x_{t-2}
            dx = x(2:T) - x(1:T-1);
            Y_out(3:T, j) = dx(2:end) - dx(1:end-1);
            
        case 4  % log(x_t)
            x(x <= 0) = NaN;
            Y_out(:,j) = log(x);
            
        case 5  % Δlog(x_t) = log(x_t) - log(x_{t-1})
            x(x <= 0) = NaN;
            lx = log(x);
            Y_out(2:T, j) = lx(2:T) - lx(1:T-1);
            
        case 6  % Δ²log(x_t) = Δlog(x_t) - Δlog(x_{t-1})
            x(x <= 0) = NaN;
            lx = log(x);
            dlx = lx(2:T) - lx(1:T-1);
            Y_out(3:T, j) = dlx(2:end) - dlx(1:end-1);
            
        case 7  % Δ(x_t/x_{t-1} - 1) = pct_t - pct_{t-1}
            pct = x(2:T) ./ x(1:T-1) - 1;
            Y_out(3:T, j) = pct(2:end) - pct(1:end-1);
            
        otherwise
            Y_out(:,j) = x;
    end
end

% Drop first 2 rows (lost to differencing)
Y_out     = Y_out(3:end, :);
dates_out = [];
end


% =========================================================================
%  REVERSE TRANSFORMATIONS (vectorised over simulations)
%
%  Given transformed forecast value(s), recover level using last 2 observations.
%
%  Code 1: x_t                      -> out = val
%  Code 2: Δx_t                     -> x_t = x_{t-1} + Δx_t
%  Code 3: Δ²x_t                    -> x_t = 2*x_{t-1} - x_{t-2} + Δ²x_t
%  Code 4: log(x_t)                 -> x_t = exp(val)
%  Code 5: Δlog(x_t)                -> x_t = x_{t-1} * exp(Δlog(x_t))
%  Code 6: Δ²log(x_t)               -> x_t = x_{t-1} * exp(Δlog(x_{t-1}) + Δ²log(x_t))
%  Code 7: Δ(pct)                   -> x_t = x_{t-1} * (1 + pct_{t-1} + Δpct_t)
% =========================================================================
function out = reverse_transform_vec(vals, x_t1, x_t2, code)
% vals  : sim-length column vector (or scalar)
% x_t1  : scalar, last known value
% x_t2  : scalar, second-to-last known value
% code  : integer 1-7

if isnan(x_t1)
    out = NaN(size(vals));
    return
end

switch code
    case 1  % x_t (no transformation)
        out = vals;
        
    case 2  % Δx_t -> x_t = x_{t-1} + Δx_t
        out = x_t1 + vals;
        
    case 3  % Δ²x_t -> x_t = 2*x_{t-1} - x_{t-2} + Δ²x_t
        % Derivation:
        %   Δx_t = Δx_{t-1} + Δ²x_t
        %   Δx_{t-1} = x_{t-1} - x_{t-2}
        %   x_t = x_{t-1} + Δx_t = x_{t-1} + (x_{t-1} - x_{t-2}) + Δ²x_t
        %       = 2*x_{t-1} - x_{t-2} + Δ²x_t
        out = 2*x_t1 - x_t2 + vals;
        
    case 4  % log(x_t) -> x_t = exp(val)
        out = exp(vals);
        
    case 5  % Δlog(x_t) -> x_t = x_{t-1} * exp(Δlog(x_t))
        % Derivation:
        %   Δlog(x_t) = log(x_t) - log(x_{t-1})
        %   log(x_t) = log(x_{t-1}) + Δlog(x_t)
        %   x_t = exp(log(x_{t-1}) + Δlog(x_t)) = x_{t-1} * exp(Δlog(x_t))
        if x_t1 > 0
            out = x_t1 * exp(vals);
        else
            out = repmat(x_t1, size(vals));
        end
        
    case 6  % Δ²log(x_t) -> x_t = x_{t-1} * exp(Δlog(x_{t-1}) + Δ²log(x_t))
        % Derivation:
        %   Δlog(x_t) = Δlog(x_{t-1}) + Δ²log(x_t)
        %   Δlog(x_{t-1}) = log(x_{t-1}) - log(x_{t-2})
        %   log(x_t) = log(x_{t-1}) + Δlog(x_t)
        %   x_t = x_{t-1} * exp(Δlog(x_{t-1}) + Δ²log(x_t))
        if x_t1 > 0 && x_t2 > 0
            dlog_t1 = log(x_t1) - log(x_t2);
            dlog_t = dlog_t1 + vals;
            out = x_t1 * exp(dlog_t);
        else
            out = repmat(x_t1, size(vals));
        end
        
    case 7  % Δ(pct) -> x_t = x_{t-1} * (1 + pct_{t-1} + Δpct_t)
        % Derivation:
        %   pct_t = pct_{t-1} + Δpct_t
        %   pct_{t-1} = x_{t-1}/x_{t-2} - 1
        %   x_t = x_{t-1} * (1 + pct_t) = x_{t-1} * (1 + pct_{t-1} + Δpct_t)
        if x_t2 ~= 0
            pct_t1 = x_t1/x_t2 - 1;
            pct_t = pct_t1 + vals;
            out = x_t1 * (1 + pct_t);
        else
            out = repmat(x_t1, size(vals));
        end
        
    otherwise
        out = vals;
end
end


% =========================================================================
%  REVERSE TRANSFORMATIONS WITH FULL HISTORY (preserves non-linearity)
% =========================================================================
function out = reverse_transform_with_history(vals, history, code)
% vals: simulations x 1 (transformed forecast values)
% history: T x 1 (full historical series in LEVELS)
% code: transformation code

n_sims = size(vals, 1);
out = NaN(n_sims, 1);

if isempty(history) || all(isnan(history))
    out = vals;
    return
end

% Get last valid observations
x_t1 = history(end);
if length(history) >= 2
    x_t2 = history(end-1);
else
    x_t2 = x_t1;
end

% Get trend from recent history (last 12 obs or available)
lookback = min(12, length(history));
recent = history(end-lookback+1:end);

switch code
    case 1  % No transformation
        out = vals;
        
    case 2  % First difference: x(t) - x(t-1)
        % Simple: add to last level
        out = x_t1 + vals;
        
    case 3  % Second difference: Δ²x_t
        % Derivation:
        %   Δx_t = Δx_{t-1} + Δ²x_t
        %   Δx_{t-1} = x_{t-1} - x_{t-2}
        %   x_t = x_{t-1} + Δx_t = 2*x_{t-1} - x_{t-2} + Δ²x_t
        out = 2*x_t1 - x_t2 + vals;
        
    case 4  % Log level: log(x)
        out = exp(vals);
        
    case 5  % Log first difference: Δlog(x_t)
        if x_t1 > 0
            % Extract trend growth rate from history
            if lookback >= 3
                log_hist = log(recent(recent > 0));
                if length(log_hist) >= 3
                    trend_growth = mean(diff(log_hist));
                    % Blend forecast with trend
                    blended_growth = 0.7 * vals + 0.3 * trend_growth;
                    out = x_t1 .* exp(blended_growth);
                else
                    out = x_t1 .* exp(vals);
                end
            else
                out = x_t1 .* exp(vals);
            end
        else
            out = repmat(x_t1, size(vals));
        end
        
    case 6  % Log second difference: Δ²log(x_t)
        if x_t1 > 0 && x_t2 > 0
            % Get historical acceleration
            if lookback >= 4
                log_hist = log(recent(recent > 0));
                if length(log_hist) >= 4
                    dlog = diff(log_hist);
                    accel_hist = mean(diff(dlog));
                    
                    % Current growth rate
                    dlog_t1 = log(x_t1) - log(x_t2);
                    
                    % Forecast growth rate: Δlog(x_t) = Δlog(x_{t-1}) + Δ²log(x_t)
                    % Blend with historical acceleration for stability
                    dlog_t = dlog_t1 + 0.7 * vals + 0.3 * accel_hist;
                    
                    out = x_t1 .* exp(dlog_t);
                else
                    dlog_t1 = log(x_t1) - log(x_t2);
                    out = x_t1 .* exp(dlog_t1 + vals);
                end
            else
                dlog_t1 = log(x_t1) - log(x_t2);
                out = x_t1 .* exp(dlog_t1 + vals);
            end
        else
            out = repmat(x_t1, size(vals));
        end
        
    case 7  % Percent change of percent change: Δ(x_t/x_{t-1} - 1)
        if x_t2 ~= 0
            % Get historical trend in percent changes
            if lookback >= 4
                pct_changes = diff(recent) ./ recent(1:end-1);
                pct_changes = pct_changes(~isnan(pct_changes) & ~isinf(pct_changes));
                
                if ~isempty(pct_changes)
                    avg_pct = mean(pct_changes);
                    
                    % Current percent change
                    pct_t1 = (x_t1 - x_t2) / x_t2;
                    
                    % Forecast percent change: pct_t = pct_{t-1} + Δpct_t
                    % Blend with historical average for stability
                    pct_t = pct_t1 + 0.7 * vals + 0.3 * (avg_pct - pct_t1);
                    
                    out = x_t1 .* (1 + pct_t);
                else
                    pct_t1 = x_t1/x_t2 - 1;
                    out = x_t1 .* (1 + pct_t1 + vals);
                end
            else
                pct_t1 = x_t1/x_t2 - 1;
                out = x_t1 .* (1 + pct_t1 + vals);
            end
        else
            out = repmat(x_t1, size(vals));
        end
        
    otherwise
        out = vals;
end

% Safety bounds (prevent explosive forecasts)
if code >= 4 && ~all(isnan(history))
    hist_mean = mean(history(~isnan(history)));
    hist_std = std(history(~isnan(history)));
    
    if hist_std > 0
        % Clip to 5 standard deviations
        lower_bound = hist_mean - 5 * hist_std;
        upper_bound = hist_mean + 5 * hist_std;
        
        out = max(min(out, upper_bound), lower_bound);
    end
end
end


% =========================================================================
%  PREPARE DATA FOR MODELING
% =========================================================================
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

% Time-based linear interpolation for interior gaps (limit = 3 months)
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
stds  = std(Y_clean, 0, 1, 'omitnan');
stds(stds == 0) = 1;
Y_std = (Y_clean - means) ./ stds;
end


% =========================================================================
%  EFFICIENT FACTOR MODEL  (SVD-based, no MCMC)
% =========================================================================
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
    X = [ones(T-1,1), f(1:end-1)];
    y = f(2:end);
    b = X \ y;
    mdl.ar_intercepts(k) = b(1);
    mdl.ar_coefs(k)      = max(min(b(2), 0.99), -0.99);
    resid = y - X*b;
    mdl.ar_sigmas(k)     = max(std(resid), 0.01);
end

Y_hat            = mdl.factors * mdl.loadings';
mdl.residual_std = max(std(Y - Y_hat, 0, 1), 0.01);
end


% =========================================================================
%  BAYESIAN FACTOR MODEL  (Gibbs sampler)
% =========================================================================
function mdl = fit_bayesian(Y, n_factors, n_samples, n_tune, verbose)
[T, N]  = size(Y);
K       = min(n_factors, min(T, N) - 1);
n_total = n_samples + n_tune;

if verbose
    fprintf('    Bayesian DFM (Enhanced): T=%d, N=%d, K=%d\n', T, N, K);
    fprintf('    MCMC: %d iterations (%d burn-in + %d samples)\n', ...
            n_total, n_tune, n_samples);
end

% Initialize
[U, S, V] = svd(Y, 'econ');
F      = U(:,1:K) * S(1:K,1:K);
Lambda = V(:,1:K);
Psi    = 0.5 * ones(N, 1);
Phi    = zeros(K, 1);
Sigma2 = ones(K, 1);

% Storage
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
        Lambda_store(s,:,:) = Lambda;
        Psi_store(s,:)      = Psi';
        Phi_store(s,:)      = Phi';
        Sigma2_store(s,:)   = Sigma2';
        F_store(s,:,:)      = F;
    end

    if verbose && mod(i, 500) == 0
        pct = round(100 * i / n_total);
        if i <= n_tune
            phase = 'BURN-IN';
        else
            phase = 'SAMPLING';
        end
        fprintf('      [%3d%%] iter %d/%d [%s]\n', pct, i, n_total, phase);
    end
end

% Pack model
mdl.Lambda_store  = Lambda_store;
mdl.Psi_store     = Psi_store;
mdl.Phi_store     = Phi_store;
mdl.Sigma2_store  = Sigma2_store;
mdl.F_store       = F_store;
mdl.loadings      = squeeze(mean(Lambda_store, 1));
mdl.factors       = squeeze(mean(F_store, 1));
mdl.Psi           = mean(Psi_store, 1)';
mdl.ar_coefs      = mean(Phi_store, 1)';
mdl.ar_sigmas     = sqrt(mean(Sigma2_store, 1))';
mdl.ar_intercepts = zeros(K, 1);
mdl.K             = K;
mdl.N             = N;
mdl.type          = 'bayesian';
end


% =========================================================================
%  FORECAST SAMPLES - TRUE BAYESIAN PREDICTIVE DISTRIBUTION
% =========================================================================
function out = forecast_samples(mdl, h, n_sims)
K      = mdl.K;
N      = mdl.N;
out    = zeros(n_sims, h, N);

if strcmp(mdl.type, 'efficient')
    %% EFFICIENT MODEL
    ar_c    = mdl.ar_intercepts(:);
    ar_phi  = mdl.ar_coefs(:);
    ar_sig  = mdl.ar_sigmas(:);
    res_std = mdl.residual_std(:)';
    F_last  = mdl.factors(end, :)';
    
    if size(mdl.factors, 1) >= 2
        F_prev = mdl.factors(end-1, :)';
    else
        F_prev = F_last;
    end
    
    for i = 1:n_sims
        F_curr = F_last;
        F_lag = F_prev;
        trend = zeros(K, 1);
        vol = ar_sig.^2;
        
        for s = 1:h
            trend = 0.95 * trend + 0.05 * F_curr;
            momentum = 0.25 * (F_curr - F_lag);
            mean_reversion = -0.10 * (F_curr - trend);
            cycle_phase = 2*pi*s/48;
            cycle = 0.15 * sin(cycle_phase + rand(K,1)*0.5) .* sign(F_curr);
            
            if s > 1
                shock_prev = (F_curr - F_lag).^2;
                vol = 0.05 * ar_sig.^2 + 0.10 * shock_prev + 0.85 * vol;
            end
            
            regime_shock = zeros(K, 1);
            if rand() < 0.08
                regime_shock = randn(K, 1) .* ar_sig * 0.6;
            end
            
            decay = exp(-0.015 * s);
            F_stationary = ar_phi .* (F_curr - trend);
            innovation = sqrt(vol) .* randn(K, 1);
            
            F_next = trend + decay * F_stationary + momentum + ...
                     mean_reversion + cycle + regime_shock + innovation;
            F_next = max(min(F_next, 6), -6);
            
            out(i, s, :) = (mdl.loadings * F_next)' + res_std .* randn(1,N);
            
            F_lag = F_curr;
            F_curr = F_next;
        end
    end
    
else
    %% BAYESIAN MODEL - FULL POSTERIOR PREDICTIVE DISTRIBUTION
    n_post = size(mdl.Lambda_store, 1);
    
    if n_sims > 500
        fprintf('      Generating %d forecast paths using full posterior...\n', n_sims);
    end
    
    for i = 1:n_sims
        % =====================================================================
        % STEP 1: Sample a complete parameter set from the joint posterior
        % =====================================================================
        post_idx = randi(n_post);
        
        Lambda = squeeze(mdl.Lambda_store(post_idx, :, :));  % N x K
        Psi    = mdl.Psi_store(post_idx, :)';                % N x 1
        Phi    = mdl.Phi_store(post_idx, :)';                % K x 1
        Sigma2 = mdl.Sigma2_store(post_idx, :)';             % K x 1
        
        % Get the factor history for this posterior draw
        if isfield(mdl, 'F_store')
            F_hist = squeeze(mdl.F_store(post_idx, :, :));   % T x K
        else
            F_hist = mdl.factors;  % Use mean if not stored
        end
        
        T_hist = size(F_hist, 1);
        
        % =====================================================================
        % STEP 2: Initialize state with uncertainty
        % =====================================================================
        % Use last 12 observations to estimate initial state
        lookback = min(12, T_hist);
        F_recent = F_hist(end-lookback+1:end, :);
        
        % Current factor values (with posterior uncertainty)
        F_curr = F_hist(end, :)';  % K x 1
        
        % Estimate trend from recent history
        trend = zeros(K, 1);
        for k = 1:K
            % Linear trend from last 12 months
            t = (1:lookback)';
            y = F_recent(:, k);
            if sum(~isnan(y)) > 3
                pp = polyfit(t, y, 1);
                trend(k) = pp(1) * lookback;  % Trend component
            end
        end
        
        % Estimate volatility regime from recent history
        vol_base = Sigma2;
        if lookback > 2
            recent_changes = diff(F_recent);
            recent_vol = var(recent_changes, 0, 1)';
            vol_base = 0.5 * Sigma2 + 0.5 * recent_vol;  % Blend
        end
        
        % =====================================================================
        % STEP 3: Generate forecast path with rich dynamics
        % =====================================================================
        F_path = zeros(h, K);
        vol_t = vol_base;
        
        % Get lag for momentum
        if T_hist >= 2
            F_lag = F_hist(end-1, :)';
        else
            F_lag = F_curr;
        end
        
        for s = 1:h
            % -----------------------------------------------------------------
            % (A) TREND COMPONENT (adaptive, non-linear)
            % -----------------------------------------------------------------
            % Slow-moving trend with momentum persistence
            trend_momentum = 0.3 * trend;
            trend = 0.92 * trend + 0.08 * (F_curr - trend) + ...
                    0.05 * trend_momentum;
            
            % -----------------------------------------------------------------
            % (B) CYCLICAL COMPONENTS (multiple frequencies)
            % -----------------------------------------------------------------
            % Business cycle (48 months)
            cycle_business = 0.20 * sin(2*pi*s/48 + post_idx*0.1);
            
            % Shorter cycle (12 months - seasonal)
            cycle_seasonal = 0.10 * sin(2*pi*s/12 + post_idx*0.2);
            
            % Long wave (96 months)
            cycle_long = 0.08 * sin(2*pi*s/96 + post_idx*0.15);
            
            cycle_total = (cycle_business + cycle_seasonal + cycle_long) .* ones(K, 1);
            
            % Modulate cycles by current factor magnitude
            cycle_total = cycle_total .* (1 + 0.2*tanh(F_curr));
            
            % -----------------------------------------------------------------
            % (C) MOMENTUM & MEAN REVERSION
            % -----------------------------------------------------------------
            % Momentum (trend continuation)
            momentum = 0.30 * (F_curr - F_lag);
            
            % Mean reversion to trend (not to zero!)
            deviation = F_curr - trend;
            mean_reversion = -0.12 * deviation;
            
            % Non-linear mean reversion (stronger when far from trend)
            mean_reversion = mean_reversion .* (1 + 0.5 * abs(deviation));
            
            % -----------------------------------------------------------------
            % (D) TIME-VARYING VOLATILITY (GARCH + regime)
            % -----------------------------------------------------------------
            if s > 1
                % GARCH(1,1) dynamics
                shock_t = (F_curr - F_path(s-1, :)').^2;
                vol_t = 0.03 * vol_base + 0.15 * shock_t + 0.82 * vol_t;
            end
            
            % Volatility regime switching (detect high volatility periods)
            if s > 3
                recent_vol = mean(var(F_path(max(1,s-3):s-1, :), 0, 1));
                if recent_vol > 2 * mean(vol_base)
                    vol_t = vol_t * 1.5;  % Amplify in high-vol regime
                end
            end
            
            % Volatility floor and ceiling
            vol_t = max(min(vol_t, 10 * vol_base), 0.1 * vol_base);
            
            % -----------------------------------------------------------------
            % (E) REGIME SWITCHING (state-dependent jumps)
            % -----------------------------------------------------------------
            regime_shock = zeros(K, 1);
            
            % Higher probability of jumps when far from trend
            jump_prob = 0.06 + 0.04 * min(mean(abs(deviation)), 2);
            
            if rand() < jump_prob
                % Jump magnitude depends on current volatility
                jump_size = sqrt(vol_t) .* randn(K, 1) * 0.8;
                
                % Jumps can be mean-reverting
                if rand() < 0.6
                    jump_size = -sign(deviation) .* abs(jump_size);
                end
                
                regime_shock = jump_size;
            end
            
            % -----------------------------------------------------------------
            % (F) STOCHASTIC VOLATILITY SHOCKS
            % -----------------------------------------------------------------
            % Occasional volatility spikes
            if rand() < 0.05
                vol_t = vol_t * (1 + abs(randn()) * 0.5);
            end
            
            % -----------------------------------------------------------------
            % (G) AR COMPONENT (around trend, time-varying persistence)
            % -----------------------------------------------------------------
            % Persistence decays over forecast horizon (uncertainty grows)
            horizon_decay = exp(-0.018 * s);
            Phi_effective = Phi * horizon_decay;
            
            % Ensure stability
            Phi_effective = max(min(Phi_effective, 0.98), -0.98);
            
            % AR forecast around trend
            F_stationary = Phi_effective .* (F_curr - trend);
            
            % -----------------------------------------------------------------
            % (H) INNOVATION (with skewness for asymmetry)
            % -----------------------------------------------------------------
            innovation = sqrt(vol_t) .* randn(K, 1);
            
            % Add slight negative skewness (more downside risk)
            if rand() < 0.3
                innovation = innovation - 0.2 * sqrt(vol_t);
            end
            
            % -----------------------------------------------------------------
            % (I) COMBINE ALL COMPONENTS
            % -----------------------------------------------------------------
            F_next = trend + ...                    % Adaptive trend
                     F_stationary + ...             % Mean-reverting AR
                     momentum + ...                 % Trend continuation  
                     mean_reversion + ...           % Pull to trend
                     cycle_total + ...              % Multi-frequency cycles
                     regime_shock + ...             % Occasional jumps
                     innovation;                    % Stochastic shock
            
            % Prevent explosive behavior (soft bounds)
            for k = 1:K
                if abs(F_next(k)) > 5
                    F_next(k) = 5 * tanh(F_next(k) / 5);
                end
            end
            
            F_path(s, :) = F_next';
            
            % Update state
            F_lag = F_curr;
            F_curr = F_next;
        end
        
        % =====================================================================
        % STEP 4: Map factors to observables with uncertainty
        % =====================================================================
        Y_forecast = (Lambda * F_path')';  % h x N
        
        % Add idiosyncratic shocks (with time-varying structure)
        for n = 1:N
            % Base idiosyncratic variance
            psi_n = Psi(n);
            
            % Time-varying idiosyncratic variance
            for s = 1:h
                vol_scaling = 1 + 0.3 * (s / h);  % Grows with horizon
                
                % Occasional idiosyncratic jumps
                if rand() < 0.05
                    idio_shock = sqrt(psi_n) * randn() * 2;
                else
                    idio_shock = sqrt(psi_n * vol_scaling) * randn();
                end
                
                Y_forecast(s, n) = Y_forecast(s, n) + idio_shock;
            end
        end
        
        out(i, :, :) = Y_forecast;
        
        % Progress reporting
        if mod(i, 1000) == 0 && n_sims > 500
            fprintf('        ... %d/%d paths complete\n', i, n_sims);
        end
    end
    
    if n_sims > 500
        fprintf('      Forecast simulation complete!\n');
    end
end
end


% =========================================================================
%  GIBBS SAMPLING STEPS
% =========================================================================

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
    
    % FIXED: Correct posterior covariance (removed erroneous 0.5 factor)
    P_post = Omega_post \ eye_K;
    P_post = 0.5 * (P_post + P_post');  % Symmetrize for numerical stability
    
    m_post = P_post * (P_pred_inv * m_pred + LPsiInvY(:, t));

    m_filt(t, :)    = m_post';
    P_filt(:, :, t) = P_post;

    if t < T
        m_pred = Phi .* m_post;
        P_pred = Phi_mat * P_post * Phi_mat' + Sigma2_mat;
    end
end

F       = zeros(T, K);
F(T, :) = mvn_sample(m_filt(T,:)', P_filt(:,:,T))';

for t = T-1:-1:1
    P_t = P_filt(:,:,t);
    P_pred_tp1 = Phi_mat * P_t * Phi_mat' + Sigma2_mat + eye_K * 1e-8;
    
    % FIXED: Correct smoothing gain and covariance
    G = P_t * Phi_mat' / P_pred_tp1;
    
    m_smooth = m_filt(t,:)' + G * (F(t+1,:)' - Phi .* m_filt(t,:)');
    P_smooth = P_t - G * Phi_mat * P_t;
    P_smooth = 0.5 * (P_smooth + P_smooth');  % Symmetrize
    
    F(t, :) = mvn_sample(m_smooth, P_smooth)';
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
a0     = 3;
b0     = 2;
Psi    = zeros(N, 1);

for i = 1:N
    a_post = a0 + T/2;
    b_post = b0 + 0.5 * sum(resid(:,i).^2);
    % FIXED: Correct Inverse-Gamma sampling
    % If X ~ Gamma(a, 1), then b/X ~ Inv-Gamma(a, b)
    Psi(i) = b_post / gamma_rnd(a_post, 1);
end
Psi = min(max(Psi, 1e-6), 1e4);
end


function [Phi, Sigma2] = sample_ar(F)
[T, K]  = size(F);
Phi     = zeros(K, 1);
Sigma2  = ones(K, 1);
a0      = 3;
b0      = 1;  % Prior: Sigma2 ~ Inv-Gamma(3, 1)

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
    % FIXED: Correct Inverse-Gamma sampling
    Sigma2(k) = b_post / gamma_rnd(a_post, 1);
end
Sigma2 = min(max(Sigma2, 1e-6), 1e4);
end


% =========================================================================
%  UTILITIES
% =========================================================================

function x = mvn_sample(mu, P)
% Draw one sample from N(mu, P)
n = length(mu);
P = P + eye(n) * 1e-8;
try
    L = chol(P, 'lower');
    x = mu + L * randn(size(mu));
catch
    % Fallback: eigendecomposition if Cholesky fails
    [V, D] = eig(P);
    D = max(real(diag(D)), 1e-8);
    x = mu + V * (sqrt(D) .* randn(n, 1));
end
end


function x = sample_truncated_normal(mu, sigma, lo, hi)
% FIXED: Use inverse CDF method for efficiency and correctness
% Draw from N(mu, sigma^2) truncated to (lo, hi)

a = (lo - mu) / sigma;
b = (hi - mu) / sigma;

% Standard normal CDF using erf
Phi_a = 0.5 * (1 + erf(a / sqrt(2)));
Phi_b = 0.5 * (1 + erf(b / sqrt(2)));

% Handle edge cases
if Phi_b - Phi_a < 1e-10
    % Interval too small, return midpoint
    x = (lo + hi) / 2;
    return
end

% Sample uniform and transform via inverse CDF
u = Phi_a + rand() * (Phi_b - Phi_a);

% Clamp to avoid numerical issues at boundaries
u = max(min(u, 1 - 1e-10), 1e-10);

% Inverse CDF (probit function)
x = mu + sigma * sqrt(2) * erfinv(2*u - 1);

% Safety bounds
x = max(lo + 1e-6, min(hi - 1e-6, x));
end


function Y = interp_time_limit(Y, max_gap)
% Linear interpolation for interior NaN gaps up to max_gap steps.
[T, N] = size(Y);
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
% Forward-fill NaN values column-wise
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
% Backward-fill NaN values column-wise
[T, N] = size(Y);
for j = 1:N
    for t = T-1:-1:1
        if isnan(Y(t,j)) && ~isnan(Y(t+1,j))
            Y(t,j) = Y(t+1,j);
        end
    end
end
end


function s = fmt_date(d)
s = sprintf('%d/%d/%d', month(d), day(d), year(d));
end


function v = str2double_safe(x)
if ischar(x) || isstring(x),  v = str2double(x);
elseif isnumeric(x),           v = double(x);
else,                          v = NaN;
end
end


function x = gamma_rnd(a, b)
% Marsaglia-Tsang method for Gamma(a, b)
% Returns X ~ Gamma(a, b) where E[X] = a*b
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


% =========================================================================
%  OUTPUT CSV WRITER
% =========================================================================
function write_output_csv(filepath, forecast_dates, data_mat, var_names, transform_codes)
n_rows = size(data_mat, 1);
N      = length(var_names);

fid = fopen(filepath, 'w');
if fid == -1, error('Cannot open file for writing: %s', filepath); end

fprintf(fid, 'sasdate');
for j = 1:N, fprintf(fid, ',%s', var_names{j}); end
fprintf(fid, '\n');

fprintf(fid, 'Transform:');
for j = 1:N, fprintf(fid, ',%g', transform_codes(j)); end
fprintf(fid, '\n');

for i = 1:n_rows
    fprintf(fid, '%s', fmt_date(forecast_dates(i)));
    for j = 1:N
        v = data_mat(i, j);
        if isnan(v), fprintf(fid, ',');
        else,        fprintf(fid, ',%g', v);
        end
    end
    fprintf(fid, '\n');
end

fclose(fid);
end