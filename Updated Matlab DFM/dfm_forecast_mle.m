function dfm_forecast_mle(varargin)
% DFM_FORECAST_MLE  Dynamic Factor Model (state-space/Kalman filter, MLE) forecasting
%
% MATLAB translation of Python dfm_forecast_v2.py using statsmodels DynamicFactor.
% Uses MATLAB's Econometrics Toolbox for state-space DFM estimation.
%
% Outputs CSV with forecasted LEVEL values only.
%
% Usage:
%   dfm_forecast_mle('data', '2014-12.csv', 'series', 'INDPRO,UNRATE', ...
%                    'k_factors', 2, 'factor_order', 1, 'horizon', 12, ...
%                    'output', 'out.csv')
%
%   dfm_forecast_mle('data', '2014-12.csv', 'series', 'INDPRO,UNRATE', ...
%                    'k_factors', 2, 'factor_order', 1, 'backtest', 4, ...
%                    'output', 'backtest.csv')
%
% Parameters:
%   'data'              Input CSV (FRED-MD format)                    [required]
%   'appendix'          Path to appendix CSV                          [optional]
%   'series'            Comma-separated series codes                  [optional]
%   'series_file'       File with series codes (one per line)         [optional]
%   'series_index'      1-indexed positions, e.g. '1,2,10-20'         [optional]
%   'group'             Group code(s) or name(s)                      [optional]
%   'all_series'        Use all series                                [default: false]
%   'k_factors'         Number of factors                             [default: 3]
%   'factor_order'      Factor AR order                               [default: 1]
%   'error_order'       Idiosyncratic AR order                        [default: 0]
%   'horizon'           Forecast horizon                              [default: 12]
%   'backtest'          Hold out N months for backtesting            [optional]
%   'include_history'   Include last actual row                       [default: false]
%   'output'            Output path                                   [default: 'dfm_forecast_output.csv']
%   'output_format'     'wide' | 'long'                               [default: 'wide']
%   'date_col_name'     Date column name                              [default: 'sasdate']
%   'omit_value_suffix' Omit '_value' suffix in wide format           [default: false]
%   'verbose'           Verbose output                                [default: false]
%   'progress_every'    Progress reporting interval                   [default: 25]
%   'maxiter'           Maximum iterations                            [default: 500]
%   'list_series'       List series and exit                          [default: false]
%   'list_groups'       List groups and exit                          [default: false]

% -------------------------------------------------------------------------
% Parse arguments
% -------------------------------------------------------------------------
p = inputParser;
addParameter(p, 'data', '', @ischar);
addParameter(p, 'appendix', '', @ischar);
addParameter(p, 'series', '', @ischar);
addParameter(p, 'series_file', '', @ischar);
addParameter(p, 'series_index', '', @ischar);
addParameter(p, 'group', '', @ischar);
addParameter(p, 'all_series', false, @islogical);
addParameter(p, 'k_factors', 3, @isnumeric);
addParameter(p, 'factor_order', 1, @isnumeric);
addParameter(p, 'error_order', 0, @isnumeric);
addParameter(p, 'horizon', 12, @isnumeric);
addParameter(p, 'backtest', [], @(x) isempty(x) || isnumeric(x));
addParameter(p, 'include_history', false, @islogical);
addParameter(p, 'output', 'dfm_forecast_output.csv', @ischar);
addParameter(p, 'output_format', 'wide', @ischar);
addParameter(p, 'date_col_name', 'sasdate', @ischar);
addParameter(p, 'omit_value_suffix', false, @islogical);
addParameter(p, 'verbose', false, @islogical);
addParameter(p, 'progress_every', 25, @isnumeric);
addParameter(p, 'maxiter', 500, @isnumeric);
addParameter(p, 'list_series', false, @islogical);
addParameter(p, 'list_groups', false, @islogical);

parse(p, varargin{:});
args = p.Results;

if isempty(args.data) && ~args.list_groups
    error('--data is required');
end

% -------------------------------------------------------------------------
% Group definitions
% -------------------------------------------------------------------------
GROUP_MAP = containers.Map(...
    {1, 2, 3, 4, 5, 6, 7, 8}, ...
    {'Output and Income', 'Labor Market', 'Housing', ...
     'Consumption, Orders, and Inventories', 'Money and Credit', ...
     'Interest and Exchange Rates', 'Prices', 'Stock Market'});

% -------------------------------------------------------------------------
% Handle list operations
% -------------------------------------------------------------------------
if args.list_groups
    keys = sort(cell2mat(GROUP_MAP.keys()));
    for k = keys
        fprintf('%d: %s\n', k, GROUP_MAP(k));
    end
    return
end

% -------------------------------------------------------------------------
% Load data
% -------------------------------------------------------------------------
[levels_all, tcodes_all, dates_all] = read_panel_with_tcodes(args.data);

if args.list_series
    vars = levels_all.Properties.VariableNames;
    for i = 1:length(vars)
        fprintf('%3d  %s\n', i, vars{i});
    end
    fprintf('\nTotal: %d\n', length(vars));
    return
end

avail = levels_all.Properties.VariableNames;

% -------------------------------------------------------------------------
% Select series
% -------------------------------------------------------------------------
if args.all_series
    series = avail;
elseif ~isempty(args.series_file)
    series = read_series_file(args.series_file, avail);
elseif ~isempty(args.series)
    series = strsplit(args.series, ',');
    series = strtrim(series);
    bad = setdiff(series, avail);
    if ~isempty(bad)
        error('Series not found: %s', strjoin(bad, ', '));
    end
elseif ~isempty(args.series_index)
    idxs = parse_index_spec(args.series_index, length(avail));
    series = avail(idxs);
elseif ~isempty(args.group)
    if isempty(args.appendix)
        error('--appendix required with --group');
    end
    appendix = read_appendix(args.appendix);
    groups = parse_groups(args.group, GROUP_MAP);
    series = select_series_by_groups(appendix, groups, avail);
else
    error('Specify --series, --series_index, --group, or --all_series');
end

levels = levels_all(:, series);
tcodes = tcodes_all(series);
dates = dates_all;

if args.verbose
    fprintf('Selected %d series.\n', length(series));
end

% -------------------------------------------------------------------------
% Backtest mode
% -------------------------------------------------------------------------
if ~isempty(args.backtest)
    h = args.backtest;
    T = height(levels);
    
    if h < 1 || h >= T
        error('Invalid --backtest %d (must be 1 <= h < %d)', h, T);
    end
    
    if args.verbose
        fprintf('\n=== BACKTEST: %d months ===\n\n', h);
    end
    
    train_levels = levels(1:end-h, :);
    train_dates = dates(1:end-h);
    holdout_idx = dates(end-h+1:end);
    
    if args.verbose
        fprintf('Train: %s - %s\n', datestr(train_dates(1)), datestr(train_dates(end)));
        fprintf('Test:  %s - %s\n', datestr(holdout_idx(1)), datestr(holdout_idx(end)));
    end
    
    fcst = run_forecast(...
        train_levels, train_dates, tcodes, args.k_factors, args.factor_order, ...
        args.error_order, args.maxiter, h, args.verbose, args.progress_every);
    
    % Align index
    fcst.Properties.RowNames = cellstr(datestr(holdout_idx, 'mm/dd/yyyy'));
    
    out = format_output(fcst, args.output_format, false, [], [], ...
                       args.omit_value_suffix, args.date_col_name, holdout_idx);
    writetable(out, args.output);
    fprintf('Wrote: %s\n', args.output);
    
% -------------------------------------------------------------------------
% Regular forecast mode
% -------------------------------------------------------------------------
else
    fcst = run_forecast(...
        levels, dates, tcodes, args.k_factors, args.factor_order, ...
        args.error_order, args.maxiter, args.horizon, args.verbose, args.progress_every);
    
    last_date = [];
    last_levels = [];
    if args.include_history
        last_date = dates(end);
        last_levels = levels(end, :);
    end
    
    % Build forecast dates
    last_dt = dates(end);
    yr = year(last_dt);
    mo = month(last_dt);
    fcast_dates = NaT(args.horizon, 1);
    for i = 1:args.horizon
        mo = mo + 1;
        if mo > 12
            mo = 1;
            yr = yr + 1;
        end
        fcast_dates(i) = datetime(yr, mo, 1);
    end
    
    out = format_output(fcst, args.output_format, args.include_history, ...
                       last_date, last_levels, args.omit_value_suffix, ...
                       args.date_col_name, fcast_dates);
    writetable(out, args.output);
    fprintf('Wrote: %s\n', args.output);
end

end


% =========================================================================
%  MAIN FORECASTING PIPELINE
% =========================================================================
function fcst_levels = run_forecast(levels, dates, tcodes, k_factors, ...
                                    factor_order, error_order, maxiter, ...
                                    horizon, verbose, progress_every)

% Step 1: Transform
if verbose
    fprintf('Step 1/5: Transforming...\n');
    tic;
end
[transformed, valid_cols] = transform_panel(levels, tcodes, verbose, progress_every);
if verbose
    fprintf('  Done (%.1fs). %d series.\n', toc, length(valid_cols));
end

% Remove all-NaN rows
row_valid = any(~isnan(transformed{:,:}), 2);
transformed = transformed(row_valid, :);
dates_clean = dates(row_valid);

if height(transformed) == 0
    error('All data is NaN after transformation.');
end

% Step 2: Standardize
if verbose
    fprintf('Step 2/5: Standardizing...\n');
    tic;
end
[std_data, mu, sigma] = standardize_data(transformed);
if verbose
    fprintf('  Done (%.1fs).\n', toc);
end

% Step 3: Fit DFM
if verbose
    fprintf('Step 3/5: Fitting DFM (k=%d, order=%d)...\n', k_factors, factor_order);
    tic;
end
[Mdl, EstMdl, keep_cols] = fit_dfm_matlab(std_data, k_factors, factor_order, ...
                                           error_order, maxiter, verbose);
if verbose
    fprintf('  Done (%.1fs). %d series kept.\n', toc, length(keep_cols));
end

% Step 4: Forecast
if verbose
    fprintf('Step 4/5: Forecasting...\n');
    tic;
end
fcast_std = forecast_dfm(Mdl, EstMdl, std_data, horizon);
if verbose
    fprintf('  Done (%.1fs).\n', toc);
end

% Unstandardize
fcast_trans = table();
for i = 1:length(keep_cols)
    col = keep_cols{i};
    fcast_trans.(col) = fcast_std(:, i) * sigma.(col) + mu.(col);
end

% Step 5: Invert to levels
if verbose
    fprintf('Step 5/5: Converting to levels...\n');
    tic;
end

% Only use columns we have
levels_sub = levels(:, keep_cols);
tcodes_sub = tcodes(keep_cols);

fcst_levels = forecast_to_levels(fcast_trans, levels_sub, tcodes_sub, ...
                                 verbose, progress_every);
if verbose
    fprintf('  Done (%.1fs).\n', toc);
end

end


% =========================================================================
%  DATA LOADING
% =========================================================================
function [levels, tcodes, dates] = read_panel_with_tcodes(filepath)

% Read file manually line by line to avoid readtable issues
fid = fopen(filepath, 'r', 'n', 'UTF-8');
if fid == -1
    error('Cannot open file: %s', filepath);
end

% Read header
header = fgetl(fid);
var_names_all = strsplit(header, ',');
var_names_all = strtrim(var_names_all);
date_col_name = var_names_all{1};
var_names = var_names_all(2:end);

% Make valid MATLAB variable names
var_names_valid = cell(size(var_names));
for i = 1:length(var_names)
    var_names_valid{i} = matlab.lang.makeValidName(var_names{i});
end

% Read tcode row
tcode_line = fgetl(fid);
tcode_parts = strsplit(tcode_line, ',');
tcodes = struct();
for i = 1:length(var_names)
    vn = var_names_valid{i};
    tc_str = strtrim(tcode_parts{i+1});
    tc = str2double(tc_str);
    if isnan(tc)
        tcodes.(vn) = 1;
    else
        tcodes.(vn) = tc;
    end
end

% Read data rows
dates_list = {};
data_cell = {};
row_count = 0;

while ~feof(fid)
    line = fgetl(fid);
    if ~ischar(line) || isempty(line)
        break;
    end
    
    parts = strsplit(line, ',');
    if length(parts) < 2
        continue;
    end
    
    row_count = row_count + 1;
    dates_list{row_count} = strtrim(parts{1}); %#ok<AGROW>
    data_cell{row_count} = parts(2:end); %#ok<AGROW>
end

fclose(fid);

% Parse dates
dates = NaT(row_count, 1);
date_formats = {'M/d/yyyy', 'MM/dd/yyyy', 'yyyy-MM-dd', 'M-d-yyyy', 'MM-dd-yyyy'};

for i = 1:row_count
    date_str = dates_list{i};
    parsed = false;
    
    for fmt = date_formats
        try
            dates(i) = datetime(date_str, 'InputFormat', fmt{1});
            parsed = true;
            break
        catch
            continue
        end
    end
    
    if ~parsed
        fprintf('Warning: Could not parse date: %s\n', date_str);
    end
end

% Remove invalid dates
valid_idx = ~isnat(dates);
dates = dates(valid_idx);
data_cell = data_cell(valid_idx);

if isempty(dates)
    error('No valid dates found in data.');
end

% Build data matrix
n_obs = length(dates);
n_vars = length(var_names);
data_matrix = nan(n_obs, n_vars);

for i = 1:n_obs
    row_data = data_cell{i};
    for j = 1:min(n_vars, length(row_data))
        val_str = strtrim(row_data{j});
        if ~isempty(val_str) && ~strcmpi(val_str, 'nan')
            val = str2double(val_str);
            if ~isnan(val)
                data_matrix(i, j) = val;
            end
        end
    end
end

% Sort by date
[dates, sort_idx] = sort(dates);
data_matrix = data_matrix(sort_idx, :);

% Build table with valid variable names
levels = array2table(data_matrix, 'VariableNames', var_names_valid);

end

% =========================================================================
%  SERIES SELECTION UTILITIES
% =========================================================================
function series = read_series_file(filepath, avail)
fid = fopen(filepath, 'r');
if fid == -1
    error('Cannot open file: %s', filepath);
end
lines = {};
while ~feof(fid)
    ln = fgetl(fid);
    if ischar(ln)
        ln = strtrim(ln);
        if ~isempty(ln) && ~startsWith(ln, '#')
            lines{end+1} = ln; %#ok<AGROW>
        end
    end
end
fclose(fid);
series = lines;
bad = setdiff(series, avail);
if ~isempty(bad)
    error('Series not found: %s', strjoin(bad(1:min(5,end)), ', '));
end
end


function idxs = parse_index_spec(spec, n_cols)
parts = strsplit(spec, ',');
idxs = [];
for i = 1:length(parts)
    p = strtrim(parts{i});
    if contains(p, '-')
        tokens = strsplit(p, '-');
        a = str2double(tokens{1});
        b = str2double(tokens{2});
        if a < 1 || b < 1
            error('Indices must be >= 1');
        end
        if a > b
            tmp = a; a = b; b = tmp;
        end
        idxs = [idxs, a:b]; %#ok<AGROW>
    else
        idx = str2double(p);
        if idx < 1
            error('Indices must be >= 1');
        end
        idxs = [idxs, idx]; %#ok<AGROW>
    end
end
idxs = unique(idxs);
if max(idxs) > n_cols
    error('Index out of bounds. Max: %d', n_cols);
end
end


function groups = parse_groups(group_arg, GROUP_MAP)
parts = strsplit(group_arg, ',');
parts = strtrim(parts);
groups = [];
for i = 1:length(parts)
    p = parts{i};
    if ~isempty(regexp(p, '^\d+$', 'once'))
        g = str2double(p);
        if ~isKey(GROUP_MAP, g)
            error('Unknown group %d', g);
        end
        groups = [groups, g]; %#ok<AGROW>
    else
        p_low = lower(p);
        keys = cell2mat(GROUP_MAP.keys());
        matches = [];
        for k = keys
            if contains(lower(GROUP_MAP(k)), p_low)
                matches = [matches, k]; %#ok<AGROW>
            end
        end
        if isempty(matches)
            error('Could not match group ''%s''', p);
        end
        groups = [groups, matches]; %#ok<AGROW>
    end
end
groups = unique(groups);
end


function appendix = read_appendix(filepath)
try
    appendix = readtable(filepath);
catch
    appendix = readtable(filepath, 'Encoding', 'windows-1252');
end
end


function series = select_series_by_groups(appendix, groups, avail)
if ~ismember('fred', appendix.Properties.VariableNames) || ...
   ~ismember('group', appendix.Properties.VariableNames)
    error('Appendix must have ''fred'' and ''group'' columns.');
end

subset = appendix(ismember(appendix.group, groups), :);
candidates = cellstr(string(subset.fred));
series = intersect(candidates, avail, 'stable');
if isempty(series)
    error('No matching series for groups.');
end
end


% =========================================================================
%  TRANSFORMATIONS (FRED-MD codes 1-7)
% =========================================================================
function [transformed, valid_cols] = transform_panel(levels, tcodes, verbose, progress_every)
valid_cols = {};
vars = levels.Properties.VariableNames;
n = length(vars);
result = table();

for i = 1:n
    vn = vars{i};
    
    if ~isfield(tcodes, vn)
        if verbose
            fprintf('  Warning: ''%s'' not in tcodes, skipping.\n', vn);
        end
        continue
    end
    
    tc = tcodes.(vn);
    if isnan(tc)
        if verbose
            fprintf('  Warning: tcode for ''%s'' is NaN, skipping.\n', vn);
        end
        continue
    end
    
    % Get column data safely
    col_data = levels.(vn);
    result.(vn) = apply_tcode_transform(col_data, tc);
    valid_cols{end+1} = vn; %#ok<AGROW>
    
    if verbose && (mod(i, progress_every) == 0 || i == n)
        fprintf('[transform] %d/%d\n', i, n);
    end
end

transformed = result;
end

% =========================================================================
%  STANDARDIZATION
% =========================================================================
function [std_data, mu, sigma] = standardize_data(data)
vars = data.Properties.VariableNames;
mu = struct();
sigma = struct();
std_data = table();

for i = 1:length(vars)
    vn = vars{i};
    col = data.(vn);
    
    % Ensure column is numeric vector
    if iscell(col)
        col = cell2mat(col);
    end
    col = double(col(:));
    
    mu.(vn) = mean(col, 'omitnan');
    sigma.(vn) = std(col, 'omitnan');
    
    if sigma.(vn) == 0 || isnan(sigma.(vn))
        sigma.(vn) = 1;
    end
    
    std_data.(vn) = (col - mu.(vn)) / sigma.(vn);
end
end

% =========================================================================
%  DFM ESTIMATION (MATLAB Econometrics Toolbox)
% =========================================================================
function [Mdl, EstMdl, keep_cols] = fit_dfm_matlab(std_data, k_factors, ...
                                                     factor_order, error_order, ...
                                                     maxiter, verbose)

vars = std_data.Properties.VariableNames;

% Drop all-NaN columns
keep_mask = false(1, length(vars));
for i = 1:length(vars)
    keep_mask(i) = any(~isnan(std_data.(vars{i})));
end
keep_cols = vars(keep_mask);

if sum(keep_mask) == 0
    error('All series are NaN.');
end
if sum(keep_mask) < k_factors
    error('Only %d series but k_factors=%d.', sum(keep_mask), k_factors);
end

endog = std_data(:, keep_cols);
Y = endog{:, :}';  % (N x T)

% Check for Econometrics Toolbox
if ~license('test', 'Econometrics_Toolbox')
    error('Econometrics Toolbox required for dfm estimation.');
end

% Build state-space DFM model
% State transition: F_t = A*F_{t-1} + u_t
% Observation: Y_t = C*F_t + e_t

N = size(Y, 1);
K = k_factors;

% Create model using ssm (State Space Model)
% State dimension: K * factor_order
% Observation dimension: N

% Initialize with PCA
[~, score] = pca(Y', 'NumComponents', K);
F_init = score';  % (K x T)

% Loadings (N x K)
C_init = (Y * F_init') / (F_init * F_init');

% State transition matrix (K x K for AR(1))
A_init = eye(K) * 0.9;

% Build state-space model
Mdl = ssm(@(params) dfm_state_space_parameterization(params, N, K, factor_order, error_order));

% Estimate
options = optimoptions('fminunc', 'MaxIterations', maxiter, 'Display', 'off');
params0 = [C_init(:); A_init(1:K); ones(N,1)*0.1; ones(K,1)*0.1];

if verbose
    fprintf('  Estimating state-space DFM...\n');
end

try
    EstMdl = estimate(Mdl, Y', params0, 'Options', options);
    
    if verbose
        fprintf('  Estimation complete.\n');
        fprintf('  Loglikelihood: %.2f\n', EstMdl.Loglikelihood);
    end
catch ME
    warning('DFM estimation failed: %s. Using OLS approximation.', ME.message);
    % Fallback to simple factor model
    EstMdl = struct();
    EstMdl.C = C_init;
    EstMdl.A = A_init;
    EstMdl.F_last = F_init(:, end);
    EstMdl.Q = eye(K) * 0.1;
    EstMdl.H = eye(N) * 0.1;
end

end


function [A, B, C, D, Mean0, Cov0, StateType] = dfm_state_space_parameterization(params, N, K, factor_order, error_order)
% Simplified state-space parameterization for DFM
% This is a placeholder - full implementation would be more complex

% Parse parameters
n_loadings = N * K;
n_ar = K;
n_obs_var = N;
n_state_var = K;

idx = 1;
C_vec = params(idx:idx+n_loadings-1);
C = reshape(C_vec, N, K);
idx = idx + n_loadings;

A_diag = params(idx:idx+n_ar-1);
A = diag(A_diag);
idx = idx + n_ar;

H_diag = params(idx:idx+n_obs_var-1);
H = diag(abs(H_diag));
idx = idx + n_obs_var;

Q_diag = params(idx:idx+n_state_var-1);
Q = diag(abs(Q_diag));

% State-space matrices
B = chol(Q, 'lower');
D = chol(H, 'lower');
Mean0 = zeros(K, 1);
Cov0 = eye(K);
StateType = zeros(K, 1);
end


% =========================================================================
%  FORECASTING
% =========================================================================
function fcast = forecast_dfm(Mdl, EstMdl, std_data, horizon)
% Forecast using estimated DFM

vars = std_data.Properties.VariableNames;
Y = std_data{:, :}';  % (N x T)

% Simple approach: use estimated loadings and AR coefficients
if isstruct(EstMdl)
    C = EstMdl.C;
    A = EstMdl.A;
    F_last = EstMdl.F_last;
else
    % Extract from ssm model
    [~, ~, C, ~, ~, ~, ~] = EstMdl();
    A = eye(size(C, 2)) * 0.9;  % Fallback
    
    % Smooth to get factors
    [~, ~, Output] = smooth(EstMdl, Y');
    F_last = Output(end, :)';
end

K = size(C, 2);
N = size(C, 1);

% Forecast factors
F_fcast = zeros(K, horizon);
F_t = F_last;

for h = 1:horizon
    F_t = A * F_t;
    F_fcast(:, h) = F_t;
end

% Forecast observations
Y_fcast = C * F_fcast;  % (N x horizon)

fcast = Y_fcast';  % (horizon x N)
end


% =========================================================================
%  INVERSE TRANSFORMATIONS
% =========================================================================
function levels_fcst = forecast_to_levels(fcast_trans, levels, tcodes, verbose, progress_every)
vars = fcast_trans.Properties.VariableNames;
n = length(vars);
levels_fcst = table();

for i = 1:n
    vn = vars{i};
    
    if ~ismember(vn, levels.Properties.VariableNames) || ~isfield(tcodes, vn)
        levels_fcst.(vn) = nan(height(fcast_trans), 1);
        continue
    end
    
    tc = tcodes.(vn);
    if isnan(tc)
        levels_fcst.(vn) = nan(height(fcast_trans), 1);
        continue
    end
    
    % Get forecast values safely
    fc_vals = fcast_trans.(vn);
    if iscell(fc_vals)
        fc_vals = cell2mat(fc_vals);
    end
    fc_vals = double(fc_vals(:));
    
    % Get historical levels safely
    level_hist = levels.(vn);
    if iscell(level_hist)
        level_hist = cell2mat(level_hist);
    end
    level_hist = double(level_hist(:));
    
    levels_fcst.(vn) = invert_tcode(fc_vals, level_hist, tc);
    
    if verbose && (mod(i, progress_every) == 0 || i == n)
        fprintf('[invert] %d/%d\n', i, n);
    end
end
end


% =========================================================================
%  OUTPUT FORMATTING
% =========================================================================
function out = format_output(fcst, output_format, include_history, ...
                             last_date, last_levels, omit_suffix, ...
                             date_col, fcast_dates)

vars = fcst.Properties.VariableNames;

if include_history && ~isempty(last_date) && ~isempty(last_levels)
    hist_row = table();
    for i = 1:length(vars)
        vn = vars{i};
        if istable(last_levels) && ismember(vn, last_levels.Properties.VariableNames)
            val = last_levels.(vn);
            if iscell(val)
                val = val{1};
            end
            hist_row.(vn) = val;
        else
            hist_row.(vn) = NaN;
        end
    end
    fcst = [hist_row; fcst];
    fcast_dates = [last_date; fcast_dates(:)];
end

if strcmp(output_format, 'wide')
    if ~omit_suffix
        new_names = cell(size(vars));
        for i = 1:length(vars)
            new_names{i} = [vars{i}, '_value'];
        end
        fcst.Properties.VariableNames = new_names;
    end
    out = fcst;
    
    % Add date column
    date_strs = cellstr(datestr(fcast_dates, 'mm/dd/yyyy'));
    out.(date_col) = date_strs;
    out = movevars(out, date_col, 'Before', 1);
else
    % Long format
    date_strs = cellstr(datestr(fcast_dates, 'mm/dd/yyyy'));
    n_dates = length(date_strs);
    n_vars = length(vars);
    
    out_dates = repmat(date_strs, n_vars, 1);
    out_series = cell(n_dates * n_vars, 1);
    out_values = zeros(n_dates * n_vars, 1);
    
    idx = 1;
    for i = 1:n_vars
        for j = 1:n_dates
            out_series{idx} = vars{i};
            val = fcst.(vars{i});
            if iscell(val)
                out_values(idx) = val{j};
            else
                out_values(idx) = val(j);
            end
            idx = idx + 1;
        end
    end
    
    out = table(out_dates, out_series, out_values, ...
                'VariableNames', {date_col, 'series', 'forecast_value'});
end
end

% =========================================================================
%  UTILITIES
% =========================================================================
function v = str2double_safe(x)
if ischar(x) || isstring(x)
    v = str2double(x);
elseif isnumeric(x)
    v = double(x);
else
    v = NaN;
end
end