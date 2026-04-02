function run_all_forecasts()
% Run DFM forecasts for all horizons with dfm_XXm.csv naming

horizons = [1, 2, 4, 6, 12, 24, 60];
data_file = 'filled_2026-02-MD.csv';
k_factors = 3;
factor_order = 3;

fprintf('Starting DFM Forecasts\n');
fprintf('Data: %s\n', data_file);
fprintf('Factors: %d | Factor Order: %d\n', k_factors, factor_order);
fprintf('Horizons: %s\n\n', mat2str(horizons));

for h = horizons
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('HORIZON: %d months\n', h);
    fprintf('%s\n\n', repmat('=', 1, 60));
    
    output_file = sprintf('dfm_%dm.csv', h);
    
    try
        dfm_forecast_mle('data', data_file, ...
                         'all_series', true, ...
                         'k_factors', k_factors, ...
                         'factor_order', factor_order, ...
                         'horizon', h, ...
                         'output', output_file, ...
                         'verbose', true);
        
        fprintf('\n✓ SUCCESS: %s\n', output_file);
    catch ME
        fprintf('\n✗ ERROR: %s\n', ME.message);
        continue;
    end
end

fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('ALL FORECASTS COMPLETE\n');
fprintf('%s\n\n', repmat('=', 1, 60));

% List output files
fprintf('Output files:\n');
for h = horizons
    fname = sprintf('dfm_%dm.csv', h);
    if exist(fname, 'file')
        info = dir(fname);
        fprintf('  ✓ %s (%.1f KB, %d months)\n', fname, info.bytes/1024, h);
    else
        fprintf('  ✗ %s (MISSING)\n', fname);
    end
end

end