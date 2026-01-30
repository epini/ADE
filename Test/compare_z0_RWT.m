% compare_z0_RWT.m
% Confronto tra z0 calcolato da BC_ADE_corrected, BC_ADE_Alerstam e z0_RWT (RWT pencil-beam)
% Non esegue Monte Carlo esterno: chiama la funzione z0_RWT per ogni g.

clear
close all

% --- user parameters ---
lx = 20;
ly = 20;
lz = 20;
n_in = 1;
n_ext = 1;

g_vals = -0.9:0.1:0.9;
N = numel(g_vals);

% Preallocate (only z0 values now)
z0_corrected = zeros(1, N);
z0_Alerstam = zeros(1, N);

% RWT (Monte Carlo) settings — adjust as needed
Nwalkers_default = 1e5;    % number of photons
NstepsBase_default = 1e3; % base scattering events
adaptive_default = true;   % adapt Nsteps with g

z0_rwt = nan(1, N);
z0_rwt_se = nan(1, N);
z0_rwt_time = nan(1, N);

% Progress UI
h = waitbar(0, 'Starting...');
tStart = tic;

% Tolerances used by ADE functions (kept from your code)
Lmax = 7;
reltol = 1e-2;
abstol = 1e-2;

for i = 1:N
    g = g_vals(i);

    % effective mean-free-paths scaled by (1-g)
    lex = lx*(1-g);
    ley = ly*(1-g);
    lez = lz*(1-g);

    % --- compute only z0 variants from theory ---
    % BC_ADE_corrected returns [ze_corrected, z0_corrected], but we only keep z0
    try
        [~, z0_corr] = BC_ADE_corrected(n_in, n_ext, lex, ley, lez, g, Lmax, reltol, abstol);
    catch ME
        warning('BC_ADE_corrected failed at g=%.3f: %s', g, ME.message);
        z0_corr = NaN;
    end
    z0_corrected(i) = z0_corr;

    try
        [~, z0_al] = BC_ADE_Alerstam(n_in, n_ext, lex, ley, lez, g, reltol, abstol);
    catch ME
        warning('BC_ADE_Alerstam failed at g=%.3f: %s', g, ME.message);
        z0_al = NaN;
    end
    z0_Alerstam(i) = z0_al;

    % ---- call your RWT function z0_RWT (pencil-beam) ----
    try
        t0 = tic;
        % assumed signature:
        % [z0_mean, z0_se, stats] = z0_RWT(Nwalkers, g, lex, ley, lez, ...)
        [z0_mean, z0_se, stats] = z0_RWT(Nwalkers_default, g, lex, ley, lez, ...
            'NstepsBase', NstepsBase_default, 'Adaptive', adaptive_default, 'Verbose', false);
        t_elapsed = toc(t0);

        z0_rwt(i) = z0_mean;
        z0_rwt_se(i) = z0_se;
        z0_rwt_time(i) = t_elapsed;
    catch ME
        warning('z0_RWT failed at g=%.3f: %s', g, ME.message);
        z0_rwt(i) = NaN;
        z0_rwt_se(i) = NaN;
        z0_rwt_time(i) = NaN;
    end

    % ---- Time estimation & progressbar ----
    elapsed = toc(tStart);
    avgTime = elapsed / i;
    remaining = avgTime * (N - i);
    etaStr = datestr(seconds(remaining), 'HH:MM:SS');

    waitbar(i/N, h, sprintf('Progress: %d/%d | ETA: %s', i, N, etaStr));
end

close(h)

%% --- Single plot of z0 vs g ---
figure(1); clf; hold on; grid on; box on;
h1 = plot(g_vals, z0_corrected, 'b-', 'LineWidth', 1.5);          % ADE corrected
h2 = plot(g_vals, z0_Alerstam, 'g--', 'LineWidth', 1.5);          % Alerstam
% plot RWT with errorbars if available
valid_idx = ~isnan(z0_rwt);
if any(valid_idx)
    h3 = errorbar(g_vals(valid_idx), z0_rwt(valid_idx), z0_rwt_se(valid_idx), 'ro', ...
        'MarkerFaceColor', 'r', 'LineWidth', 1.2, 'MarkerSize', 5);
else
    h3 = plot(g_vals, z0_rwt, 'ro', 'MarkerFaceColor', 'r');
end

legend([h1 h2 h3], '$z_0$ (ADE corrected)', '$z_0$ (Alerstam)', '$z_0$ (RWT)', ...
       'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 12);

xlabel('$g$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$z_0$ (same units as $\ell$)', 'Interpreter', 'latex', 'FontSize', 14);
title(sprintf('$\\ell_x=%d(1-g),\\,\\ell_y=%d(1-g),\\,\\ell_z=%d(1-g)$', lx, ly, lz), ...
      'Interpreter', 'latex', 'FontSize', 12);

% tidy
xlim([-1 1]);
drawnow;

%% --- Print comparison table ---
fprintf('\nComparison table for z0 (units same as lx,ly,lz):\n');
fprintf(' g    z0_corr    z0_Alerstam    z0_RWT     se_RWT    diff(RWT - corr)\n');
for k = 1:N
    diff_rc = NaN;
    if ~isnan(z0_rwt(k)) && ~isnan(z0_corrected(k))
        diff_rc = z0_rwt(k) - z0_corrected(k);
    end
    fprintf('%5.3f  %8.4f   %10.4f    %8.4f   %7.4f   %10.4f\n', ...
        g_vals(k), z0_corrected(k), z0_Alerstam(k), z0_rwt(k), z0_rwt_se(k), diff_rc);
end

%% --- Save results ---
save('z0_comparison_results.mat', 'g_vals', 'z0_corrected', 'z0_Alerstam', 'z0_rwt', 'z0_rwt_se', 'z0_rwt_time');
fprintf('\nSaved results to z0_comparison_results.mat\n');

