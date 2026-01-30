clear
% close all

lx = 40;
ly = 20;
lz = 10;
n_in = 1.2;
n_ext = 1;

g_vals = -0.9:0.1:0.9;
N = numel(g_vals);

% Preallocate
ze = zeros(1, N);
z0 = zeros(1, N);
ze_corrected = zeros(1, N);
z0_corrected = zeros(1, N);
ze_Alerstam = zeros(1, N);
z0_Alerstam = zeros(1, N);

h = waitbar(0, 'Starting...');
tStart = tic;

Lmax = 7;
reltol = 1e-2;
abstol = 1e-2;

for i = 1:N
    g = g_vals(i);

    lex = lx*(1-g);
    ley = ly*(1-g);
    lez = lz*(1-g);

    [ze(i), z0(i)] = BC_ADE(n_in, n_ext, lex, ley, lez, g, Lmax, reltol, abstol);
    [ze_corrected(i), z0_corrected(i)] = BC_ADE_corrected(n_in, n_ext, lex, ley, lez, g, Lmax, reltol, abstol);
    [ze_Alerstam(i), z0_Alerstam(i)] = BC_ADE_Alerstam(n_in, n_ext, lex, ley, lez, g, reltol, abstol);

    % ---- Time estimation ----
    elapsed = toc(tStart);
    avgTime = elapsed / i;
    remaining = avgTime * (N - i);

    % Format remaining time nicely
    etaStr = datestr(seconds(remaining), 'HH:MM:SS');

    waitbar(i/N, h, ...
        sprintf('Progress: %d/%d | ETA: %s', i, N, etaStr));
end

close(h)

%% Load MC results and overlay
% Looks for global files first, otherwise per-g 'tr_fit_g_*.mat'
mc_found = false;
mc_g = [];
mc_z0_um = [];

% candidate global filenames
candidates = {'z0_vs_g_results_reverted.mat', 'z0_vs_g_results.mat', 'z0_vs_g.mat'};
for k = 1:numel(candidates)
    fn = candidates{k};
    if exist(fn, 'file')
        S = load(fn);
        if isfield(S, 'g') && isfield(S, 'z0_um')
            mc_g = double(S.g(:))';
            mc_z0_um = double(S.z0_um(:))';
            mc_found = true;
            fprintf('Loaded global MC file: %s\n', fn);
            break;
        end
    end
end

% if no global file, try per-g files tr_fit_g_*.mat
if ~mc_found
    files = dir('tr_fit_g_*.mat');
    if ~isempty(files)
        gg = [];
        z0m = [];
        for k = 1:numel(files)
            fn = files(k).name;
            S = load(fn);
            gz = NaN;
            zfit_um = NaN;
            % try meta.g
            if isfield(S, 'meta') && isfield(S.meta, 'g')
                gz = double(S.meta.g);
            else
                % try parse from filename e.g. tr_fit_g_0.10.mat
                tok = regexp(fn, 'tr_fit_g_([0-9]+\.[0-9]+)\.mat', 'tokens');
                if ~isempty(tok)
                    gz = str2double(tok{1}{1});
                else
                    % alternative pattern with minus sign
                    tok2 = regexp(fn, 'tr_fit_g_(-?[0-9]+\.[0-9]+)\.mat', 'tokens');
                    if ~isempty(tok2)
                        gz = str2double(tok2{1}{1});
                    end
                end
            end

            % try z_fit_m or meta.z_fit_um
            if isfield(S, 'z_fit_m')
                zfit_um = double(S.z_fit_m) * 1e6;
            elseif isfield(S, 'meta') && isfield(S.meta, 'z_fit_um')
                zfit_um = double(S.meta.z_fit_um);
            elseif isfield(S, 'meta') && isfield(S.meta, 'z_fit_m')
                zfit_um = double(S.meta.z_fit_m) * 1e6;
            end

            if ~isnan(gz) && ~isnan(zfit_um)
                gg(end+1) = gz; %#ok<SAGROW>
                z0m(end+1) = zfit_um; %#ok<SAGROW>
            else
                fprintf('Warning: skipping %s — missing g or z_fit\n', fn);
            end
        end
        if ~isempty(gg)
            [ggs, idx] = sort(gg);
            mc_g = ggs;
            mc_z0_um = z0m(idx);
            mc_found = true;
            fprintf('Loaded %d per-g MC files (tr_fit_g_*.mat)\n', numel(ggs));
        end
    end
end

if ~mc_found
    fprintf('No MC results found in working directory. Continuing with theory-only plots.\n');
end

%% plot

figure(1), clf; hold on, grid on, box on
plot(g_vals, ze, 'r', 'LineWidth', 1.5)
plot(g_vals, z0, 'b', 'LineWidth', 1.5)
plot(g_vals, ze_corrected, 'r--', 'LineWidth', 1.2)
plot(g_vals, z0_corrected, 'b--', 'LineWidth', 1.2)
plot(g_vals, ze_Alerstam, 'r:', 'LineWidth', 1.0)
plot(g_vals, z0_Alerstam, 'b:', 'LineWidth', 1.0)

% overlay MC points (if available)
if mc_found
    % For clarity use filled markers
    plot(mc_g, mc_z0_um, 'kp', 'MarkerFaceColor', 'y', 'MarkerSize', 10, 'LineWidth', 1.2)
    % also plot MC points only for z0 on separate series
    % create legend entries dynamically below
end

% build legend entries
leg = {'$z_e$ $\lambda(s)$', '$z_0$ $\lambda(s)$', '$z_e$ $\lambda(s)$ off. corr.', ...
       '$z_0$ $\lambda(s)$ off. corr.', '$z_e$ Alerstam', '$z_0$ Alerstam'};
if mc_found
    leg{end+1} = 'MC z_0 points';
end

legend(leg, 'interpreter', 'latex', 'fontsize', 10, 'Location', 'best')
xlabel('$g$', 'interpreter', 'latex', 'fontsize', 14)
title(sprintf('$\\ell_x=%d(1-g),\\, \\ell_y=%d(1-g),\\, \\ell_z=%d(1-g)$', lx, ly, lz), ...
      'interpreter', 'latex', 'fontsize', 12)

%% z0

figure(2), clf; hold on, grid on, box on
plot(g_vals, z0, 'b', 'LineWidth', 1.5)
plot(g_vals, z0_corrected, 'b--', 'LineWidth', 1.2)
plot(g_vals, z0_Alerstam, 'b:', 'LineWidth', 1.0)
plot(g_vals, lz*ones(1, length(g_vals)), 'k--', 'LineWidth', 1.0)

% overlay MC points (if available)
if mc_found
    % find points where mc_g overlaps display range and plot markers
    % If mc_g are different grid spacing, just plot them where they are
    plot(mc_g, mc_z0_um, 'or', 'MarkerSize', 5, 'LineWidth', 1.2)
end

legend('$z_0$ $\lambda(s)$','$z_0$ $\lambda(s)$ off. corr.','$z_0$ Alerstam','$z_0$ simplistic', '$z_0$ MC', 'interpreter', 'latex', 'fontsize', 12, 'Location', 'best')
xlabel('$g$', 'interpreter', 'latex', 'fontsize', 14)
title(sprintf('$\\ell_x=%d(1-g),\\, \\ell_y=%d(1-g),\\, \\ell_z=%d(1-g)$', lx, ly, lz), ...
      'interpreter', 'latex', 'fontsize', 12)

% annotate numeric table of MC vs theory (optional)
if mc_found
    fprintf('\nMC points loaded (g, z0 [um]):\n');
    for k = 1:length(mc_g)
        fprintf('  g = %6.3f, z0_MC = %8.3f um\n', mc_g(k), mc_z0_um(k));
    end

    % If you want to print a small table comparing theory z0 at MC g's:
    % interpolate theoretical z0 at mc_g positions
    z0_theory_interp = interp1(g_vals, z0, mc_g, 'linear', NaN);
    fprintf('\nComparison (g, z0_theory, z0_MC, diff [um]):\n');
    for k = 1:length(mc_g)
        fprintf('  %6.3f  %8.3f  %8.3f  %8.3f\n', mc_g(k), z0_theory_interp(k), mc_z0_um(k), mc_z0_um(k)-z0_theory_interp(k));
    end
end

% tidy up figures
drawnow;
