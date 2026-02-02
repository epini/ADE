% compare_z0_clean.m
% z0 comparison: ADE (corrected) vs Alerstam vs MC (RWT)
% - Uses g = linspace(-0.95,0.95,191) like the previous script
% - Plots anisotropic + isotropic (mux=muy=muz=1) with shaded ±2σ bands for MC
%
% Requires:
%   BC_ADE.m
%   BC_ADE_Alerstam.m
%   z0_RWT.m

clear; close all; clc;

%% --- Parameters (anisotropic case) ---
mux = 1;
muy = 2;
muz = 4;

lx0 = 1/mux;
ly0 = 1/muy;
lz0 = 1/muz;

% Isotropic case
mux_iso = 1; muy_iso = 1; muz_iso = 1;
lx0_iso = 1/mux_iso;
ly0_iso = 1/muy_iso;
lz0_iso = 1/muz_iso;

n_in  = 1;
n_ext = 1;

% Same g-grid as your other script
g = linspace(-0.95, 0.95, 191);
Nx = numel(g);

%% --- Theory settings ---
Lmax   = 15;
reltol = 1e-3;
abstol = 1e-3;

%% --- MC (RWT) settings ---
Nwalkers_batch   = 5e6;   % photons per batch
Nbatches_max     = 10;    % batches per g (increase if you want smaller SE)
NstepsBase       = 1e2;   % as requested (single case)
adaptive_default = false;

% Reproducibility (optional)
rng(1);

%% --- Storage: anisotropic ---
z0_ADE      = nan(1, Nx);
z0_Al       = nan(1, Nx);
z0_MC_mean  = nan(1, Nx);
z0_MC_se    = nan(1, Nx);

%% --- Storage: isotropic ---
z0_ADE_iso      = nan(1, Nx);
z0_Al_iso       = nan(1, Nx);
z0_MC_mean_iso  = nan(1, Nx);
z0_MC_se_iso    = nan(1, Nx);

%% --- Progress bar (ETA) ---
h = waitbar(0, 'Starting...', 'Name', 'z0: MC vs ADE (progress)');
cleanupObj = onCleanup(@() safeCloseWaitbar(h)); %#ok<NASGU>

t0 = tic;
perIter = zeros(1, Nx);

for ii = 1:Nx
    ti = tic;
    gi = g(ii);

    % Scale mean free paths by (1-g) (same as your diffusion script)
    lex = lx0*(1-gi);  ley = ly0*(1-gi);  lez = lz0*(1-gi);
    lex_i = lx0_iso*(1-gi); ley_i = ly0_iso*(1-gi); lez_i = lz0_iso*(1-gi);

    % -----------------------------
    % Theory (anisotropic)
    % -----------------------------
    try
        [~, z0corr] = BC_ADE(n_in, n_ext, lex, ley, lez, gi, Lmax, reltol, abstol);
    catch ME
        warning('BC_ADE failed at g=%.3f: %s', gi, ME.message);
        z0corr = NaN;
    end
    z0_ADE(ii) = z0corr;

    try
        [~, z0al] = BC_ADE_Alerstam(n_in, n_ext, lex, ley, lez, gi, reltol, abstol);
    catch ME
        warning('BC_ADE_Alerstam failed at g=%.3f: %s', gi, ME.message);
        z0al = NaN;
    end
    z0_Al(ii) = z0al;

    % -----------------------------
    % Theory (isotropic)
    % -----------------------------
    try
        [~, z0corr_i] = BC_ADE(n_in, n_ext, lex_i, ley_i, lez_i, gi, Lmax, reltol, abstol);
    catch ME
        warning('BC_ADE (iso) failed at g=%.3f: %s', gi, ME.message);
        z0corr_i = NaN;
    end
    z0_ADE_iso(ii) = z0corr_i;

    try
        [~, z0al_i] = BC_ADE_Alerstam(n_in, n_ext, lex_i, ley_i, lez_i, gi, reltol, abstol);
    catch ME
        warning('BC_ADE_Alerstam (iso) failed at g=%.3f: %s', gi, ME.message);
        z0al_i = NaN;
    end
    z0_Al_iso(ii) = z0al_i;

    % -----------------------------
    % MC (anisotropic): streaming over batch means (Welford)
    % -----------------------------
    [mcMean, mcSE] = runBatchedZ0MC(Nwalkers_batch, Nbatches_max, gi, lex, ley, lez, ...
        NstepsBase, adaptive_default);

    z0_MC_mean(ii) = mcMean;
    z0_MC_se(ii)   = mcSE;

    % -----------------------------
    % MC (isotropic)
    % -----------------------------
    [mcMean_i, mcSE_i] = runBatchedZ0MC(Nwalkers_batch, Nbatches_max, gi, lex_i, ley_i, lez_i, ...
        NstepsBase, adaptive_default);

    z0_MC_mean_iso(ii) = mcMean_i;
    z0_MC_se_iso(ii)   = mcSE_i;

    % -----------------------------
    % Progress
    % -----------------------------
    perIter(ii) = toc(ti);
    k0 = max(1, ii-4);
    avgIter = median(perIter(k0:ii));
    remaining = avgIter * (Nx - ii);

    msg = sprintf([ ...
        'g index: %d/%d (g=%+.3f)\n' ...
        'Elapsed: %s\n' ...
        'ETA:     %s\n' ...
        'Avg/pt:  %s'], ...
        ii, Nx, gi, fmtTime(toc(t0)), fmtTime(remaining), fmtTime(avgIter));
    waitbar(ii/Nx, h, msg);

    fprintf('Done %d/%d (g=%+.3f)\n', ii, Nx, gi);
end

%% --- Plot (normalized like your previous script) ---
% Use constant simplistic reference (same style as Dx_simpl=(v/3)*lex in prior script)
z0_simpl     = lz0 * ones(size(g));      % anisotropic baseline
z0_simpl_iso = lz0_iso * ones(size(g));  % isotropic baseline

% Colors / styles matching your earlier conventions
mcGray   = 0.35;                    % MC lines gray
bandA    = 0.15;                    % anisotropic band alpha
bandIsoA = 0.08;                    % isotropic band alpha (lighter)
purple   = [0.75 0.45 0.85];             % ADE corrected = purple
lightBlue = [0.3 0.7 1.0];          % Alerstam = light blue
isoGray  = [0.6 0.6 0.6];           % isotropic ADE = light gray

figure(1); clf;
hold on; box on; grid on;

% --- Shaded bands first (±2σ), normalized ---
fill([g(:); flipud(g(:))], ...
     [z0_MC_mean(:)./z0_simpl(:) + 2*(z0_MC_se(:)./z0_simpl(:)); ...
      flipud(z0_MC_mean(:)./z0_simpl(:) - 2*(z0_MC_se(:)./z0_simpl(:)))], ...
     'k', 'FaceAlpha', bandA, 'EdgeColor', 'none');

fill([g(:); flipud(g(:))], ...
     [z0_MC_mean_iso(:)./z0_simpl_iso(:) + 2*(z0_MC_se_iso(:)./z0_simpl_iso(:)); ...
      flipud(z0_MC_mean_iso(:)./z0_simpl_iso(:) - 2*(z0_MC_se_iso(:)./z0_simpl_iso(:)))], ...
     'k', 'FaceAlpha', bandIsoA, 'EdgeColor', 'none');

% --- Curves on top ---
% Anisotropic theory
pADE  = plot(g, z0_ADE./z0_simpl, '-', 'Color', purple,    'LineWidth', 1.5);
pAl   = plot(g, z0_Al ./z0_simpl, '-', 'Color', lightBlue, 'LineWidth', 1.5);

% Anisotropic MC
pMC   = plot(g, z0_MC_mean./z0_simpl, '--', 'Color', [mcGray mcGray mcGray], 'LineWidth', 1.2);

% Isotropic theory + MC (same “method” as previous: iso analytic light gray, iso MC dotted black)
pADEi = plot(g, z0_ADE_iso./z0_simpl_iso, '-', 'Color', isoGray, 'LineWidth', 1.5);
pMCi  = plot(g, z0_MC_mean_iso./z0_simpl_iso, ':', 'Color', [0 0 0], 'LineWidth', 1.2);

% Ensure theory/MC are above bands
uistack(pADE,  'top');
uistack(pAl,   'top');
uistack(pMC,   'top');
uistack(pADEi, 'top');
uistack(pMCi,  'top');

xlim([-1 1]);
ylim([0.8 1.3]);

xlabel('$g$', 'Interpreter','latex', 'FontSize', 16);
ylabel('$z_0/z_0^{\mathrm{simpl}}$', 'Interpreter','latex', 'FontSize', 16);

lgd = legend([pADE pAl pMC pADEi pMCi], ...
    {'$z_0$ (ADE)', ...
     '$z_0$ (Alerstam)', ...
     '$z_0$ (MC sim.)', ...
     'Iso. (DE)', ...
     'Iso. MC sim.'}, ...
    'Location','northwest', ...
    'Interpreter','latex', ...
    'FontSize', 10);

lgd.Units = 'normalized';
ax = gca;
ax.Units = 'normalized';

% Force legend to sit exactly at top-right corner of axes
lgd.Position(1) = ax.Position(1);                          % left edge
lgd.Position(2) = ax.Position(2) + ax.Position(4) ...
                  - lgd.Position(4);                       % top edge

% Export
exportgraphics(gca, 'Fig4b.pdf', 'ContentType','vector');

%% --- Save results ---
save('z0_clean_results.mat', ...
    'g', ...
    'mux','muy','muz','lx0','ly0','lz0', ...
    'mux_iso','muy_iso','muz_iso','lx0_iso','ly0_iso','lz0_iso', ...
    'n_in','n_ext','Lmax','reltol','abstol', ...
    'Nwalkers_batch','Nbatches_max','NstepsBase','adaptive_default', ...
    'z0_ADE','z0_Al','z0_MC_mean','z0_MC_se', ...
    'z0_ADE_iso','z0_Al_iso','z0_MC_mean_iso','z0_MC_se_iso');

fprintf('Saved: z0_clean_results.mat\n');

%% ===== Local helper functions =====

function [mean_b, se_mean] = runBatchedZ0MC(Nwalkers_batch, Nbatches_max, g, lex, ley, lez, NstepsBase, adaptive_default)
% Runs z0_RWT in Nbatches_max batches and returns the mean over batch means
% plus standard error of that mean (between-batch variance).

    nb = 0;
    mean_b = 0;
    M2_b = 0;

    for b = 1:Nbatches_max
        [z0b_mean, ~, ~] = z0_RWT(Nwalkers_batch, g, lex, ley, lez, ...
            'NstepsBase', NstepsBase, ...
            'Adaptive', adaptive_default, ...
            'Verbose', false);

        nb = nb + 1;
        delta = z0b_mean - mean_b;
        mean_b = mean_b + delta/nb;
        M2_b   = M2_b + delta*(z0b_mean - mean_b);
    end

    if nb > 1
        var_between = M2_b / (nb - 1);
        se_mean = sqrt(var_between / nb);
    else
        se_mean = NaN;
    end
end

function s = fmtTime(tsec)
    if ~isfinite(tsec) || tsec < 0
        s = '--:--:--';
        return;
    end
    hh = floor(tsec/3600);
    mm = floor((tsec - 3600*hh)/60);
    ss = floor(tsec - 3600*hh - 60*mm);
    s = sprintf('%02d:%02d:%02d', hh, mm, ss);
end

function safeCloseWaitbar(h)
    if ~isempty(h) && isgraphics(h)
        close(h);
    end
end
