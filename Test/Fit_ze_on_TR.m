%% Batch fit ze(g) from TOTAL time-resolved transmission (no spatial grid)
% Corrected for Tt_ADE_zefit units: L,lx,ly,lz,ze in [um], t in [ps], mua in [1/um]
clear; close all; clc;

%% Paths (edit)
addpath('C:\Users\ernes\Desktop\PostDoc\Scripts MATLAB')
addpath('C:\Users\ernes\Desktop\PostDoc\MC\Anisotropy\Anisotropic Diffusion Equation\Functions\Generalized ADE')
addpath('C:\Users\ernes\Desktop\PostDoc\MC\Anisotropy\Anisotropic Diffusion Equation\Functions\Generalized ADE\Test')
addpath('C:\Users\ernes\Desktop\PostDoc\MC\Anisotropy\Anisotropic Diffusion Equation\Functions\Generalized ADE\Old versions')
addpath('C:\Users\ernes\Desktop\PostDoc\MC\Anisotropy\Giusfredi formulas test\mc_totalTR_g_sweep')

%% Folder with MAT files produced by Python (current folder)
files = dir(fullfile('TSRT_totalTR_scaledparams_g_*.mat'));
assert(~isempty(files), 'No MC files found in current folder.');

%% Sort files by numeric g (read from inside each mat) so it starts at -0.9
g_file = nan(numel(files),1);
for i = 1:numel(files)
    tmp = load(fullfile(files(i).folder, files(i).name), 'g');
    g_file(i) = double(tmp.g);
end
[~, ord] = sort(g_file, 'ascend');
files = files(ord);

%% Fit settings
n_ext = 1.0;

% Fit window in ps (set to ignore early/late times if needed)
tmin_ps = 300;
tmax_ps = 9000;

useLogFit = true;   % fit in log space (recommended)
eps0 = 1e-30;

% Units for the fit (must match Tt_ADE_zefit):
%   L,lx,ly,lz,ze in [um], t in [ps], mua in [1/um]
zeLB = 0;        % [um]
zeUB = 50e3;     % 50 mm = 50,000 um

%% Storage
Ng = numel(files);
g_all    = nan(Ng,1);
ze_fit   = nan(Ng,1);   % [um]
ze_se    = nan(Ng,1);   % [um]
ze_ADE   = nan(Ng,1);   % [um]
fval_all = nan(Ng,1);
exitflag = nan(Ng,1);

curves = struct('t_ps',[],'mc',[],'fit',[],'g',[]);

%% ---------------- Waitbar with ETA ----------------
tStart = tic;
wb = waitbar(0, 'Starting...', 'Name', 'Fitting ze(g)...', ...
    'CreateCancelBtn', 'setappdata(gcbf,''canceling'',1)');
setappdata(wb, 'canceling', 0);

sec2hms = @(s) sprintf('%02d:%02d:%02d', floor(s/3600), floor(mod(s,3600)/60), round(mod(s,60)));

%% Main loop
for k = 1:Ng
    % Cancel support
    if ishandle(wb) && getappdata(wb,'canceling')
        disp('Canceled by user.');
        break;
    end

    fname = fullfile(files(k).folder, files(k).name);
    S = load(fname);

    % --- Required fields ---
    assert(isfield(S,'t') && isfield(S,'T'), 'Missing t or T in %s', files(k).name);
    assert(isfield(S,'g') && isfield(S,'L') && isfield(S,'lx') && isfield(S,'ly') && isfield(S,'lz'), ...
        'Missing params in %s', files(k).name);
    assert(isfield(S,'mua') && isfield(S,'n_sample'), 'Missing mua/n_sample in %s', files(k).name);

    % --- Read params (Python outputs in SI) ---
    g_SI    = double(S.g);
    L_SI    = double(S.L);        % [m]
    lx_SI   = double(S.lx);       % [m]
    ly_SI   = double(S.ly);       % [m]
    lz_SI   = double(S.lz);       % [m]
    mua_SI  = double(S.mua);      % [1/m]
    n_in    = double(S.n_sample);

    % Fix -0.0 printing
    if abs(g_SI) < 5e-13, g_SI = 0; end

    % --- Convert to units expected by Tt_ADE_zefit ---
    g   = g_SI;
    L   = L_SI  * 1e6;    % [um]
    lx  = lx_SI * 1e6;    % [um]
    ly  = ly_SI * 1e6;    % [um]
    lz  = lz_SI * 1e6;    % [um]
    mua = mua_SI * 1e-6;  % [1/um]

    % --- Time axis and MC curve ---
    t_ps = double(S.t(:)) * 1e12;   % [ps]
    MC   = double(S.T(:));          % total transmittance vs time bin (as saved)

    % --- Mask / fit window ---
    mask = isfinite(t_ps) & isfinite(MC) & (t_ps >= tmin_ps) & (t_ps <= tmax_ps);
    t_fit = t_ps(mask);
    y_fit = MC(mask);

    if numel(t_fit) < 10
        warning('Too few points after masking for %s, skipping.', files(k).name);
        continue;
    end

    dt_ps = mean(diff(t_fit));

    % --- ADE prediction for ze (used as initial guess) ---
    [ze_ADE(k), ~] = BC_ADE_corrected(n_in, n_ext, lx, ly, lz, g);   % [um]

    % --- Objective function over ze (ze in [um]) ---
    if useLogFit
        obj = @(ze) objective_log(ze, t_fit, y_fit, dt_ps, L, n_in, n_ext, lx, ly, lz, mua, g, eps0);
    else
        obj = @(ze) objective_lin(ze, t_fit, y_fit, dt_ps, L, n_in, n_ext, lx, ly, lz, mua, g);
    end

    % --- Fit ze using ze_ADE(k) as starting point (bounded via transform) ---
    % --- Fit ze using fminbnd with a tight bracket around ze_ADE ---
    ze0 = min(max(ze_ADE(k), zeLB), zeUB);

    % Define a local search interval around the ADE estimate
    br_lo = max(zeLB, 0.2 * ze0);
    br_hi = min(zeUB, 5.0 * ze0);

    % Fallback if bracket collapses
    if br_hi <= br_lo
        br_lo = zeLB;
        br_hi = zeUB;
    end

    opts = optimset('Display','off', ...
        'TolX',1e-8, ...
        'TolFun',1e-10, ...
        'MaxIter',200);

    [ze_best, fval, ef] = fminbnd(obj, br_lo, br_hi, opts);

    % --- Estimate uncertainty from curvature (deltaSSE = 1 heuristic) ---
    ze_sigma = curvature_se(obj, ze_best, zeLB, zeUB);

    % --- Store results ---
    g_all(k)    = g;
    ze_fit(k)   = ze_best;
    ze_se(k)    = ze_sigma;
    fval_all(k) = fval;
    exitflag(k) = ef;

    % --- Compute best-fit theory curve for plotting ---
    % NOTE: keep *dt_ps only if your MC curve is per-bin. If MC is a density, remove it.
    T_theory = Tt_ADE_zefit(t_fit, L, n_in, n_ext, lx, ly, lz, mua, g, ze_best) * dt_ps;

    curves(k).t_ps = t_fit;
    curves(k).mc   = y_fit;
    curves(k).fit  = T_theory;
    curves(k).g    = g;

    fprintf('g=%+.1f | ze_fit=%.6g um | se~%.3g um | fval=%.3g | exit=%d | ze_ADE=%.6g um\n', ...
        g, ze_best, ze_sigma, fval, ef, ze_ADE(k));

    % --- Update waitbar + ETA ---
    if ishandle(wb)
        frac = k / Ng;
        elapsed = toc(tStart);
        avgPer = elapsed / max(k,1);
        eta = avgPer * (Ng - k);

        msg = sprintf(['File %d/%d | g=%+.1f\n' ...
            'Elapsed: %s | ETA: %s\n' ...
            '%s'], ...
            k, Ng, g, sec2hms(elapsed), sec2hms(eta), files(k).name);

        waitbar(frac, wb, msg);
        drawnow;
    end
end

% Close waitbar
if exist('wb','var') && ishandle(wb)
    delete(wb);
end

%% Sort by g
[gs, idx] = sort(g_all);
ze_fit_s  = ze_fit(idx);
ze_se_s   = ze_se(idx);
ze_ADE_s  = ze_ADE(idx);
curves    = curves(idx);

valid = isfinite(gs) & isfinite(ze_fit_s);
gs = gs(valid);
ze_fit_s = ze_fit_s(valid);
ze_se_s  = ze_se_s(valid);
ze_ADE_s = ze_ADE_s(valid);
curves   = curves(valid);

Ngv = numel(gs);

%% Plot 1: MC vs best-fit curves for each g
ncols = 4;
nrows = ceil(Ngv/ncols);

figure('Name','TOTAL TR: MC vs ADE best-fit (ze per g)','Color','w');
for i = 1:Ngv
    subplot(nrows,ncols,i); hold on; box on; grid on;
    t_ps_i = curves(i).t_ps;
    mc1    = curves(i).mc;
    th1    = curves(i).fit;

    plot(t_ps_i, th1, 'LineWidth', 1.2);
    plot(t_ps_i, mc1, ':', 'LineWidth', 1.2);

    set(gca,'YScale','log');
    xlabel('t [ps]');
    ylabel('T(t) [a.u.]');
    title(sprintf('g=%+.1f | z_e=%.3g um', gs(i), ze_fit_s(i)));
    ylim([1e-8 1e-2])
end
legend({'ADE best-fit','MC'}, 'Location','best');

%% Plot 2: ze(g) with error bars + ADE prediction
figure('Name','Fitted z_e vs g','Color','w'); hold on; box on; grid on;
hasErr = any(isfinite(ze_se_s) & ze_se_s>0);
if hasErr
    errorbar(gs, ze_fit_s, ze_se_s, 'o-', 'LineWidth',1.2, 'MarkerSize',5);
else
    plot(gs, ze_fit_s, 'o-', 'LineWidth',1.2, 'MarkerSize',5);
end
plot(gs, ze_ADE_s, 'x-', 'LineWidth',1.2, 'MarkerSize',5);

xlabel('$g$', 'Interpreter', 'latex');
ylabel('$z_e$ [$\mu$m]', 'Interpreter', 'latex');
legend('$z_e$ MC fit', '$z_e$ ADE', 'Interpreter', 'latex', 'fontsize', 14);

%% ---------------- Helper functions ----------------
function sse = objective_lin(ze, t_ps, y_mc, dt_ps, L, n_in, n_ext, lx, ly, lz, mua, g)
y_th = Tt_ADE_zefit(t_ps, L, n_in, n_ext, lx, ly, lz, mua, g, ze) * dt_ps;
r = (y_th(:) - y_mc(:));
sse = sum(r.^2);
end

function sse = objective_log(ze, t_ps, y_mc, dt_ps, L, n_in, n_ext, lx, ly, lz, mua, g, eps0)
y_th = Tt_ADE_zefit(t_ps, L, n_in, n_ext, lx, ly, lz, mua, g, ze) * dt_ps;
r = log10(y_th(:) + eps0) - log10(y_mc(:) + eps0);
sse = sum(r.^2);
end

function se = curvature_se(obj, x0, lb, ub)
h = max(1e-6*max(1,abs(x0)), 1e-6); % in [um], keep step not too tiny
x1 = max(lb, min(ub, x0 - h));
x2 = x0;
x3 = max(lb, min(ub, x0 + h));

if (x1==x2) || (x3==x2)
    se = NaN; return;
end

f1 = obj(x1);
f2 = obj(x2);
f3 = obj(x3);

denom = (x3 - x2)^2;
fpp = (f1 - 2*f2 + f3) / denom;

if ~isfinite(fpp) || fpp <= 0
    se = NaN;
else
    se = sqrt(2 / fpp);
end
end
