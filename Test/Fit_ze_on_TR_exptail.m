%% Estimate z_e from MC using late-time tail method (Option 2)
% Fits only t > 5 ns and excludes last time bin.
clear; close all; clc;

%% Paths (edit)
addpath('C:\Users\ernes\Desktop\PostDoc\Scripts MATLAB')
addpath('C:\Users\ernes\Desktop\PostDoc\MC\Anisotropy\Anisotropic Diffusion Equation\Functions\Generalized ADE')
addpath('C:\Users\ernes\Desktop\PostDoc\MC\Anisotropy\Anisotropic Diffusion Equation\Functions\Generalized ADE\Test')
addpath('C:\Users\ernes\Desktop\PostDoc\MC\Anisotropy\Giusfredi formulas test\mc_totalTR_g_sweep')

%% User options
files = dir('TSRT_totalTR_scaledparams_g_*.mat');
assert(~isempty(files), 'No MC files found in current folder.');

% Bootstrap settings
B = 300;   % number of bootstrap samples for uncertainty
rng(0);    % reproducible bootstrap

% Late-time cutoff
tmin_ns = 3;            % use only t > 5 ns
tmin_ps = tmin_ns * 1e3; % 5000 ps

% Small constant to avoid log(0)
eps_val = 1e-300;

% Storage
Ng = numel(files);
g_all    = nan(Ng,1);
ze_mc    = nan(Ng,1);   % [um]
ze_se    = nan(Ng,1);   % [um]
L_eff    = nan(Ng,1);
fit_ok   = false(Ng,1);

% Sort files by numeric g (robust)
g_file = nan(Ng,1);
for i=1:Ng
    tmp = load(fullfile(files(i).folder, files(i).name), 'g');
    g_file(i) = double(tmp.g);
end
[~,ord] = sort(g_file,'ascend');
files = files(ord);

%% Loop
for k = 1:numel(files)
    fname = fullfile(files(k).folder, files(k).name);
    S = load(fname);

    % Required fields
    if ~isfield(S,'t') || ~isfield(S,'T') || ~isfield(S,'L') || ~isfield(S,'lx')
        warning('Skipping %s: missing required fields.', files(k).name);
        continue;
    end

    % Read parameters from MAT (SI)
    t_s = double(S.t(:));    % seconds
    T   = double(S.T(:));    % total transmittance per bin (1D)
    L_SI = double(S.L);      % m
    lx_SI = double(S.lx);    % m
    ly_SI = double(S.ly);    % m
    lz_SI = double(S.lz);    % m
    mua_SI = double(S.mua);  % 1/m
    n_in = double(S.n_sample);
    if isfield(S,'nphotons')
        nphotons = double(S.nphotons);
    else
        nphotons = NaN;
    end
    g_val = double(S.g);
    if abs(g_val) < 5e-13, g_val = 0; end

    % Convert units for ADE function: lengths -> um, time -> ps, mua -> 1/um
    L  = L_SI  * 1e6;   % um
    lx = lx_SI * 1e6;
    ly = ly_SI * 1e6;
    lz = lz_SI * 1e6;
    mua = mua_SI * 1e-6; % 1/um

    t_ps = t_s * 1e12;  % ps

    % Exclude last time bin and select t>tmin_ps
    if numel(t_ps) < 5
        warning('Too few time bins in %s, skipping.', files(k).name);
        continue;
    end
    idx_all = 1:numel(t_ps);
    idx = idx_all(1:end-1);  % exclude last bin
    mask = (t_ps(idx) > tmin_ps);
    t_sel = t_ps(idx);
    T_sel = T(idx);
    t_sel = t_sel(mask);
    T_sel = T_sel(mask);

    if numel(t_sel) < 5
        warning('Not enough late-time bins after masking in %s, skipping.', files(k).name);
        continue;
    end

    % compute Dz and z0 from ADE helpers (they expect um units)
    [~, ~, Dz] = D_Tensor_ADE(n_in, lx, ly, lz, g_val);  % Dz in um^2/ps
    [~, z0] = BC_ADE(n_in, 1.0, lx, ly, lz, g_val);      % z0 in um; n_ext assumed 1

    % speed in um/ps
    v = 299.7924589 / n_in;   % um/ps

    % Correct for absorption: T_corr = T * exp(v*mua*t)
    T_corr = T_sel .* exp( v .* mua .* t_sel );

    % Build y = ln(T_corr) + 3/2 ln(t)
    y = log(T_corr + eps_val) + 1.5 * log(t_sel);

    % Linear fit y = m*(1/t) + c  -> use natural log
    x = 1 ./ t_sel;
    P = polyfit(x, y, 1);  % P(1)=m, P(2)=c
    m = P(1); c = P(2);

    % From slope -> L_eff^2 = -4 D m
    L_eff_sq = -4 * Dz * m;
    if ~(isfinite(L_eff_sq) && L_eff_sq > 0)
        warning('Non-positive L_eff^2 for %s (m=%g, Dz=%g). Skipping.', files(k).name, m, Dz);
        continue;
    end
    L_eff_k = sqrt(L_eff_sq); % um

    % Map to ze: using convention L_eff = L + 2*z_e - z0
    ze_k = (L_eff_k - L + z0) / 2;  % um

    % Save basic values
    g_all(k) = g_val;
    L_eff(k) = L_eff_k;
    ze_mc(k) = ze_k;

    % ---------------- Bootstrap for uncertainty ----------------
    ze_boot = nan(B,1);
    % If nphotons present and T_sel likely derived from counts, use Poisson parametric bootstrap
    use_poisson = ~isnan(nphotons) && nphotons>1 && all(T_sel>=0);

    if use_poisson
        % approximate counts per bin: counts = T_sel * nphotons
        counts = max(round(T_sel * nphotons), 0);
        % avoid zero counts all-around
        counts(counts<0) = 0;
        for b = 1:B
            % sample new counts per bin
            counts_b = poissrnd(counts);
            % normalized T
            T_b = counts_b ./ nphotons;
            % apply same processing: exclude zero bins when taking log (add eps)
            Tcorr_b = T_b .* exp(v .* mua .* t_sel);
            yb = log(Tcorr_b + eps_val) + 1.5 * log(t_sel);
            % robust linear fit: if too many zeros, skip
            if all(~isfinite(yb))
                ze_boot(b) = NaN;
                continue;
            end
            Pb = polyfit(x, yb, 1);
            mb = Pb(1);
            L_eff_sq_b = -4 * Dz * mb;
            if ~(isfinite(L_eff_sq_b) && L_eff_sq_b>0)
                ze_boot(b) = NaN;
                continue;
            end
            L_eff_b = sqrt(L_eff_sq_b);
            ze_boot(b) = (L_eff_b - L + z0) / 2;
        end
    else
        % Residual bootstrap: fit once, get residuals on y, resample residuals
        yhat = polyval(P, x);
        resid = y - yhat;
        for b = 1:B
            rstar = resid(randi(numel(resid), numel(resid), 1));
            yb = yhat + rstar;
            Pb = polyfit(x, yb, 1);
            mb = Pb(1);
            L_eff_sq_b = -4 * Dz * mb;
            if ~(isfinite(L_eff_sq_b) && L_eff_sq_b>0)
                ze_boot(b) = NaN; continue;
            end
            L_eff_b = sqrt(L_eff_sq_b);
            ze_boot(b) = (L_eff_b - L + z0) / 2;
        end
    end

    % summarize bootstrap
    ze_boot = ze_boot(isfinite(ze_boot));
    if isempty(ze_boot)
        ze_se(k) = NaN;
    else
        ze_se(k) = std(ze_boot, 1); % 1-sigma
    end

    fit_ok(k) = true;

    % --- Plot overlay of MC and asymptotic fit for inspection ---
    figure(100); clf; hold on; box on; grid on;
    plot(t_sel*1e-3, T_sel, 'k.-', 'DisplayName','MC (selected)'); % t in ns
    % reconstruct model T_model from fitted y: T_model = exp(y_fit - 1.5 ln t) * exp(-v*mua*t)
    y_fit = polyval(P, x);        % fitted y on 1/t grid
    T_model = exp(y_fit) .* exp(-v .* mua .* t_sel) .* (t_sel.^(-1.5));
    plot(t_sel*1e-3, T_model, 'r-', 'LineWidth',1.5, 'DisplayName','asymptotic fit');
    xlabel('t (ns)'); ylabel('T(t) [a.u.]');
    title(sprintf('g=%+.2f  -> z_e = %.3g um ± %.3g um', g_val, ze_k, ze_se(k)));
    legend('Location','best');
    set(gca,'YScale','log');
    xlim([min(t_sel)*1e-3, max(t_sel)*1e-3]);
    drawnow;

    fprintf('Processed %s | g=%+.2f | ze=%.4g um | se=%.3g um | L_eff=%.3g um\n', ...
        files(k).name, g_val, ze_k, ze_se(k), L_eff_k);
end

%% Collect and plot ze(g)
valid_idx = find(fit_ok);
if isempty(valid_idx)
    error('No successful fits.');
end

g_plot = g_all(valid_idx);
ze_plot = ze_mc(valid_idx);
ze_err  = ze_se(valid_idx);

[gs, idxs] = sort(g_plot);
ze_sorted = ze_plot(idxs);
ze_err_sorted = ze_err(idxs);

figure('Name','z_e (MC late-time) vs g','Color','w');
errorbar(gs, ze_sorted, ze_err_sorted, 'o-', 'LineWidth',1.2, 'MarkerSize',6);
xlabel('g'); ylabel('z_e [\mum]');
title('z_e from MC late-time tail (t>5 ns, last bin excluded)');
grid on;

% Save results
out.name = 'ze_MC_late_time_results.mat';
out.g = gs;
out.ze = ze_sorted;
out.ze_se = ze_err_sorted;
save(out.name, '-struct', 'out');
fprintf('Saved results to %s\n', out.name);
