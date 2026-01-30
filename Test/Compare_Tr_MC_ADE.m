%% Compare MC (2 sources) vs theory (3 models) using params from MAT meta
clear; close all; clc;

%% Paths
addpath('C:\Users\ernes\Desktop\PostDoc\Scripts MATLAB')
addpath('C:\Users\ernes\Desktop\PostDoc\MC\Anisotropy\Anisotropic Diffusion Equation\Functions\ADE function with g')

%% Load MAT file
matFile = 'tr_transmission_curves_3.mat';   % set full path if needed
S = load(matFile);

% --- time axis (ps) ---
if isfield(S,'t_ps')
    t_ps = S.t_ps(:);
else
    t_ps = S.t(:) * 1e12;  % seconds -> ps
end

% --- MC curves ---
MC_pencil = S.pencil_beam(:);
MC_iso    = S.isotropic_source(:);

%% Extract parameters from meta
assert(isfield(S,'meta'), 'MAT file does not contain "meta".');

meta = S.meta;
% SciPy sometimes loads as 1x1 struct array
if numel(meta) > 1
    meta = meta(1);
end

g      = meta.g;
lx     = meta.lx;
ly     = meta.ly;
lz     = meta.lz;
mua    = meta.mua_l;
n_in   = meta.n_sample;
n_ext  = 1.0;          % external medium
L      = 5000;         % still not in MAT -> keep here or add to meta in Python

nphoton    = meta.nphotons;

% Normalize for shape comparison (comment out if you want raw counts)
MC_pencil = MC_pencil/nphoton;
MC_iso    = MC_iso/nphoton;

%% Theory curves (3 models)
dt_ps = mean(diff(t_ps));

T_ADE  = Tt_ADE(t_ps, L, n_in, n_ext, lx, ly, lz, mua, g) * dt_ps;
T_CORR = Tt_ADE_corrected(t_ps, L, n_in, n_ext, lx, ly, lz, mua, g) * dt_ps;
T_ALER = Tt_ADE_Alerstam(t_ps, L, n_in, n_ext, lx, ly, lz, mua, g) * dt_ps;

% Normalize theory too (comment out if you want absolute scaling)
% T_ADE  = T_ADE;
% T_CORR = T_CORR;
% T_ALER = T_ALER;

%% Plot: Overlay
figure('Name','TR transmission: MC vs theory'); hold on; box on; grid on;

plot(t_ps, T_ADE,  'LineWidth', 1.2);
plot(t_ps, T_CORR, 'LineWidth', 1.2);
plot(t_ps, T_ALER, 'LineWidth', 1.2);
plot(t_ps, MC_pencil, ':', 'LineWidth', 1.4);
plot(t_ps, MC_iso,    ':', 'LineWidth', 1.4);

set(gca, 'yscale', 'log')

xlabel('$t$ [ps]','Interpreter','latex','FontSize',14);
ylabel('$I(t)$ [a.u.]','Interpreter','latex','FontSize',14);
title(sprintf('Transmittance through anisotropic slab, g=%.2f', g), ...
      'Interpreter','latex','FontSize',14);

legend({'ADE $\lambda$', ...
        'ADE $\lambda$ corrected', ...
        'ADE Alerstam', ...
        'MC pencil beam', ...
        'MC isotropic source'}, ...
        'Interpreter','latex','Location','best','FontSize',12);

xlim([0 6000]);
ylim([1e-7 5e-2]);

%% Plot: Relative errors (vs each MC curve)
eps0 = 1e-30;

figure('Name','Relative error vs MC pencil'); hold on; box on; grid on;
plot(t_ps, (T_ADE  - MC_pencil) ./ (MC_pencil + eps0), 'LineWidth', 1.2);
% plot(t_ps, (T_CORR - MC_pencil) ./ (MC_pencil + eps0), 'LineWidth', 1.2);
% plot(t_ps, (T_ALER - MC_pencil) ./ (MC_pencil + eps0), 'LineWidth', 1.2);
xlabel('$t$ [ps]','Interpreter','latex','FontSize',14);
ylabel('$\Delta I(t)/I_{MC}(t)$','Interpreter','latex','FontSize',14);
title('Relative error (reference: MC pencil beam)','Interpreter','latex','FontSize',14);
legend({'ADE $\lambda$','ADE $\lambda$ corrected','ADE Alerstam'}, ...
       'Interpreter','latex','Location','best','FontSize',12);
xlim([500 10000]); ylim([-0.25 0.25]);

figure('Name','Relative error vs MC isotropic'); hold on; box on; grid on;
plot(t_ps, (T_ADE  - MC_iso) ./ (MC_iso + eps0), 'LineWidth', 1.2);
% plot(t_ps, (T_CORR - MC_iso) ./ (MC_iso + eps0), 'LineWidth', 1.2);
% plot(t_ps, (T_ALER - MC_iso) ./ (MC_iso + eps0), 'LineWidth', 1.2);
xlabel('$t$ [ps]','Interpreter','latex','FontSize',14);
ylabel('$\Delta I(t)/I_{MC}(t)$','Interpreter','latex','FontSize',14);
title('Relative error (reference: MC isotropic source)','Interpreter','latex','FontSize',14);
legend({'ADE $\lambda$','ADE $\lambda$ corrected','ADE Alerstam'}, ...
       'Interpreter','latex','Location','best','FontSize',12);
xlim([500 10000]); ylim([-0.25 0.25]);
