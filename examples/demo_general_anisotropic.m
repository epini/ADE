%% demo_general_anisotropic.m
% Demo for the generalized ADE MATLAB functions in the fully anisotropic case.
%
% This example shows:
%   1) diffusion tensor and boundary conditions
%   2) total steady-state reflectance/transmittance
%   3) total time-resolved reflectance/transmittance
%   4) space-resolved reflectance/transmittance maps
%   5) directional anisotropy in the time domain
%
% Units convention:
%   lengths in mm, optical coefficients in mm^-1, time in ns.
%
% Author:       Ernesto Pini
% Affiliation:  Istituto Nazionale di Ricerca Metrologica (INRiM)
% Email:        pinie@lens.unifi.it

clear; close all; clc;

%% Add repository root to path
thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
addpath(repoRoot);

%% Medium and slab parameters
L     = 20.0;   % slab thickness [mm]
n_in  = 1.40;   % refractive index inside slab
n_ext = 1.00;   % refractive index outside slab

% Fully anisotropic scattering coefficients [mm^-1]
musx = 12.0;
musy = 5.0;
musz = 8.0;

g   = 0.85;     % Henyey-Greenstein asymmetry factor [-]
mua = 0.01;     % absorption coefficient [mm^-1]

% Initial lateral widths for space-time solutions [mm]
sx = 0.05;
sy = 0.05;

%% Grids
x = linspace(-30, 30, 121);     % [mm]
y = linspace(-30, 30, 121);     % [mm]
t = linspace(0.01, 4.0, 300); % [ns]

%% Diffusion tensor and boundary conditions
[Dx, Dy, Dz, infoD] = D_Tensor_ADE(n_in, musx, musy, musz, g);
[ze, z0, infoBC]    = BC_ADE(n_in, n_ext, musx, musy, musz, g);

fprintf('------------------------------------------------------------\n');
fprintf('General anisotropic ADE demo\n');
fprintf('------------------------------------------------------------\n');
fprintf('Input parameters:\n');
fprintf('  L    = %.3f mm\n', L);
fprintf('  n_in = %.3f\n', n_in);
fprintf('  n_ext= %.3f\n', n_ext);
fprintf('  musx = %.3f mm^-1\n', musx);
fprintf('  musy = %.3f mm^-1\n', musy);
fprintf('  musz = %.3f mm^-1\n', musz);
fprintf('  g    = %.3f\n', g);
fprintf('  mua  = %.4f mm^-1\n', mua);
fprintf('\n');
fprintf('Computed ADE parameters:\n');
fprintf('  Dx   = %.6f mm^2/ns\n', Dx);
fprintf('  Dy   = %.6f mm^2/ns\n', Dy);
fprintf('  Dz   = %.6f mm^2/ns\n', Dz);
fprintf('  ze   = %.6f mm\n', ze);
fprintf('  z0   = %.6f mm\n', z0);
fprintf('\n');
fprintf('Numerical info:\n');
fprintf('  D_Tensor_ADE converged: %d\n', infoD.converged);
if isfield(infoD, 'LmaxUsed')
    fprintf('  D_Tensor_ADE LmaxUsed:  %d\n', infoD.LmaxUsed);
end
if isfield(infoBC, 'LmaxUsed')
    fprintf('  BC_ADE LmaxUsed:        %d\n', infoBC.LmaxUsed);
end
fprintf('------------------------------------------------------------\n');

%% Total steady-state reflectance/transmittance
R = R_ADE(L, n_in, n_ext, musx, musy, musz, g, mua);
T = T_ADE(L, n_in, n_ext, musx, musy, musz, g, mua);
A = 1 - R - T;

fprintf('Energy balance:\n');
fprintf('  R = %.6f\n', R);
fprintf('  T = %.6f\n', T);
fprintf('  A = %.6f\n', A);
fprintf('------------------------------------------------------------\n');

%% Total time-resolved reflectance/transmittance
Rt = Rt_ADE(t, L, n_in, n_ext, musx, musy, musz, g, mua);
Tt = Tt_ADE(t, L, n_in, n_ext, musx, musy, musz, g, mua);

figure('Name','Total time-resolved signals','Color','w');
semilogy(t, Rt, 'LineWidth', 1.8); hold on;
semilogy(t, Tt, 'LineWidth', 1.8);
grid on; box on;
xlabel('t [ns]');
ylabel('Signal [ns^{-1}]');
ylim([1e-7, 10])
legend('R_t', 'T_t', 'Location', 'best');
title('Total time-resolved reflectance and transmittance');

%% Space-resolved steady-state maps
Rxy = Rxy_ADE(x, y, L, n_in, n_ext, musx, musy, musz, g, mua);
Txy = Txy_ADE(x, y, L, n_in, n_ext, musx, musy, musz, g, mua);

% Make sure maps are arranged as (y,x) for plotting
Rxy = ensureYX2D(Rxy, x, y, 'Rxy_ADE');
Txy = ensureYX2D(Txy, x, y, 'Txy_ADE');

% Floors for logarithmic visualization
Rxy_log = log10(max(Rxy, realmin));
Txy_log = log10(max(Txy, realmin));

fig = figure('Name','Space-resolved steady-state maps', ...
    'Color','w', ...
    'Position',[100 100 800 700]);

tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% Linear reflectance
nexttile;
imagesc(x, y, Rxy);
axis image; axis xy;
xlabel('x [mm]');
ylabel('y [mm]');
title('R(x,y)');
cb1 = colorbar;
cb1.Label.String = 'mm^{-2}';

% Linear transmittance
nexttile;
imagesc(x, y, Txy);
axis image; axis xy;
xlabel('x [mm]');
ylabel('y [mm]');
title('T(x,y)');
cb2 = colorbar;
cb2.Label.String = 'mm^{-2}';

% Log reflectance
nexttile;
imagesc(x, y, Rxy_log);
axis image; axis xy;
xlabel('x [mm]');
ylabel('y [mm]');
title('log_{10} R(x,y)');
cb3 = colorbar;
cb3.Label.String = 'log_{10}(mm^{-2})';

% Log transmittance
nexttile;
imagesc(x, y, Txy_log);
axis image; axis xy;
xlabel('x [mm]');
ylabel('y [mm]');
title('log_{10} T(x,y)');
cb4 = colorbar;
cb4.Label.String = 'log_{10}(mm^{-2})';

title(tl, 'Space-resolved steady-state reflectance and transmittance');

%% Time- and space-resolved maps at selected times
Rxyt = Rxyt_ADE(x, y, t, L, n_in, n_ext, musx, musy, musz, g, sx, sy, mua);
Txyt = Txyt_ADE(x, y, t, L, n_in, n_ext, musx, musy, musz, g, sx, sy, mua);

% Make sure arrays are arranged as (y,x,t)
Rxyt = ensureYX3D(Rxyt, x, y, t, 'Rxyt_ADE');
Txyt = ensureYX3D(Txyt, x, y, t, 'Txyt_ADE');

tSel = [0.05, 0.5, 2];  % ns
idxSel = arrayfun(@(tt) nearestIndex(t, tt), tSel);

fig = figure('Name','Selected space-time frames', ...
    'Color','w', ...
    'Position',[200 200 1100 650]);
tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

% First row: reflectance
for k = 1:numel(idxSel)
    it = idxSel(k);

    nexttile(k);
    imagesc(x, y, Rxyt(:,:,it));
    axis image; axis xy;
    xlabel('x [mm]');
    ylabel('y [mm]');
    title(sprintf('R, t = %.2f ns', t(it)));
    cb = colorbar;
    cb.Label.String = 'mm^{-2} ns^{-1}';
end

% Second row: transmittance
for k = 1:numel(idxSel)
    it = idxSel(k);

    nexttile(numel(idxSel) + k);
    imagesc(x, y, Txyt(:,:,it));
    axis image; axis xy;
    xlabel('x [mm]');
    ylabel('y [mm]');
    title(sprintf('T, t = %.2f ns', t(it)));
    cb = colorbar;
    cb.Label.String = 'mm^{-2} ns^{-1}';
end

title(tl, 'Selected space-time frames');

%% Directional anisotropy in the time domain
% Compare profiles along x (y = 0) and along y (x = 0)
xProbe = [0, 10, 20, 30];  % mm
yProbe = [0, 10, 20, 30];  % mm

ixProbe = arrayfun(@(xx) nearestIndex(x, xx), xProbe);
iyProbe = arrayfun(@(yy) nearestIndex(y, yy), yProbe);

ix0 = nearestIndex(x, 0);
iy0 = nearestIndex(y, 0);

figure('Name','Directional anisotropy in time','Color','w');

subplot(2,2,1); hold on;
for k = 1:numel(ixProbe)
    plot(t, squeeze(Rxyt(iy0, ixProbe(k), :)), 'LineWidth', 1.5);
    set(gca, 'yscale', 'log')
end
grid on; box on;
xlabel('t [ns]');
ylabel('R(x,0,t) [mm^{-2} ns^{-1}]');
ylim([1e-10 1])
legend(compose('x = %.1f mm', x(ixProbe)), 'Location', 'northeast');
title('Reflectance along x');

subplot(2,2,2); hold on;
for k = 1:numel(iyProbe)
    plot(t, squeeze(Rxyt(iyProbe(k), ix0, :)), 'LineWidth', 1.5);
    set(gca, 'yscale', 'log')
end
grid on; box on;
xlabel('t [ns]');
ylabel('R(0,y,t) [mm^{-2} ns^{-1}]');
ylim([1e-10 1]);
legend(compose('y = %.1f mm', y(iyProbe)), 'Location', 'northeast');
title('Reflectance along y');

subplot(2,2,3); hold on;
for k = 1:numel(ixProbe)
    plot(t, squeeze(Txyt(iy0, ixProbe(k), :)), 'LineWidth', 1.5);
    set(gca, 'yscale', 'log')
end
grid on; box on;
xlabel('t [ns]');
ylabel('T(x,0,t) [mm^{-2} ns^{-1}]');
ylim([1e-10 1])
legend(compose('x = %.1f mm', x(ixProbe)), 'Location', 'northeast');
title('Transmittance along x');

subplot(2,2,4); hold on;
for k = 1:numel(iyProbe)
    plot(t, squeeze(Txyt(iyProbe(k), ix0, :)), 'LineWidth', 1.5);
    set(gca, 'yscale', 'log')
end
grid on; box on;
xlabel('t [ns]');
ylabel('T(0,y,t) [mm^{-2} ns^{-1}]');
ylim([1e-10 1])
legend(compose('y = %.1f mm', y(iyProbe)), 'Location', 'northeast');
title('Transmittance along y');

sgtitle('Directional anisotropy: x versus y');

%% Simple 1D cuts of the steady-state maps
figure('Name','Steady-state cuts','Color','w');

subplot(1,2,1); hold on;
plot(x, Rxy(iy0, :), 'LineWidth', 1.8);
plot(y, Rxy(:, ix0), '--', 'LineWidth', 1.8);
grid on; box on;
xlabel('position [mm]');
ylabel('R [mm^{-2}]');
legend('R(x,0)', 'R(0,y)', 'Location', 'best');
title('Steady-state reflectance cuts');

subplot(1,2,2); hold on;
plot(x, Txy(iy0, :), 'LineWidth', 1.8);
plot(y, Txy(:, ix0), '--', 'LineWidth', 1.8);
grid on; box on;
xlabel('position [mm]');
ylabel('T [mm^{-2}]');
legend('T(x,0)', 'T(0,y)', 'Location', 'best');
title('Steady-state transmittance cuts');

%% Local helper functions
function Z = ensureYX2D(Z, x, y, fname)
    sx = numel(x);
    sy = numel(y);

    if isequal(size(Z), [sy, sx])
        return;
    elseif isequal(size(Z), [sx, sy])
        Z = Z.';
    else
        error('%s returned an array of unexpected size.', fname);
    end
end

function F = ensureYX3D(F, x, y, t, fname)
    sx = numel(x);
    sy = numel(y);
    st = numel(t);

    if isequal(size(F), [sy, sx, st])
        return;
    elseif isequal(size(F), [sx, sy, st])
        F = permute(F, [2, 1, 3]);
    else
        error('%s returned an array of unexpected size.', fname);
    end
end

function idx = nearestIndex(v, x0)
    [~, idx] = min(abs(v - x0));
end