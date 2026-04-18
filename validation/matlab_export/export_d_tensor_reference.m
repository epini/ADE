% export_d_tensor_reference.m
% Export MATLAB benchmark values for D_Tensor_ADE to JSON.
%
% Run this script from MATLAB with the repository root or matlab/ folder on
% the path. It writes d_tensor_reference.json in the current folder.
% NOTE: python/tests/test_diffusion_reference.py expects a top-level JSON
% array. Keep the test reader in sync if this schema changes.

clear; clc;

cases = {
    struct('name','iso_g0',    'n_in',1.4, 'musx',10.0, 'musy',10.0, 'musz',10.0, 'g',0.0), ...
    struct('name','iso_g085',  'n_in',1.4, 'musx',10.0, 'musy',10.0, 'musz',10.0, 'g',0.85), ...
    struct('name','aniso_g0',  'n_in',1.4, 'musx',12.0, 'musy',8.0,  'musz',5.0,  'g',0.0), ...
    struct('name','aniso_g085','n_in',1.4, 'musx',12.0, 'musy',8.0,  'musz',5.0,  'g',0.85) ...
};

out = struct([]);

for k = 1:numel(cases)
    c = cases{k};
    [Dx, Dy, Dz, info] = D_Tensor_ADE(c.n_in, c.musx, c.musy, c.musz, c.g);

    out(k).name  = c.name;
    out(k).n_in  = c.n_in;
    out(k).musx  = c.musx;
    out(k).musy  = c.musy;
    out(k).musz  = c.musz;
    out(k).g     = c.g;
    out(k).Dx    = Dx;
    out(k).Dy    = Dy;
    out(k).Dz    = Dz;
    out(k).info  = struct( ...
        'converged', logical(info.converged), ...
        'LmaxUsed', info.LmaxUsed, ...
        'Nchi', info.Nchi, ...
        'Nphi', info.Nphi, ...
        'RelTol', info.RelTol, ...
        'AbsTol', info.AbsTol, ...
        'LmaxStart', info.LmaxStart ...
    );
end

json_text = jsonencode(out, 'PrettyPrint', true);
fid = fopen('d_tensor_reference.json', 'w');
if fid < 0
    error('Could not open d_tensor_reference.json for writing.');
end
cleanupObj = onCleanup(@() fclose(fid));
fwrite(fid, json_text, 'char');

fprintf('Wrote d_tensor_reference.json with %d benchmark cases.\n', numel(out));
