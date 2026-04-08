function export_resolved_reference()
%EXPORT_RESOLVED_REFERENCE Export MATLAB reference outputs for ADE validation.
%
%   This script computes MATLAB reference outputs for the remaining ADE
%   functions and saves them to resolved_reference.json in the current
%   folder. Units convention:
%     lengths in mm, optical coefficients in mm^-1, time in ns.

out = struct();

%% Common benchmark case
params = struct();
params.L     = 20.0;
params.n_in  = 1.40;
params.n_ext = 1.00;
params.musx  = 12.0;
params.musy  = 8.0;
params.musz  = 5.0;
params.g     = 0.85;
params.mua   = 0.01;
params.sx    = 0.05;
params.sy    = 0.07;

out.params = params;

%% Total steady-state
out.total = struct();
out.total.R = R_ADE(params.L, params.n_in, params.n_ext, params.musx, params.musy, params.musz, params.g, params.mua);
out.total.T = T_ADE(params.L, params.n_in, params.n_ext, params.musx, params.musy, params.musz, params.g, params.mua);

%% Total time-resolved
out.time = struct();
out.time.t = linspace(0.02, 3.0, 121);
out.time.Rt = Rt_ADE(out.time.t, params.L, params.n_in, params.n_ext, params.musx, params.musy, params.musz, params.g, params.mua);
out.time.Tt = Tt_ADE(out.time.t, params.L, params.n_in, params.n_ext, params.musx, params.musy, params.musz, params.g, params.mua);

%% Space-resolved steady-state
out.space = struct();
out.space.x = linspace(-3.0, 3.0, 31);
out.space.y = linspace(-2.5, 2.5, 27);
out.space.Rxy = Rxy_ADE(out.space.x, out.space.y, params.L, params.n_in, params.n_ext, params.musx, params.musy, params.musz, params.g, params.mua);
out.space.Txy = Txy_ADE(out.space.x, out.space.y, params.L, params.n_in, params.n_ext, params.musx, params.musy, params.musz, params.g, params.mua);

%% Time- and space-resolved
out.spacetime = struct();
out.spacetime.x = linspace(-2.0, 2.0, 17);
out.spacetime.y = linspace(-1.5, 1.5, 15);
out.spacetime.t = linspace(0.02, 1.20, 25);
out.spacetime.Rxyt = Rxyt_ADE(out.spacetime.x, out.spacetime.y, out.spacetime.t, params.L, params.n_in, params.n_ext, params.musx, params.musy, params.musz, params.g, params.sx, params.sy, params.mua);
out.spacetime.Txyt = Txyt_ADE(out.spacetime.x, out.spacetime.y, out.spacetime.t, params.L, params.n_in, params.n_ext, params.musx, params.musy, params.musz, params.g, params.sx, params.sy, params.mua);

json_text = jsonencode(out);
fid = fopen('resolved_reference.json', 'w');
fprintf(fid, '%s', json_text);
fclose(fid);

fprintf('Wrote resolved_reference.json\n');
end
