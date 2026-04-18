function export_bc_reference()
%EXPORT_BC_REFERENCE Export MATLAB BC_ADE benchmark values to JSON.
% NOTE: python/tests/test_boundary_reference.py expects a top-level JSON
% array. Keep the test reader in sync if this schema changes.

cases = {
    struct('name','iso_nmatch_g0',   'n_in',1.4,'n_ext',1.4,'musx',10.0,'musy',10.0,'musz',10.0,'g',0.0), ...
    struct('name','iso_nmatch_g085', 'n_in',1.4,'n_ext',1.4,'musx',10.0,'musy',10.0,'musz',10.0,'g',0.85), ...
    struct('name','iso_mismatch_g0', 'n_in',1.4,'n_ext',1.0,'musx',10.0,'musy',10.0,'musz',10.0,'g',0.0), ...
    struct('name','aniso_g0',        'n_in',1.4,'n_ext',1.0,'musx',12.0,'musy',8.0,'musz',5.0,'g',0.0), ...
    struct('name','aniso_g085',      'n_in',1.4,'n_ext',1.0,'musx',12.0,'musy',8.0,'musz',5.0,'g',0.85) ...
};

out = struct([]);
for k = 1:numel(cases)
    c = cases{k};
    [ze, z0, info] = BC_ADE(c.n_in, c.n_ext, c.musx, c.musy, c.musz, c.g);
    out(k).name = c.name;
    out(k).n_in = c.n_in;
    out(k).n_ext = c.n_ext;
    out(k).musx = c.musx;
    out(k).musy = c.musy;
    out(k).musz = c.musz;
    out(k).g = c.g;
    out(k).ze = ze;
    out(k).z0 = z0;
    out(k).case = info.case;
end

json = jsonencode(out, PrettyPrint=true);
fid = fopen('bc_reference.json', 'w');
fwrite(fid, json, 'char');
fclose(fid);

fprintf('Wrote bc_reference.json\n');
end
