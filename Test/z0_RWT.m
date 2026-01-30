function [z0_mean, z0_se, stats] = z0_RWT(Nwalkers, g, lx, ly, lz, varargin)
% estimate_z0_pencil_parfor  Parallel estimate of z0 from pencil-beam RWT
%
% Usage:
%  [z0_mean, z0_se, stats] = estimate_z0_pencil_parfor(Nwalkers, g, lx, ly, lz, ...)
%
% Same semantics as the non-parallel function. Key differences:
%  - uses parfor to split walkers across workers
%  - each worker uses a private RandStream for reproducible RNG
%  - aggregates sums and sums-of-squares to compute mean/std/se
%
% Optional name/value pairs:
%   'NstepsBase' (default 1000), 'Adaptive' (default true),
%   'Seed' (default []), 'Verbose' (default true), 'Nchunks' (default: num workers)
%
% Example:
%  [z0, z0_se, stats] = estimate_z0_pencil_parfor(1e5, 0.0, 40, 20, 10, 'Seed', 1234);

% ---- parse inputs ----
p = inputParser;
addRequired(p, 'Nwalkers', @(x) isnumeric(x) && isscalar(x) && x>0);
addRequired(p, 'g', @(x) isnumeric(x) && isscalar(x));
addRequired(p, 'lx', @(x) isnumeric(x) && isscalar(x) && x>0);
addRequired(p, 'ly', @(x) isnumeric(x) && isscalar(x) && x>0);
addRequired(p, 'lz', @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'NstepsBase', 1000, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'Adaptive', true, @(x) islogical(x) || ismember(x,[0 1]));
addParameter(p, 'Seed', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
addParameter(p, 'Verbose', true, @(x) islogical(x) || ismember(x,[0 1]));
addParameter(p, 'Nchunks', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x>0));
parse(p, Nwalkers, g, lx, ly, lz, varargin{:});

Nwalkers = round(p.Results.Nwalkers);
g = double(p.Results.g);
lx = double(p.Results.lx);
ly = double(p.Results.ly);
lz = double(p.Results.lz);
NstepsBase = round(p.Results.NstepsBase);
Adaptive = logical(p.Results.Adaptive);
seed = p.Results.Seed;
Verbose = logical(p.Results.Verbose);
Nchunks_user = p.Results.Nchunks;

% ---- determine parallel pool and chunks ----
pobj = gcp('nocreate');
if isempty(pobj)
    if Verbose
        fprintf('No parallel pool found: creating one with default settings...\n');
    end
    pobj = parpool(); % create default pool
end
numWorkers = pobj.NumWorkers;

if isempty(Nchunks_user)
    Nchunks = numWorkers;  % one chunk per worker by default
else
    Nchunks = min(Nchunks_user, Nwalkers);
end

% allow more fine-grained chunking if desired (e.g. multiple chunks per worker)
% but keep it <= Nwalkers
Nchunks = min(Nchunks, Nwalkers);

if Verbose
    fprintf('Parallel run: %d workers, %d chunks (Nwalkers=%d)\n', numWorkers, Nchunks, Nwalkers);
end

% ---- scattering coefficients mu = 1/ell ----
mux = 1.0 ./ lx;
muy = 1.0 ./ ly;
muz = 1.0 ./ lz;

% ---- choose Nsteps (adaptive) ----
if Adaptive
    Nsteps = max(NstepsBase, round(NstepsBase * (1 + abs(g))));
else
    Nsteps = NstepsBase;
end
Nsteps = max(1, round(Nsteps));

if Verbose
    fprintf('Using Nsteps=%d (Adaptive=%d)\n', Nsteps, Adaptive);
end

% ---- determine chunk sizes (balanced) ----
baseChunk = floor(Nwalkers / Nchunks);
remainder = mod(Nwalkers, Nchunks);
chunkSizes = baseChunk * ones(1, Nchunks);
chunkSizes(1:remainder) = chunkSizes(1:remainder) + 1;
chunkStarts = [1, cumsum(chunkSizes(1:end-1))+1];
chunkEnds = cumsum(chunkSizes);

% ---- prepare arrays to collect partial sums ----
partial_sum_x = zeros(1, Nchunks);
partial_sum_y = zeros(1, Nchunks);
partial_sum_z = zeros(1, Nchunks);
partial_sumx2 = zeros(1, Nchunks);
partial_sumy2 = zeros(1, Nchunks);
partial_sumz2 = zeros(1, Nchunks);
partial_time = zeros(1, Nchunks);

% ---- parallel loop over chunks ----
musFloor = 1e-12;
tStartTotal = tic;

parfor ich = 1:Nchunks
    % local rng per worker+chunk for reproducibility
    % Use getCurrentTask to vary seed per worker. If running without pool, getCurrentTask empty -> use base seed.
    tid = getCurrentTask();
    if isempty(tid)
        workerID = ich; % fallback single-worker
    else
        workerID = tid.ID + ich; % combine task id and chunk idx
    end
    if isempty(seed)
        seed_local = sum(100*clock) + ich; % not perfectly reproducible if seed not given
    else
        seed_local = double(seed) + double(ich) + double(workerID)*17;
    end
    rs = RandStream('Threefry','Seed',uint32(mod(seed_local,2^31-1)));
    % Use rand(rs,...) for RNG calls below

    % chunk size and local number of walkers
    w0 = chunkStarts(ich);
    w1 = chunkEnds(ich);
    W = w1 - w0 + 1;

    % allocate local vectors (single precision to save mem)
    xpos = zeros(1, W, 'single');
    ypos = zeros(1, W, 'single');
    zpos = zeros(1, W, 'single');

    % initial pencil directions (0,0,1)
    sx = zeros(1, W, 'single');
    sy = zeros(1, W, 'single');
    sz = ones(1, W, 'single');

    t0 = tic;
    % main loop across steps
    for k = 1:Nsteps
        % compute mus (double for stability)
        sx_d = double(sx); sy_d = double(sy); sz_d = double(sz);
        mus = mux .* (sx_d.^2) + muy .* (sy_d.^2) + muz .* (sz_d.^2);
        mus = max(mus, musFloor);

        % sample L: use rand(rs,1,W)
        U = rand(rs, 1, W);
        L = single(-log(double(U)) ./ mus);

        % update positions
        xpos = xpos + L .* sx;
        ypos = ypos + L .* sy;
        zpos = zpos + L .* sz;

        % sample HG cos(theta)
        xi = rand(rs, 1, W);
        if abs(g) < 1e-12
            ct = 2*xi - 1;
        else
            numer = (1 - g^2);
            denomTerm = (1 - g + 2*g.*xi);
            denomTerm = max(denomTerm, 1e-12);
            tmp = (numer ./ denomTerm).^2;
            ct = (1/(2*g)).*(1 + g^2 - tmp);
            ct = max(-1, min(1, ct));
        end
        st = sqrt(max(0, 1 - ct.^2));
        phi = 2*pi*rand(rs, 1, W);

        % rotate directions (call local helper)
        [sx, sy, sz] = rotate_fast_vec_local(sx, sy, sz, ct, st, phi);
    end
    t_elapsed = toc(t0);
    % compute partial sums and sums of squares (convert to double before aggregating to reduce rounding error)
    sx_d = double(xpos); sy_d = double(ypos); sz_d = double(zpos);
    partial_sum_x(ich) = sum(sx_d);
    partial_sum_y(ich) = sum(sy_d);
    partial_sum_z(ich) = sum(sz_d);
    partial_sumx2(ich) = sum(sx_d.^2);
    partial_sumy2(ich) = sum(sy_d.^2);
    partial_sumz2(ich) = sum(sz_d.^2);
    partial_time(ich) = t_elapsed;
end

% ---- aggregate global sums ----
total_sum_x = sum(partial_sum_x);
total_sum_y = sum(partial_sum_y);
total_sum_z = sum(partial_sum_z);

total_sumx2 = sum(partial_sumx2);
total_sumy2 = sum(partial_sumy2);
total_sumz2 = sum(partial_sumz2);

% compute means
mean_x = total_sum_x / Nwalkers;
mean_y = total_sum_y / Nwalkers;
mean_z = total_sum_z / Nwalkers;

% compute variances (population)
var_x = (total_sumx2 / Nwalkers) - mean_x^2;
var_y = (total_sumy2 / Nwalkers) - mean_y^2;
var_z = (total_sumz2 / Nwalkers) - mean_z^2;

% numerical guard
var_x = max(0, var_x); var_y = max(0, var_y); var_z = max(0, var_z);

std_x = sqrt(var_x);
std_y = sqrt(var_y);
std_z = sqrt(var_z);

se_x = std_x / sqrt(Nwalkers);
se_y = std_y / sqrt(Nwalkers);
se_z = std_z / sqrt(Nwalkers);

z0_mean = mean_z;
z0_se = se_z;

total_time = toc(tStartTotal);

% stats
stats = struct();
stats.mean = [mean_x, mean_y, mean_z];
stats.std  = [std_x, std_y, std_z];
stats.se   = [se_x, se_y, se_z];
stats.Nwalkers = Nwalkers;
stats.Nsteps = Nsteps;
stats.g = g;
stats.lx = lx; stats.ly = ly; stats.lz = lz;
stats.nchunks = Nchunks;
stats.runtime_s = total_time;
stats.worker_times = partial_time;

if Verbose
    fprintf('Parallel run done in %.2f s (wall clock). Mean z = %.6g, se = %.6g\n', total_time, z0_mean, z0_se);
end

end

%% local helper: rotate (works inside parfor)
function [sx2, sy2, sz2] = rotate_fast_vec_local(sx, sy, sz, ct, st, phi)
    % operate in double internally for numeric stability, return single
    sx_d = double(sx); sy_d = double(sy); sz_d = double(sz);
    cphi = cos(double(phi));
    sphi = sin(double(phi));

    W = numel(sx_d);
    ux = zeros(1,W); uy = zeros(1,W); uz = zeros(1,W);
    vx = zeros(1,W); vy = zeros(1,W); vz = zeros(1,W);

    poleThr = 0.999999;
    mask = abs(sz_d) < poleThr;

    if any(mask)
        szm = sz_d(mask);
        inv = 1 ./ sqrt(1 - szm.^2);

        uxm = -sy_d(mask) .* inv;
        uym =  sx_d(mask) .* inv;

        vxm = -sx_d(mask) .* szm .* inv;
        vym = -sy_d(mask) .* szm .* inv;
        vzm = (1 - szm.^2) .* inv;

        ux(mask) = uxm; uy(mask) = uym; uz(mask) = 0;
        vx(mask) = vxm; vy(mask) = vym; vz(mask) = vzm;
    end

    if any(~mask)
        ux(~mask) = 1; uy(~mask) = 0; uz(~mask) = 0;
        vx(~mask) = 0; vy(~mask) = 1; vz(~mask) = 0;
    end

    tx = cphi .* ux + sphi .* vx;
    ty = cphi .* uy + sphi .* vy;
    tz = cphi .* uz + sphi .* vz;

    sx2 = double(ct) .* sx_d + double(st) .* tx;
    sy2 = double(ct) .* sy_d + double(st) .* ty;
    sz2 = double(ct) .* sz_d + double(st) .* tz;

    invn = 1 ./ max(sqrt(sx2.^2 + sy2.^2 + sz2.^2), 1e-12);
    sx2 = sx2 .* invn;
    sy2 = sy2 .* invn;
    sz2 = sz2 .* invn;

    sx2 = single(sx2);
    sy2 = single(sy2);
    sz2 = single(sz2);
end
