function reducedOpticalThicknessZ = warn_transport_thin_regime(L, musz, g)
%WARN_TRANSPORT_THIN_REGIME Warn when the ADE slab is transport-thin along z.
%
%   reducedOpticalThicknessZ = WARN_TRANSPORT_THIN_REGIME(L, musz, g)
%   computes the reduced optical thickness L*musz*(1-g). When this value
%   is below 1, the parameters lie well beyond the usual validity range of
%   the diffusion approximation, so the resulting signals may exhibit
%   nonphysical sign instabilities.

reducedOpticalThicknessZ = L * musz * (1 - g);

if reducedOpticalThicknessZ < 1
    warning('generalized_ade:ThinTransportThickness', ...
        ['Reduced optical thickness along z is such that L*musz*(1-g) = %.3g < 1. ', ...
         'These parameters lie well beyond the usual validity range of the diffusion approximation, ', ...
         'so the outputs may exhibit nonphysical sign instabilities.'], ...
        reducedOpticalThicknessZ);
end

end
