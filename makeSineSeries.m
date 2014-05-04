function T = makeSineSeries(Ncomp, freq_limits, tmax, ptsPerSec)
%T = makeSineSeries(Ncomp, freq_limits, tmax, ptsPerSec)
%
% Ncomp - number of sine wave components
% freq_limits - 2-element vector, min/max allowed frequencies
% tmax - time length of series
% ptsPerSec - Sampling rate
%
% T - [1 x Nsamples] time series

Npts = tmax*ptsPerSec;
x = linspace(0,tmax,Npts);
freqs = rand(Ncomp,1)*(freq_limits(2) - freq_limits(1)) + freq_limits(1);
xall = repmat(freqs,1,Npts).*repmat(x,Ncomp,1);
T = sum(sin(xall))*(1/Ncomp);

end