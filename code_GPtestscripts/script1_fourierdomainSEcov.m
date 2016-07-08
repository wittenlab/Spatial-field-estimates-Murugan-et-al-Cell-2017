% script1_fourierdomainSEcov.m
%
% Examine original vs. Fourier-domain (diagonlized Fourier-domain) squared exponential covariance

% squared exponential (SE) covariance function
kSE = @(r,l,x)(r*exp(-.5*(bsxfun(@minus,x(:),x(:)')/l).^2));

% Test scale invariance by changing this constant (changes support).
aa = .25;  

% Set up grid for visualization
gridends = [-10 10]*aa; % range of function to consider
nx = 100; % number of grid points
xx = linspace(gridends(1),gridends(2),nx)';
gridrnge = diff(gridends)*(nx/(nx-1));

% Set params of SE covariance
rho = 1; % marginal variance
len = 1*aa;  % length scale
Kprior = kSE(rho,len,xx); % the covariance

% sample true function
ftrue = mvnrnd(zeros(1,nx),Kprior);

%% Now let's redo everything in Fourier domain

% Set Fourier-domain version 
Tcirc = gridrnge+4*len; % location of circular boundary
minl = len; % mininum length scale to consider 
condthresh = 1e14; % threshold on condition number of covariance matrix

% set up Fourier frequencies
maxw = floor((Tcirc/(pi*minl))*sqrt(.5*log(condthresh)));  % max freq to use
nw = maxw*2+1; % number of fourier frequencies
[Bfft,wvec] = realnufftbasis(xx,Tcirc,nw); % make basis 

% test that Fourier-domain covariance matches
kfdiag = sqrt(2*pi)*rho*len*exp(-(2*pi^2/Tcirc^2)*len^2*wvec.^2); % spectral density
Kfprior = Bfft'*diag(kfdiag)*Bfft;

% make plots
subplot(221);imagesc(xx,xx,Kprior); axis image; title('prior on grid');
subplot(222); imagesc(xx,xx,Kfprior); axis image; title('prior on grid (fourier)')
subplot(223); plot(xx,ftrue); title('sample from GP (on grid)'); axis tight;
subplot(224);  plot(xx,Kprior(:,1),xx,Kfprior(:,1),'r--'); axis tight; 
title('first column of covs'); legend('standard', 'fourier');


