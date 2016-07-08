% Check standard dual vs. primal Fourier implementation of GP regression

% Define squared exponential function
kSE = @(r,l,x)(r*exp(-.5*(bsxfun(@minus,x(:),x(:)')/l).^2));

aa = .5;  % arbitrary scalar that we can use to test that we're invariant to scale

% Set up grid for visualization
gridends = [-10 10]*aa; % range of function to consider
nx = 100; % number of grid points
xx = linspace(gridends(1),gridends(2),nx)';
gridrnge = diff(gridends)*(nx/(nx-1));

% Set params of SE covariance
rho = 1; % marginal variance
len = 3*aa;  % length scale
signse = .001; % observation noise
Kprior = kSE(rho,len,xx); % the covariance
mu = 0; % prior mean

clf; subplot(221);
imagesc(xx,xx,Kprior); title('prior on grid');

% sample true function
ftrue = mvnrnd(zeros(1,nx),Kprior);
subplot(223);
plot(xx,ftrue);
title('true function (on grid)');

% Set Fourier-domain version 
Tcirc = gridrnge+5*len; % location of circular boundary
minl = len; % mininum length scale to consider 
condthresh = 1e14; % threshold on condition number of covariance matrix

% set up Fourier frequencies
maxw = floor((Tcirc/(pi*minl))*sqrt(.5*log(condthresh)));  % max freq to use
nw = maxw*2+1; % number of fourier frequencies
[Bfft,wvec] = realnufftbasis(xx,Tcirc,nw); % make basis 

% test that Fourier-domain covariance matches
kfdiag = sqrt(2*pi)*rho*len*exp(-(2*pi^2/Tcirc^2)*len^2*wvec.^2);
Kfprior = Bfft'*diag(kfdiag)*Bfft;
subplot(222);
imagesc(xx,xx,Kfprior); title('prior on grid (fourier)')
subplot(224); 
plot(xx,Kprior(:,1),xx,Kfprior(:,1),'r--');
title('first column of covs');
legend('standard', 'fourier');

%% Now do GP regression using traditional and Fourier-domain versions

npts = 6; % number of points to learn

% kernel function, standard form
kfun = @(x1,x2)(rho*exp(-.5*(bsxfun(@minus,x1(:),x2(:)')/len).^2));

% Initialize Fourier domain representation
%Kinvfft = diag(1./kfdiag); % inverse of prior covariance in Fourier domain
Kfft = diag(kfdiag);
mufft = zeros(nw,1);

% initialize data
xobs = []; yobs = []; 

clf;
for jj = 1:npts

    % Draw sample data
    x0 = rand*gridrnge+gridends(1); % draw a sample point
    y0 = interp1(xx,ftrue,x0,'spline')+randn*signse; % sample response

    % update standard GP stuff
    xobs(jj,1) = x0;
    yobs(jj,1) = y0;
    Kobs = kfun(xobs,xobs)+diag(ones(jj,1))*signse^2;
    
    % Compute GP posterior on grid using standard formula
    kstar = kfun(xx,xobs); % the k-star matrix
    mugrd = mu+kstar*(Kobs\(yobs-mu)); % posterior mean
    Kgrd = Kprior-kstar*(Kobs\kstar'); % posterior covariance
    Ksd = 2*sqrt(diag(Kgrd)); % 2SD error bars
    
    % Plot standard GP posterior
    subplot(npts,2,2*jj-1);
    plot(xx,ftrue,'k-',xx,mugrd,'r',xobs,yobs,'r.','markersize',25);
    hold on;
    errorbarFill(xx,mugrd,Ksd); % plot error bars
    hold off; box off;
    if jj==1, title('standard GP'); end
    
    % Update Fourier-domain prior
    Bx = realnufftbasis(x0,Tcirc,nw);  % NUfft of x0
    KB = Kfft*Bx; % current posterior cov * Bx
    denom = (KB'*Bx+signse^2); % scalar denominator needed for updates
    Kfft = Kfft- KB*KB'/denom; % updated posterior cov
    mufft = mufft+KB*(y0-Bx'*mufft)/denom; % updated posterior mean
    Ksdfft = 2*sqrt(diag(Bfft'*Kfft*Bfft)); % error bars

    % Plot GP posterior from Fourier representation
    subplot(npts,2,2*jj);
    plot(xx,ftrue,'k-',xx,Bfft'*mufft,'r',xobs,yobs,'r.','markersize',25);
    hold on;
    errorbarFill(xx,mugrd,Ksdfft); % plot error bars
    hold off; box off;
    if jj==1, title('primal Fourier-domain GP'); end    

    pause;
end
