%% estimPlaceFields_Tmaze.m
%
% Uses Gaussian Processes regression to make nonparametric estimates of
% spatial tuning of Ca imaging responses.

% Set some general paramters
tau_gcamp = 0.7; % assumed timescale of Ca dye impulse response (s)
dtbin = .1;  % assumed time bin size for data (s)

% Grid size for binning spatial locations (speeds up inference)
xybinwidth = 2; % grid spacing for spatial points

% Add needed paths
addpath code_fastASD;
addpath tools_misc;

% Load rat position data
load ../Mouse297_samplecells; % CHANGE THIS
x1 = X_scope'; clear X_scope
x2 = Y_scope'; clear Y_scope

% Get rid of nans
iinan = isnan(x1);
x1(iinan) = [];
x2(iinan) = [];

% Possible cell numbers
cnums = {'04','15','21','23','24','40','65'};

% Pick a cell number 
cellnum = 1;

% Load it
yname = sprintf('trace_%s',cnums{cellnum});
fprintf('Loading datafile:  %s\n', yname);
eval(sprintf('y=%s'';',yname));

% Preprocess
y(iinan) = []; % remove nans
nsamps = length(y); % number of time samples


%% Bin spatial positions

xp1orig = unique(x1); % unique x locations
xp2orig = unique(x2); % unique y locations
x1crs = round(x1/xybinwidth)*xybinwidth;  % gridded locations
x2crs = round(x2/xybinwidth)*xybinwidth;  % gridded locations

xp1 = unique(x1crs); 
xp2 = unique(x2crs);
n1 = length(xp1);
n2 = length(xp2);

% Insert stimuli into design matrix
xntrp1 = interp1(xp1,1:n1,x1crs,'nearest');
xntrp2 = interp1(xp2,1:n2,x2crs,'nearest');
xstim = sparse(1:nsamps,xntrp1+n1*(xntrp2-1),1,nsamps,n1*n2);

% Plot occupancy and spatial grid points with data
subplot(323); plot(x1,x2,'k.'); 
%hold on; plot(x1crs,x2crs,'r.', 'markersize', 8);hold off;
axis equal; axis tight; box off;


%%  Find ML estimate (position triggered averate) 
Fmlvec =(xstim'*xstim+.00001*eye(n1*n2))\(xstim'*y);
Fml = reshape(Fmlvec,n1,n2)';

% Plot it
subplot(321);
imagesc(xp1,xp2,Fml); 
 axis xy; axis equal; axis tight; box off;
colorbar;
title('ML estimate (gridded)');
set(gca,'tickdir','out');
xlabel('x position'); ylabel('y position');

%% Compute measurement matrix (for exponential filtering)

nlags = 20;  % number of bins to use for computing cross-corr.
yxc = xcorr(y,nlags, 'unbiased');  % cross-correlation
tt = (-nlags:nlags)*dtbin;

subplot(325);
plot(tt,yxc,tt,1*yxc(nlags+1)*exp(-abs(tt)/(tau_gcamp)));
xlabel('lag (ms)');
ylabel('cross-corr');
box off;
set(gca,'tickdir','out');
legend('data','Ca-dye');

%% Compute ASD estimate
fprintf('\n\n...Running GP regression...\n');

% Make dynamics matrix
Ai = spdiags(ones(nsamps,1)*[-exp(-1/(tau_gcamp/dtbin)), 1],-1:0,nsamps,nsamps);
ydeconv = Ai*y;  % deconvolved y

% Run ASD
minlens = 3 ;  % minimum length scale for each dimension
[Fmap,asdstats] = fastASD(xstim,ydeconv,[n1,n2],minlens);

% Mask spatial regions with no data
Fstim = sum(xstim)'>0;
Fmap = reshape(Fmap.*Fstim,n1,n2)';


%% Make plot
ah1 = subplot(322);
imagesc(xp1, xp2, Fmap); 
axis xy; axis equal; axis tight; box off;
colorbar;
title(['GP estimate: cell ',cnums{cellnum}]);
set(gca,'tickdir','out');
pos1 = get(ah1,'position');

% Find vertical min and max slice
[~,isliceX] = min(abs(xp2));
[~,isliceY] = min(abs(xp1));

% Error bars  (+/- 2SD);
Fci = 1.96*sqrt(reshape(asdstats.Lpostdiag,n1,n2)');

subplot(324);
plot(xp1,Fmap(isliceX,:), 'b', 'linewidth', 2);
hold on;
ebregionplot(xp1,Fmap(isliceX,:),Fci(isliceX,:),Fci(isliceX));
plot(xp1,xp1*0,'k--');
hold off; box off; axis tight;
title('horizontal slice');
xlabel('x position');
ylabel('mean resp');
set(gca,'tickdir','out');

subplot(326);
plot(xp2,Fmap(:,isliceY),'r', 'linewidth', 2);
hold on;
ebregionplot(xp2,Fmap(:,isliceY),Fci(:,isliceY),Fci(:,isliceY));
plot(xp2,xp2*0,'k--');
hold off; box off; axis tight;
title('vertical slice');
xlabel('x position');
ylabel('mean resp');
set(gca,'tickdir','out');

