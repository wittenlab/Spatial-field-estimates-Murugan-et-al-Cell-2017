%% script_160714_conditionalPlaceMaps.m
%
% Compute place tuning map for different conditionings of raw data

% Add needed paths
addpath code_fastASD;
addpath tools_misc;
dtbin = .1;  % assumed time bin size for data (s)

% Load rat position data
datapath = '../data_Malu_Jun2016/'; % CHANGE THIS
load([datapath '2863a_variables.mat']);  % load data
load([datapath 'velocity']);  % load velocity
v(1) = 0; % set velocity for first bin (avoid NaN).
[nneur,nsamps] = size(alldeltaf);

% Deconcolve calcium dynamics
tau_gcamp = 0.7; % timescale of Ca dye impulse response (s)
Ai = spdiags(ones(nsamps,1)*[-exp(-1/(tau_gcamp/dtbin)), 1],-1:0,nsamps,nsamps);
yy = Ai*alldeltaf';  % deconvolved y
% yy = yy./std(yy(:)); % normalize by mean std across all yy
yy = bsxfun(@rdivide,yy,std(yy)); % standardize variance for each neuron

%% Bin spatial locations

% Cast as double; 
x1 = double(x1); y1 = double(y1); 

% Grid size for binning spatial locations (speeds up inference)
xybinwidth = 8; % grid spacing for spatial points

xp1orig = unique(x1); % unique x locations
xp2orig = unique(y1); % unique y locations
x1crs = round(x1/xybinwidth)*xybinwidth;  % gridded locations
x2crs = round(y1/xybinwidth)*xybinwidth;  % gridded locations

xp1 = unique(x1crs); 
xp2 = unique(x2crs);
n1 = length(xp1);
n2 = length(xp2);

% Insert stimuli into design matrix
xntrp1 = interp1(xp1,1:n1,x1crs,'nearest');
xntrp2 = interp1(xp2,1:n2,x2crs,'nearest');
xstim = sparse(1:nsamps,xntrp1+n1*(xntrp2-1),1,nsamps,n1*n2);

% Plot occupancy and spatial grid points with data
clf; subplot(331); plot(x1,y1,'k.'); title('raw occupancy');
axis equal; axis tight; box off;


%% Set variables of interest for conditioning

% Variables of interest
% ---------------------
% m1_groom: when  imaging mouse (m1) grooms social target.
% m1_sniff: when imaging mouse (m1) pursues social target.
% m1_pursue: when imaging mouse (m1) pursues the social target.
% m1_allinvest: if imaging mouse is sniff/groom/pursuing the social target.
% m2_invest: if the imaging is mouse is being investigated by the social target. 

% Compute binarized velocity for low, medium or high
vlo = v>quantile(v,.25);
vmed = v>median(v); % times of 
vhi = v>quantile(v,.75);

varnames = {'m1_groom', 'm1_sniff', 'm1_pursue', 'm1_allinvest', 'm2_invest', 'vlo', 'vmed', 'vhi'};
nvar = length(varnames);

% Pick a variable of interest:
varnum = 4;
eval(sprintf('zz=%s;', varnames{varnum})); % grab relevant variable

% indices for two states
ii0 = (zz==0);
ii1 = (zz==1);

% plot occupany maps for two states
subplot(331); plot(x1(ii0),y1(ii0),'b.',x1(ii1),y1(ii1),'r.');
%hold on; plot(x1crs,y1crs,'r.', 'markersize', 8);hold off;
axis equal; axis tight; box off;
title(varnames{varnum},'interpreter','none')

%% Loop over cells (if desired

jjneur = 5;  % cell number

% Find ML estimate (position triggered average for each condition) 
Fml0 =(xstim(ii0,:)'*xstim(ii0,:)+.00001*eye(n1*n2))\(xstim(ii0,:)'*y(ii0));
Fml0 = reshape(Fml0,n1,n2)';
Fml1 =(xstim(ii1,:)'*xstim(ii1,:)+.00001*eye(n1*n2))\(xstim(ii1,:)'*y(ii1));
Fml1 = reshape(Fml1,n1,n2)';

% Plot it
subplot(332); imagesc(xp1,xp2,Fml0);
axis xy; axis equal; axis tight; box off; title('ML estimate (z=0)'); 
subplot(333); imagesc(xp1,xp2,Fml1); 
axis xy; axis equal; axis tight; box off; title('ML estimate (z=1)'); 


%% Find maximum marginal likelihood estimate for hyperparameters
minlens = 5;  % minimum length scale for each dimension
nxcirc = 2*[n1,n2];  % place circular boundary here
[FmapTot,asdstats,dstruct] = fastASD(xstim,yy(:,jjneur),[n1,n2],minlens,nxcirc);
FmapTot = reshape(FmapTot,n1,n2)'; % spatial map from all data

% Reconstruct prior covariance matrix
[C,Cinv,Sdiag,BB] = rebuildASDcovmatrix(dstruct,1e12);
nsevar = asdstats.nsevar;
x0 = xstim(:,:)*BB; % projected stimulus

% % Check that we can correctly compute posterior mean using learned prior (debugging)
% FmapTst = BB*((x0'*x0+nsevar*diag(1./Sdiag))\(x0'*yy(:,jjneur)));
% FmapTst = reshape(FmapTst,n1,n2)';
% plot([FmapTot(:), FmapTst(:)])  % should be equal
 
Fmap0 = BB*((x0(ii0,:)'*x0(ii0,:)+nsevar*diag(1./Sdiag))\(x0(ii0,:)'*yy(ii0,jjneur)));
Fmap1 = BB*((x0(ii1,:)'*x0(ii1,:)+nsevar*diag(1./Sdiag))\(x0(ii1,:)'*yy(ii1,jjneur)));
Fmap0 = reshape(Fmap0,n1,n2)';
Fmap1 = reshape(Fmap1,n1,n2)';

% compute posterior stdev for both maps
Fstd0 = sqrt(diag(BB*inv(nsevar*x0(ii0,:)'*x0(ii0,:)+diag(1./Sdiag))*BB'));
Fstd1 = sqrt(diag(BB*inv(nsevar*x0(ii1,:)'*x0(ii1,:)+diag(1./Sdiag))*BB'));
Fstd0 = reshape(Fstd0,n1,n2)';
Fstd1 = reshape(Fstd1,n1,n2)';

%% Make plots

subplot(334);imagesc(xp1,xp2,FmapTot);axis image;axis xy; 
title(['all data: cell ', num2str(jjneur)]);
title(['GP estimate: cell ', num2str(jjneur)]);
subplot(335);imagesc(xp1,xp2,Fmap0);axis image;axis xy;title('z=0');
set(gca,'tickdir','out'); 
subplot(336);imagesc(xp1,xp2,Fmap1);axis image;axis xy;title('z=1');
set(gca,'tickdir','out');

% Plot both entire maps
subplot(337); h = plot(xp1, Fmap0, 'b', xp1, Fmap1, 'r--'); 
axis tight; box off; set(gca,'tickdir', 'out');
legend(h([1,n2+1]),'z=0', 'z=1');

%% Find slice with maximal difference
[~,islice1] = max(max(abs(Fmap0'-Fmap1')));

% plot slice with 1SD error bar on Map 0
subplot(338);
plot(xp1, xp1*0, 'k--');  hold on;
ebregionplot(xp1,Fmap0(islice1,:),Fstd0(islice1,:),Fstd0(islice1,:),[.8 .8 1]);
h = plot(xp1,Fmap0(islice1,:), 'b',xp1, Fmap1(islice1,:), 'r--');
set(h(1:2), 'linewidth', 2); set(gca, 'xlim', xp1([1 end])); box off;
xlabel('x position'); ylabel('mean resp'); set(gca,'tickdir','out');
hold off;

% plot slice with 1SD error bar on Map 1
subplot(339);
plot(xp1, xp1*0, 'k--');  hold on;
ebregionplot(xp1,Fmap1(islice1,:),Fstd1(islice1,:),Fstd1(islice1,:),[1 .8 .8]);
h = plot(xp1,Fmap0(islice1,:), 'b',xp1, Fmap1(islice1,:), 'r--');
set(h(1:2), 'linewidth', 2); set(gca, 'xlim', xp1([1 end])); box off;
xlabel('x position'); ylabel('mean resp'); set(gca,'tickdir','out');
hold off;

% Generate figure
set(gcf,'paperposition',[.25 2.5 8 8]);
print('-dpng',sprintf('figs_allinvest_cell%d.png',jjneur));
