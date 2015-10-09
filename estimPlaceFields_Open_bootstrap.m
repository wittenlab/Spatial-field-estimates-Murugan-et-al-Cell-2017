%% estimPlaceFields_Open.m
%
% Uses Gaussian Processes regression to make nonparametric estimates of
% spatial tuning of Ca imaging responses.

% Set some general paramters
tau_gcamp = 0.7; % timescale of Ca dye impulse response (s)
dtbin = .1;  % assumed time bin size for data (s)

% Grid size for binning spatial locations (speeds up inference)
xybinwidth = 10; % grid spacing for spatial points

% Add needed paths
addpath code_fastASD;
addpath tools_misc;

% Load rat position data
load ../forJonathan; % CHANGE THIS
x1 = double(xscope_recordingtimes); clear xscope_recordingtimes;
x2 = double(yscope_recordingtimes); clear yscope_recordingtimes;

% Possible cell numbers
cnums = [21,28,38,45,60];

% Pick a cell 
cellnum = 1;

% Load it
yname = sprintf('Trace%d',cnums(cellnum));
fprintf('Loading datafile:  %s\n', yname);
eval(sprintf('y=%s;',yname));
nsamps = length(y); % number of time bins

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

%% Compute ASD estimate, just to find hyperparameters
fprintf('\n\n...Running GP regression...\n');

% Make dynamics matrix
Ai = spdiags(ones(nsamps,1)*[-exp(-1/(tau_gcamp/dtbin)), 1],-1:0,nsamps,nsamps);
ydeconv = Ai*y;  % deconvolved y

% Run ASD
minlens = 3 ;  % minimum length scale for each dimension
[Fmap,asdstats,dd] = fastASD(xstim,ydeconv,[n1,n2],minlens);
Fmap = reshape(Fmap,n1,n2)';

%% Now run bootstrap to get CIs

nboot = 100;  % Number of bootstrap samples to draw

% Chunk up stimuli and responses into little contiguous chunks
bootchunk = 1/dtbin;  % 10 bins (1 second) per discrete chunk.
maxi = floor(nsamps/bootchunk);
ychunk = mat2cell(ydeconv(1:maxi*bootchunk),ones(maxi,1)*bootchunk,1);
xchunk = mat2cell(xstim(1:maxi*bootchunk,:),ones(maxi,1)*bootchunk,size(xstim,2));

FmapBoot = zeros(n2,n1,nboot); % initialize

% Do bootstrap 
fprintf('\n...Running bootstrap...\nsample #:\n ');
for jjboot = 1:nboot
    iiboot = randsample(maxi,maxi,'true'); % indices
    xboot = cell2mat(xchunk(iiboot));
    yboot = cell2mat(ychunk(iiboot));
    FmapBoot(:,:,jjboot) = compASDmapestim(xboot,yboot,[n1,n2],asdstats,dd)';
    if mod(jjboot,10)==0
        fprintf(' %d', jjboot);
    end
end
fprintf('\n');

%% Make plots
ah1 = subplot(322);
imagesc(xp1, xp2, Fmap); 
axis xy; axis equal; axis tight; box off;
colorbar;
title(['GP estimate: cell ', num2str(cnums(cellnum))]);
set(gca,'tickdir','out');
pos1 = get(ah1,'position');

% Find min and max slice
[~,islice1] = max(max(Fmap'))
[~,islice2] = min(min(Fmap'))
hold on
plot([0 50], xp2(islice1)*[1 1], 'r');
plot([0 50], xp2(islice2)*[1 1], 'b');
hold off;

% Error bars
alpha = .025; % 95% error bars
FciLO = Fmap-quantile(FmapBoot,alpha,3);
FciHI = quantile(FmapBoot,1-alpha,3)-Fmap;

ah2 = subplot(324);
plot(xp1,squeeze(FmapBoot(islice2,:,:)),'color', .9*[1 1 1]);
hold on;
plot(xp1,Fmap(islice2,:), 'b', 'linewidth', 2);
ebregionplot(xp1,Fmap(islice2,:),FciLO(islice2,:),FciHI(islice2,:));
plot(xp1,xp1*0,'k--');
hold off; box off; axis tight;
title('min slice');
xlabel('x position');
ylabel('mean resp');
set(gca,'tickdir','out');
pos2 = get(ah2, 'position');
pos2(3) = pos1(3);
set(ah2,'position',pos2);

ah2 = subplot(326);

plot(xp1,squeeze(FmapBoot(islice1,:,:)),'color', .9*[1 1 1]);
hold on;
plot(xp1,Fmap(islice1,:),'r', 'linewidth', 2);
ebregionplot(xp1,Fmap(islice1,:),FciLO(islice1,:),FciHI(islice1,:));
plot(xp1,xp1*0,'k--');
hold off; box off; axis tight;
title('max slice');
xlabel('x position');
ylabel('mean resp');
set(gca,'tickdir','out');
pos2 = get(ah2, 'position');
pos2(3) = pos1(3);
set(ah2,'position',pos2);  
