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
load data.mat; % CHANGE THIS
x1 = double(X_scope'); clear X_scope
x2 = double(Y_scope'); clear Y_scope

% Get rid of nans
iinan = isnan(x1);
x1(iinan) = [];
x2(iinan) = [];

% Pick a cell number 
prompt={'Enter neuron number to analyze'};
name2 = 'Neuron number to analyze';
defaultans2 = {'1'};
options.Interpreter = 'tex';
answer = inputdlg(prompt,name2,[1 40],defaultans2,options);
cellnum=str2num(answer{1,1});

% Check if the user entered a valid neuron number
totalcell=min(size(alldeltaf_C));
if cellnum <=totalcell
    sprintf('The user entered a valid number')
else
    error('Error. Neuron number entered is not valid')
end

y=alldeltaf_C(cellnum,:);
y=y';

% Preprocess
y(iinan) = []; % remove nans
nsamps = length(x1); % number of time samples

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


%% Compute ASD estimate
fprintf('\n\n...Running GP regression...\n');

% Make dynamics matrix
Ai = spdiags(ones(nsamps,1)*[-exp(-1/(tau_gcamp/dtbin)), 1],-1:0,nsamps,nsamps);
ydeconv = Ai*y;  % deconvolved y

% Run ASD
minlens = 3 ;  % minimum length scale for each dimension
[Fmap,asdstats] = fastASD(xstim,ydeconv,[n1,n2],minlens);
Fmap = reshape(Fmap,n1,n2)';

%% Make plot
ah1 = subplot(322);
imagesc(xp1, xp2, Fmap); 
axis xy; axis equal; axis tight; box off;
colorbar;
title(['GP estimate: cell ', num2str(cellnum)]);
set(gca,'tickdir','out');
pos1 = get(ah1,'position');

% Find min and max slice
[~,islice1] = max(max(Fmap'))
[~,islice2] = min(min(Fmap'))
hold on
plot([0 50], xp2(islice1)*[1 1], 'r');
plot([0 50], xp2(islice2)*[1 1], 'b');
hold off;

% Error bars  (+/- 2SD);
Fci = 1.96*sqrt(reshape(asdstats.Lpostdiag,n1,n2)');

ah2 = subplot(325);
plot(xp1,Fmap(islice2,:), 'b', 'linewidth', 2);
hold on;
ebregionplot(xp1,Fmap(islice2,:),Fci(islice2,:),Fci(islice2));
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
plot(xp1,Fmap(islice1,:),'r', 'linewidth', 2);
hold on;
ebregionplot(xp1,Fmap(islice1,:),Fci(islice1,:),Fci(islice1));
plot(xp1,xp1*0,'k--');
hold off; box off; axis tight;
title('max slice');
xlabel('x position');
ylabel('mean resp');
set(gca,'tickdir','out');
pos2 = get(ah2, 'position');
pos2(3) = pos1(3);
set(ah2,'position',pos2);  


% Generate figure
% set(gcf,'paperposition',[.25 2.5 8 8]);
%print('-dpng',sprintf('figs_Open/cell%d.png',cnums(cellnum)));
%print('-dpdf',sprintf('figs_Open/cell%d.pdf',nums(cellnum)));
