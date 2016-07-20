%% script_160708_inspectMaluData.m
%
% Do simple inspection of tuning properties of new dataset from Malu.

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
yy = yy./std(yy(:)); % normalize by mean std across all yy


%% Let's just inspect the tuning of all other variables

% Variables of interest
% ---------------------
% m1_groom: when  imaging mouse (m1) grooms social target.
% m1_sniff: when imaging mouse (m1) pursues social target.
% m1_pursue: when imaging mouse (m1) pursues the social target.
% m1_allinvest: if imaging mouse is sniff/groom/pursuing the social target.
% m2_invest: if the imaging is mouse is being investigated by the social target. 

% Compute binarized velocity for low, medium or high
vlo = v>quantile(v,.25);  % times at which velocity is above 0.25 quantile
vmed = v>median(v);  % times at which velocity is above median
vhi = v>quantile(v,.75);   % times at which velocity is above 0.75 quantile

varnames = {'m1_groom', 'm1_sniff', 'm1_pursue', 'm1_allinvest', 'm2_invest', 'vlo', 'vmed', 'vhi'};
nvar = length(varnames);

% Loop over variables and compute tuning for each neuron
for jj = 1:nvar
    eval(sprintf('zz=%s;', varnames{jj})); % grab relevant variable
    
    ii0 = (zz==0); % time indices with 0
    ii1 = (zz==1); % time indices with 1

    % compute tuning curve (mean)
    tc0 = mean(yy(ii0,:));
    tc1 = mean(yy(ii1,:));
    
    % Compute standard error
    se0 = std(yy(ii0,:))/sqrt(sum(ii0));
    se1 = std(yy(ii1,:))/sqrt(sum(ii1));
    
    % make plot
    subplot(4,2,jj);
    h = plot(1:nneur, tc0, 1:nneur, tc1); axis tight;
    set(h,'linewidth', 2);
    hold on; 
    ebregionplot(1:nneur,tc0,se0,se0,[.8 .8 1]); 
    ebregionplot(1:nneur,tc1,se1,se1,[1 .8 .8]);
    h = get(gca,'children');
    set(gca,'children',h([4 1 3 2]));
    hold off;
    h=title(varnames{jj});
    set(h,'interpreter','none')
    
    if jj>6
        xlabel('neuron #');
    end
    
end
