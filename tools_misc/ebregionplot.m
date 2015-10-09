function h = ebregionplot(x, y, low, hi, colr);
%  h = ebregionplot(x,y,low,hi,plstr);
%
%  Plots gray error region on the current set of axes 
%   (positioned underneath the last thing plotted)
%
%  Error region defined by [y-low, y+hi]
%
% Inputs: x, y  - ordinate and abscissa values 
%   low = lower bound for error trace
%   hi = upper bound for error trace
%   colr = color for error region
%
% Outputs: h = handle to region;

if nargin < 5
    colr = .6*[1 1 1]; % default color
end

% Reshape inputs into column vectors
x = x(:); y = y(:);  % make into column vector
low = low(:);
hi = hi(:);

% Query hold state
holdstate = ishold(gca);
hold on;

xx = [x;flipud(x)];
yy = [y-low; flipud(y+hi)];
h = fill(xx,yy,colr);
set(h, 'edgecolor', colr);

% Place error surface under previous plots:
chldrn = get(gca, 'children');
if ~isempty(chldrn)
    set(gca, 'children', chldrn([2,1,3:end]));
end

if ~holdstate
    hold off;
end
