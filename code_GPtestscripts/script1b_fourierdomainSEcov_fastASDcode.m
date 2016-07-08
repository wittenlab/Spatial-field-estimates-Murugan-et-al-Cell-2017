% script1b_fourierdomainSEcov_fastASDcode.m
%
% Same basic elements as script 1, but now by using functions in the code_fastASD directory 
%
% Generates standard SE kernel (dual space) on a grid, and compare to primal
% (Fourier domain) version


%% Generate ASD covariance matrix
nk = 200;  % number of filter coeffs (assumed 1D)
rho = 1;
len = 10;
Casd = mkcov_ASD(len,rho,nk); % prior covariance matrix 

% Compare to diagonalized version w/o zero-padding (circular boundary condition)
opts1.nxcirc = nk;
opts1.condthresh = 1e8;
[Cdiag1,U1] = mkcov_ASDfactored([len;rho],nk,opts1);
Casd_approx1 = U1*diag(Cdiag1)*U1';

% Compare to diagonalized version w sufficient zero-padding
opts2.nxcirc = nk+len*4;
opts2.condthresh = 1e8;
[Cdiag2,U2] = mkcov_ASDfactored([len;rho],nk,opts2);
Casd_approx2 = U2*diag(Cdiag2)*U2';

%% Make plots
subplot(231); imagesc(Casd); axis image;title('ASD covariance');

subplot(232); imagesc(Casd_approx1); axis image; title('fft version w/ circ boundary');
subplot(233); imagesc(Casd_approx2); axis image; title('fft version with padding');

subplot(235); plot(1:nk, Casd(:,1:10)-Casd_approx1(:,1:10));  title('errors');
subplot(236); plot(1:nk, Casd(:,1:10)-Casd_approx2(:,1:10));  title('errors');

subplot(234); plot(1:nk, Casd(:,1),'b',1:nk,Casd_approx1(:,1), 'g--', ...
    1:nk, Casd_approx2(:,1:1), 'r--'); legend('orig', 'circ', 'padded');
