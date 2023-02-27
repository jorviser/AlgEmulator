% Initalize plot parameters:
fontname = 'Helvetica';
fontsize = 23;
fontunits = 'points';
set(0,'DefaultAxesFontName',fontname,'DefaultAxesFontSize',fontsize,'DefaultAxesFontUnits',fontunits,...
     'DefaultTextFontName',fontname,'DefaultTextFontSize',fontsize,'DefaultTextFontUnits',fontunits,...
     'DefaultLineLineWidth',2,'DefaultLineMarkerSize',10,'DefaultLineColor',[0 0 1],'DefaultAxesVisible','on');
%%
addpath(genpath('D:\Mis Documentos D\MAGAC\Libraries'))

%% Initialize PRISMA object with two Images L1 and L2C
conf = struct('user', 'viserjor', ... % Configuration Parameters of PRISMA Class (for SRTM30 database):
        'psw', input('NASA Earthdata password: ',"s"), ...
        'pathSRTM30', 'Auxiliary\SRTM30');
conf.solIrrFile = 'D:\Mis Documentos D\MAGAC\Auxiliary\solar_irradiance\ref_atlas_thuillier3.nc';
% Maybe use ASTER instead (ask Sergio code for Hyplant)
clc
% Reading of PRISMA image
path = 'D:\Mis Documentos D\Modtran LUTGen\Software\LUTGen\UserData\PRISMA\';
l1File  = [path,'20221005\PRS_L1_STD_OFFL_20221005091542_20221005091546_0001.he5'];
img = prisma(l1File,conf); 
img.l2Path = [path,'20221005\PRS_L2C_STD_20221005091542_20221005091546_0001.he5'];
img = img.readMTD(); % Reading Metadata and other parameters
img = img.readAtmo();

img.aot = imresize(img.aot,20);
img.aex = imresize(img.aex,20);
%% Load ALG emulator
load('emu.mat')
I0 = img.convolve(emu.wvl,emu.I0);

%% Load TOA radiance and convert into TOA reflectance
Rtoa = img.readL1();
Rtoa = img.toarad2toarefl(Rtoa,img.sza,img.I0);
rho  = nan(size(Rtoa));

%% Load input data for running the emulators
load('meteo.mat');

bsz = 10; % batch size in number of columns
% i=1;
for i=1:size(Rtoa,2)/bsz
    ind = 1+(i-1)*bsz:i*bsz;
    sza = reshape(img.sza(:,ind),[],1)';
    % Set up input conditions
    X = [reshape(o3(:,ind),[],1), reshape(img.cwv(:,ind),[],1), ...
        reshape(img.aot(:,ind),[],1), reshape(asy(:,ind),[],1), ...
        reshape(img.aex(:,ind),[],1), reshape(ssa(:,ind),[],1), ...
        reshape(img.elev(:,ind),[],1), sza', ...
        reshape(img.raa(:,ind),[],1)];
    % Run emulator
    tic, tf = emu.emulate(X); t1(i) = toc;
    % Resample to PRISMA spectral resolution
    tic
    R0 = img.convolve(emu.wvl,pi*tf(:,:,1)'./(emu.I0.*cosd(sza))); % path reflectance
    TT = img.convolve(emu.wvl,(tf(:,:,2)'.*cosd(sza)+tf(:,:,3)').*(tf(:,:,5)'+tf(:,:,6)')...
        ./(emu.I0.*cosd(sza)));
    S  = img.convolve(emu.wvl,tf(:,:,4)');
    % Invert surface reflectance
    RR = reshape(squeeze(Rtoa(:,ind,:)),size(Rtoa,1)*bsz,[])'-R0;
    rr = RR./(RR.*S + TT);
    t2(i) = toc;
    rho(:,ind,:) = reshape(rr',size(Rtoa,1),bsz,[]);
end

%% Comparison against PRISMA L2C reflectance product
clear Rtoa
rhoref = img.readL2();

fh=figure;
fill([img.wvl;flipud(img.wvl)],...
    [prctile(reshape(rhoref,1e6,[]),10)';flipud(prctile(reshape(rhoref,1e6,[]),90)')],...
    'b','FaceAlpha',0.3)
hold on, grid on
fill([img.wvl;flipud(img.wvl)],...
    [prctile(reshape(rho,1e6,[]),10)';flipud(prctile(reshape(rho,1e6,[]),90)')],...
    'r','FaceAlpha',0.3)
plot(img.wvl,mean(reshape(rhoref,1e6,[])),'b')
plot(img.wvl,mean(reshape(rho,1e6,[])),'--r')
ylim([0 0.55]), xlim([400 2500])
xlabel('Wavelength (nm)'), ylabel('Reflectance (-)')
legend('PRISMA L2C (P_{10/90})','Emulator (P_{10/90})','Mean PRISMA L2C','Mean emulator',...
    'NumColumns',2,'location','southeast')
title('PRISMA L2C vs emulation-based inverted reflectance')

pos = [0.1 0.16 0.86 0.77]; % Default axis position
set(gca,'position', pos);
set(fh,'PaperUnits','centimeters','PaperPosition',[0 0 30 15])
file = 'prisma1';
print(fh,'-depsc',fullfile(pwd,[file,'.eps']))
print(fh,'-dpng',fullfile(pwd,[file,'.png']))
savefig(fh,fullfile(pwd,[file,'.fig']))


fh=figure;
err = 100*abs(rhoref-rho)./rhoref;
err(isinf(err) | isnan(err)) = 0;
plot(img.wvl,mean(reshape(err,1e6,[])))
ylim([10 40]), xlim([400 2500])


