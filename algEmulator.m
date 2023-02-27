classdef algEmulator
    %ALGEMULATOR ALG GPR-based emulation
    %   ALGEMULATOR object allows training an emulator from ALG LUT data
    %   files based on the combination of PCA (for dimensionality
    %   reduction) and GPR (for statistical regression).
    %   Once trained, the object stores the PCA transformation matrices for
    %   reconstruction of the original spectra as well as the GPR objects
    %   to use in data processing applications.
    %   
    %   See also: GPRFIT, PREDICT, PCA
    
    properties
        lutFile % Original ALG LUT file (training dataset)
        wvl     % Wavelengths
        I0      % Solar irradiance at 1 AU (in mW/m2/sr/nm)
        Xtrain   % Input RTM values in training LUT data
        varnames % Input RTM (and emulator) variables
        outnames % Output RTM (and emulator) transfer functions
        SZA     % Solar zenith angles associated to Xtrain
        coeff   % PCA coefficients
        mu      % PCA estimated mean
        gpr     % GPR model(s)
        conf    % Matlab structure containing the configuration of emulators
                % It can include the following fields:
                % method: method to run the emulators: 'alg' (default) or
                %        'matlab' (recommended for validation).
                % kernel: kernel function for GPR: 'squaredexponential' 
                %         (default) or 'ardsquaredexponential'.
                % dimReduction: method for dimensionality reduction: 'pca'
                %         (default) for PCA decomposition or 'none'. Note
                %         that the 'none' option can be computationally
                %         expensive since one GPR is trained for each
                %         output wavelenght.
                % numCmp: number of components after dimensionality reduction.
                %         If dimReduction='pca', this value can be defined 
                %         as a scalar (=20, default value) or a vector 
                %         defining the number of PCA components for each 
                %         atmospheric transfer function.
                %         If dimReduction='none', this value is
                %         automatically replaced by the number of
                %         wavelengths unless multifidelify.method='linear',
                %         in which case numCmp=2 by definition.
                % multifidelity: Structure defining the configuration of a
                %         multifidelity emulator. The following parameters
                %         can be configured:
                %         -LFregressor: low-fidelity (LF) regression model
                %         among the options 'polyfit' (default, Ndim surface 
                %         fitting by 2nd order polynomial) or 'gpr'
                %         (with a squaredexponential kernel and 2 PCAs).
                %         -method:
                %           'delta' (default): a GP emulator is trained for
                %           the difference between the real (training) and
                %           the LF regression data. Note that this
                %           difference is spectral (multioutput)!
                %           'gain': a GP emulator is trained for ratio
                %           between the real (training) and the LF 
                %           regression data. Note that this ratio is 
                %           spectral (multioutput)!
                %           'linear': a linear regression between the LF
                %           regression and the real (training) data is
                %           calculated (Yhf = A*Ylf+B). The GP emulator is
                %           trained for the values of the fitting
                %           coefficients A and B. Note that this
                %           coefficients are scalars!
                %         -nLayers: integer value (>0) defining the number
                %           of multi-fidelity layers. This allows to 
                %           execute the lowest fidelity with LFregressor 
                %           and nLayers-1 recursive GPs with the same
                %           configuration as in conf
                %         If multifidelity is not used if is this parameter
                %         is empty, non-defined or any other non-valid value.
                % featSelection - Feature selection flag: true or false
                %         (default). If featSelection=true, a set of 
                %         features are selected for each transfer function 
                %         and applied for all the PCA components. The 
                %         method is a supervised method based on the 
                %         accuracy of 2nd degree n-D surface fitting
                %         ('polyfit' interpolation method). In case of
                %         multifidelity, the feature selection only applies 
                %         to the higher fidelity model.
        fit     % Coefficients from low fidelity model. Only available if 
                % conf.multifidelity has been defined to 'polyfit' or 'gpr'
                % - polynomial fitting (see myinterpn 'polyfit' option)
                % - GP squared exponential (algEmulator object)
        features% Vector of selected features
    end
    
    properties (Hidden=true)
        gpuGatherFlag = true % true = transfer gpuArray to local workspace
    end
    
    methods
        function obj = algEmulator(lutFile,conf)
            %ALGEMULATOR Construct an instance of this class

            if isfile(lutFile) && strcmp(lutFile(end-2:end),'.h5')
                obj.lutFile = lutFile;
                hdr = readLUThdr(obj.lutFile);
                obj.wvl      = hdr.wvl;
                obj.I0       = hdr.I0;
                obj.Xtrain   = hdr.LUTheader';
                obj.varnames = hdr.varnames;
                obj.outnames = hdr.outnames;
                obj.SZA      = hdr.SZA;
            else
                msgbox(['Input file is not valid. Please select a .h5 ',...
                    'LUT file generated with ALG.'],'algEmulator: not valid file','error')
            end
            if exist('conf') && isstruct(conf)
                fn = fieldnames(conf);
                confopt = {'method','kernel','dimReduction','numCmp',...
                    'multifidelity','featSelection'};
                % Remove non-valid fields:
                id = ~contains(fn,confopt);
                if any(id), conf = rmfield(conf,fn{id}); end

                % Check validity of configuration structure and replace 
                % non-valid inputs by default values:
                methodopt = {'matlab','alg'};
                kernelopt = {'squaredexponential','ardsquaredexponential'};
                dimredopt = {'pca','none'};
                if ~any(contains(fn,'method')) || ~ischar(conf.method) || ~contains(conf.method,methodopt)
                    conf.method = 'alg';
                end
                if ~any(contains(fn,'kernel'))  || ~ischar(conf.kernel) || ~contains(conf.kernel,kernelopt)
                    conf.kernel = 'squaredexponential';
                end
                if ~any(contains(fn,'dimReduction'))  || ~ischar(conf.dimReduction) || ~contains(conf.dimReduction,dimredopt)
                    conf.dimReduction = 'pca';
                end
                if strcmp(conf.dimReduction,'pca') && (~any(contains(fn,'numCmp')) || ~isnumeric(conf.numCmp) || any(conf.numCmp<1))
                    conf.numCmp = 20;
                elseif strcmp(conf.dimReduction,'pca') && conf.numCmp>size(obj.Xtrain,1)-1
                    conf.numCmp=size(obj.Xtrain,1)-1;
                end
                
                if ~any(contains(fn,'multifidelity'))
                    conf.multifidelity = false;
                elseif isstruct(conf.multifidelity)
                    fn2 = fieldnames(conf.multifidelity);
                    lfmethod = {'delta','gain','linear'};
                    if ~any(contains(fn2,'method')) || ~ischar(conf.multifidelity.method) || ~contains(conf.multifidelity.method,lfmethod)
                        conf.multifidelity.method = 'delta';
                    end
                    lfregressor = {'polyfit','gpr'};
                    if ~any(contains(fn2,'LFregressor')) || ~ischar(conf.multifidelity.LFregressor) || ~contains(conf.multifidelity.LFregressor,lfregressor)
                        conf.multifidelity.LFregressor = 'polyfit';
                    end
                    if ~any(contains(fn2,'nLayers')) || ~isnumeric(conf.multifidelity.nLayers) || any(conf.multifidelity.nLayers<1)
                        conf.multifidelity.nLayers = 1;
                    elseif conf.multifidelity.nLayers-round(conf.multifidelity.nLayers)~=0
                        conf.multifidelity.nLayers = ceil(conf.multifidelity.nLayers);
                    end
                    % Replace dimensionality reduction if 'linear' 
                    % multifidelity method is selected
                    if strcmp(conf.multifidelity.method,'linear')
                        conf.dimReduction='none';
                        conf.numCmp = 2;
                    end
                else
                    conf.multifidelity = false;
                end
                if ~any(contains(fn,'featSelection'))
                    conf.featSelection = false;
                end
                
                if gpuDeviceCount>0
                    % GPU is available so maybe we need to gather data from
                    % GPU to CPU:
                    obj.gpuGatherFlag = true;
                end
                
                % Train emulator:
                obj.conf = conf;
                obj = obj.trainEmulator();
            end
        end
        
        function obj = trainEmulator(obj)
            %TRAINEMULATOR Trains an emulator
            %   Train GPR models for selected LUT data using PCA as
            %   dimensionality reduction and selecting the first numCmp 
            %   components
            
            % Dimensions:
            l = numel(obj.wvl);      % number wavelengths
            v = numel(obj.outnames); % number transfer functions
            n = size(obj.Xtrain,1);  % number data points
            d = size(obj.Xtrain,2);  % input dimension
            
            % Read LUT data:
            Y = double(h5read(obj.lutFile,'/LUTdata'));
            
            wbtitle = 'Training GRP emulator';
            if isstruct(obj.conf.multifidelity)
                % Construct multifidelity model
                mfLayer = obj.conf.multifidelity.nLayers;
                [obj,Y,l] = constructMfModel(obj,Y,l,mfLayer);
                wbtitle = [wbtitle,...
                    ' (layer: ',num2str(mfLayer),')'];
            end
            
            % Scale all inputs so that they are between 0 and 1:
            % TO DO: alternatively we can scale by removing mean and 
            % dividing by standard deviation but this mean and std should
            % be saved for later prediction
            Xt = (obj.Xtrain - min(obj.Xtrain))./(max(obj.Xtrain) - min(obj.Xtrain));
			
			% Vector of selected features. By default it is all the 
			% features (i.e. input variables) but if conf.featSelection=true
			% it contains the selected features in decreasing order of 
            % importance
            obj.features = cell(1,d);
            obj.features(1:d) = {1:d};
            
            if numel(obj.conf.numCmp)~=1 && numel(obj.conf.numCmp)~=v
                msgbox(['The number of components should be a scalar be or ',...
                    'a vector of length ',num2str(v),'. Please select a valid option.'],'algEmulator: training error','error')
                return
            elseif any(obj.conf.numCmp>l)
                msgbox(['The number of components should be lower than ',...
                    num2str(l),'. Please select valid number of values.'],'algEmulator: training error','error')
                return
            %elseif numel(obj.conf.numCmp)==1
            %    obj.conf.numCmp = obj.conf.numCmp*ones(v,1);
            end
            
            if strcmp(obj.conf.dimReduction,'pca')
                % Memory allocation for variables used only in case of PCA 
                % dimensionality reduction
                obj.mu    = nan(v,l);
                if numel(obj.conf.numCmp)==1
                    obj.coeff = nan(obj.conf.numCmp,l,v);
                else
                    % Each transfer function uses a different number of
                    % components
                    obj.coeff = cell(1,v);
                end
            end
            h = waitbar(0,'Training emulator. Please wait...',...
                'Name',wbtitle);
            status = 0;
            for i=1:v
                Ytrain = Y(1+(i-1)*l:i*l,:)';
                switch obj.conf.dimReduction % Apply dimensionality reduction
                    case 'pca' % Calculate PCA
                        warning off
                        [coeff,score,~,~,~,obj.mu(i,:)] = pca(Ytrain);
                        if numel(obj.conf.numCmp)==1
                            obj.coeff(:,:,i) = coeff(:,1:obj.conf.numCmp)';
                        else
                            obj.coeff{i} = coeff(:,1:obj.conf.numCmp(i))';
                        end
                        warning on
                    case 'none' % No dim. reduction applied
                        score = Ytrain;
                end
                
                if obj.conf.featSelection
                    % Apply feature selection
                    h.Children.Title.String = ...
                        'Selecting features in input variable space...';
                    obj.features{i} = featureSelection(obj,Xt,Ytrain');
                end
                
                % Train GPR model for each PCA component
                basisfcn  = 'constant'; %'none' or 'costant' (default)
                if numel(obj.conf.numCmp)==1
                    numCmp = obj.conf.numCmp; totCmp = v*numCmp;
                else
                    numCmp = obj.conf.numCmp(i); totCmp = sum(obj.conf.numCmp);
                end
                for j=1:numCmp
                    status = status+1;
                    waitbar(status/totCmp,h,['Training emulator for ',...
                        obj.outnames{i},'. Component ',num2str(j),' of ',num2str(numCmp)]);
                    sigma = 0.1*std(score(:,j))/sqrt(2);
                    if j>1 && ~strcmp(obj.conf.kernel,'ardsquaredexponential')
                        switch obj.conf.method
                           case 'matlab'
                               sigmaL0 = obj.gpr{i,j-1}.KernelInformation.KernelParameters(1:end-1);
                           case 'alg'
                               sigmaL0 = obj.gpr{i,j-1}.sigma(1:end-1);
                        end
                        sigmaF0 = std(score(:,j))/sqrt(2);
                        gpr = fitrgp(Xt(:,obj.features{i}),score(:,j),...
                            'KernelFunction',obj.conf.kernel,'Sigma',sigma,...
							'BasisFunction',basisfcn,...
                            'KernelParameters',[sigmaL0;sigmaF0]);
                    else
                        gpr = fitrgp(Xt(:,obj.features{i}),score(:,j),...
                            'KernelFunction',obj.conf.kernel,'Sigma',sigma,...
							'BasisFunction',basisfcn);
                    end
                    switch obj.conf.method
                        case 'matlab'
                            obj.gpr{i,j} = gpr;
                        case 'alg'
                            % First all sigmaL (LenghScales) then sigmaF:
                            obj.gpr{i,j}.sigma = gpr.KernelInformation.KernelParameters;
                            obj.gpr{i,j}.Alpha = gpr.Alpha;
                            obj.gpr{i,j}.Beta  = gpr.Beta;
                    end
                end
            end
            close(h)
        end
        
        function Y = emulate(obj,X)
            %EMULATE Runs emulator for to predict inputs
            %   Runs trained GPR models for to predict RTM output spectra
            %   at new input conditions
            
            % Dimensions:
            [n,d] = size(X);            % number of input conditions & input dimension
            no    = size(obj.Xtrain,1); % number of training conditions
            v = numel(obj.outnames);    % number transfer functions
            l = numel(obj.wvl);         % number wavelength
            
            % Check if matrices size are very large and try to do
            % calculations in GPU
            gpuFlag = false; % False by default
            if strcmp(obj.conf.method,'alg') && gpuDeviceCount>0
                gpuMemory = gpuDevice; % information about GPU
                gpuMemory = gpuMemory.AvailableMemory; % available GPU memory
                % We allow using GPU if matrix size is no more than 85%
                % of the available GPU memory and are larger than 1Gb
                mSize = 8*max(no*n*d,2*n*l*v);
                if mSize>=1e9 && mSize<=0.85*gpuMemory && false
                    gpuFlag = true;
                end
            end
            
            % Check multi-fidelity and extract low fidelity data:
            mf_flag = false;
            if isstruct(obj.conf.multifidelity)
                mf_flag = true;
                mfLayer = obj.conf.multifidelity.nLayers;
                % Low-fidelity (LF) model prediction:
                if mfLayer==1 % Lowest layer of the multi-fidelity
                    switch obj.conf.multifidelity.LFregressor
                        case 'polyfit' % Polynomial n-dim surface fitting
                            %Ylf = myinterpn(X','polyfit',obj.fit)';
                            Ylf = obj.emupolyfit(X',obj.fit,'execution',gpuFlag);
                            Ylf = reshape(Ylf,[n l v]);
                        case 'gpr' % Multi-fidelity GP
                            Ylf = obj.fit.emulate(X);
                    end
                else % Intermediate (or top) layer of the multi-fidelity
                    Ylf = obj.fit.emulate(X);
                end
                % Memory allocation:
                if isequal(size(Ylf),[n l v]), Y = Ylf;
                elseif gpuFlag,                Y = nan(n,l,v,'gpuArray');
                else,                          Y = nan(n,l,v);
                end
            else % No multi-fidelity
                % Memory allocation:
                if gpuFlag, Y = nan(n,l,v,'gpuArray');
                else,       Y = nan(n,l,v);
                end
            end

            % Scale all inputs so that they are between 0 and 1:
            Xt = (obj.Xtrain - min(obj.Xtrain))./(max(obj.Xtrain) - min(obj.Xtrain));
            %Xt = obj.Xtrain;
            X  = (X - min(obj.Xtrain))./(max(obj.Xtrain) - min(obj.Xtrain));
            if gpuFlag, Xt = gpuArray(Xt); X = gpuArray(X); end
            
            % Precalculate distances between training and query points:
            if strcmp(obj.conf.method,'alg')
                switch obj.conf.kernel
                    case 'squaredexponential'
                        if obj.conf.featSelection
                            %[D2aux,gpuFlag] = obj.euclideanDistance(Xt,X,false);
                            % It's more efficient to operate with 2D
                            % matrices than 3D matrices
                            %D2aux = reshape(D2aux,no*n,d);
                            D2aux = nan(no,n,v);
                            for i=1:v % For each transfer function
                                idx = obj.features{i};
                                D2aux(:,:,i) = ...
                                    obj.euclideanDistance(Xt(:,idx),X(:,idx),true);
                            end
                            D2aux = reshape(D2aux,no*n,v);
                        else
                            D2 = obj.euclideanDistance(Xt,X,true);
                        end 
                    case 'ardsquaredexponential'
                        if obj.conf.featSelection
                            D2aux = obj.euclideanDistance(Xt,X,false);
                            D2aux = reshape(D2aux,no*n,d);
                        else
                            D2 = obj.euclideanDistance(Xt,X,false);
                            D2 = reshape(D2,no*n,d);
                        end
                end
            end
            
            % Memory allocation
            if numel(obj.conf.numCmp)==1
                if gpuFlag, score = nan(n,obj.conf.numCmp,v,'gpuArray');
                else,       score = nan(n,obj.conf.numCmp,v);
                end
            end
            
            % Now run GP prediction
            for i=1:v % For each transfer function
                if numel(obj.conf.numCmp)==1, numCmp = obj.conf.numCmp;
                else,                         numCmp = obj.conf.numCmp(i);
                end
                if obj.conf.featSelection && strcmp(obj.conf.method,'alg')
                    % If feature selection, then we calculate the distance
                    % for the specific selected features:
                    switch obj.conf.kernel
                        case 'squaredexponential'
                            %D2 = sum(D2aux(:,obj.features{i}),2);
                            D2 = D2aux(:,i);
                            D2 = reshape(D2,no,n);
                        case 'ardsquaredexponential'
                            D2 = D2aux(:,obj.features{i});
                    end
                end
                % Memory allocation
                if numel(obj.conf.numCmp)~=1
                    if gpuFlag, score = nan(n,obj.conf.numCmp(i),'gpuArray');
                    else,       score = nan(n,obj.conf.numCmp(i));
                    end
                end
                % Run GPR model for each PCA component
                if strcmp(obj.conf.method,'alg') && strcmp(obj.conf.kernel,'ardsquaredexponential')
                    % In case of ardsquaredexponential kernel, it's more
                    % convenient to 
                    sigma = nan(d+1,numCmp);
                    for j=1:numCmp
                        sigma(:,j) = max(obj.gpr{i,j}.sigma.^2,1e-12);
                    end
                    KNMaux = D2*(1./sigma(1:end-1,:));
                end
                for j=1:numCmp
                    switch obj.conf.method
                        case 'matlab'
                            if numel(obj.conf.numCmp)==1
                                score(:,j,i) = predict(obj.gpr{i,j},X(:,obj.features{i}));
                            else
                                score(:,j) = predict(obj.gpr{i,j},X(:,obj.features{i}));
                            end
                        case 'alg'
                            sigma = max(obj.gpr{i,j}.sigma.^2,1e-12);
                            switch obj.conf.kernel
                                case 'squaredexponential'
                                    KNM = sigma(end).*exp(-0.5*D2/sigma(1));
                                case 'ardsquaredexponential'
                                    % Matlab's predict function propose a
                                    % loop for every dimension but a direct
                                    % matrix multiplication seems faster.
                                    %KNM = D2*(1./sigma(1:end-1)); % slower old version
                                    % However, it is faster to make this
                                    % multiplication once for all the PCA
                                    % components
                                    KNM = sigma(end).*exp(-0.5*KNMaux(:,j));
                                    KNM = reshape(KNM,[no n]);
                            end
                            if numel(obj.conf.numCmp)==1
                                score(:,j,i) = KNM'*obj.gpr{i,j}.Alpha + obj.gpr{i,j}.Beta;
                            else
                                score(:,j) = KNM'*obj.gpr{i,j}.Alpha + obj.gpr{i,j}.Beta;
                            end
                    end
                end
                if strcmp(obj.conf.dimReduction,'pca') && numel(obj.conf.numCmp)~=1
                    % Reconstruct original spectra by inverse PCA:
                    Y(:,:,i) = score * obj.coeff{i} + obj.mu(i,:);
                elseif strcmp(obj.conf.dimReduction,'none') && ~mf_flag
                    Y(:,:,i) = score;
                end
            end
            if strcmp(obj.conf.dimReduction,'pca') && numel(obj.conf.numCmp)==1
                % All transfer functions have the same number of PCA
                % components. Matlab pagemtimes function allows doing the 
                % calculation much faster:
                Y = pagemtimes(score,obj.coeff) + reshape(obj.mu',[1,l,v]);
            end
            
            if mf_flag
                % Apply multi-fidelify:
                switch obj.conf.multifidelity.method
                    case 'delta' % Substract LF model from data
                        Y = Y + Ylf;
                    case 'gain' % Divide LF model from data
                        Y = Y.*Ylf;
                    case 'linear' % Calculate linear regresion between LF model and data
                        for k1=1:v
                            for k2=1:n % For each training point
                                A = [ones(l,1),Ylf(k2,:,k1)'];
                                Y(k2,:,k1) = reshape(A*score(k2,:,k1)',[1 l 1]);
                            end
                        end
                end
            end
            if gpuFlag && obj.gpuGatherFlag
                Y = gather(Y);
            end
        end
        
        function [err,errRho] = pcaError(obj,numCmp,flag,rho)
            % PCAERROR Evaluation of reconstruction error by PCA
            %   dimensionality reduction
            %   [ERR,ERRRHO] = PCAERROR(numCmp) evaluates the reconstruction 
            %   error of each atmospheric transfer function (ERR) and on
            %   the inverted surface reflectance (ERRRHO) as function of 
            %   the number of PCA components in the numCmp vector. 
            %   PCAERROR(...,FLAG,RHO) allows to activate (FLAG=True)
            %   plotting the errors. RHO is the reference surface
            %   reflectance value (default 0.2) used to calculate the 
            %   inversion errors.
            warning off
            if nargin==2,     flag = false;
            elseif nargin==3, rho = 0.2;
            end
            if flag
                fn = length(findobj('type','figure'));
                figure(fn+1), figure(fn+2)
            end
            % Read LUT data:
            Y = double(h5read(obj.lutFile,'/LUTdata'));
            hdr = readLUThdr(obj.lutFile);
            
            l = numel(hdr.wvl); % number wavelenghts
            v = numel(hdr.outnames); % number transfer functions
            
            if any(numCmp>l)
                msgbox(['The number of PCA components should be lower than ',...
                    num2str(l),'. Please select valid number of PCA values.'],'algEmulator: PCA analysis error','error')
                return  
            end
            
            err = nan(l,v,numel(numCmp)); errRho = nan(l,numel(numCmp));
            % Evaluate errors on each transfer function
            for i=1:v
                Ytrain = Y(1+(i-1)*l:i*l,:)';                
                % Calculate PCA:
                [coeff,score,~,~,~,mu] = pca(Ytrain);
                % Reconstruct original spectra:
                for j=1:numel(numCmp)
                    Ypca = score(:,1:numCmp(j)) * coeff(:,1:numCmp(j))' + mu;
                    e = 100*abs(Ytrain-Ypca)./Ytrain;
                    % We calculate the average error from the all the data
                    err(:,i,j) = mean(e,'omitnan');
                end
                if flag
                    figure(fn+1)
                    colororder(turbo(numel(numCmp)))
                    subplot(3,2,i)
                    semilogy(obj.wvl,squeeze(err(:,i,:))), grid on
                    title(obj.outnames{i})
                    xlabel('Wavelength (nm)'), ylabel('Rel. error (%)')
                    %ylim([0 10^ceil(log(max(prctile(squeeze(err(:,i,:)),98))))])
                    xlim([min(obj.wvl),max(obj.wvl)])
                    if i==v
                        legend(cellfun(@num2str,num2cell(numCmp),...
                            'UniformOutput',false))
                    end
                end
            end
            
            errRho = nan(l,numel(numCmp));
            % Evaluate errors on inverted surface reflectance:
            % First we calculate TOA radiance
            Ltoa = Y(1:l,:) + (1/pi)*...
                (Y(1+l:2*l,:).*cosd(obj.SZA) + Y(1+2*l:3*l,:)).*...
                (Y(1+4*l:5*l,:) + Y(1+5*l:6*l,:))*rho./...
                (1-Y(1+3*l:4*l,:)*rho);
            for i=1:v
                Ytrain = Y(1+(i-1)*l:i*l,:)';                
                % Calculate PCA:
                [coeff(:,:,i),score(:,:,i),~,~,~,mu(i,:)] = pca(Ytrain);
            end
            % Now perform atmospheric correction:
            for j=1:numel(numCmp)
                L0 = (score(:,1:numCmp(j),1) * coeff(:,1:numCmp(j),1)' + mu(1,:))';
                Etot = (score(:,1:numCmp(j),2) * coeff(:,1:numCmp(j),2)' + mu(2,:))'.*cosd(obj.SZA)+...
                    (score(:,1:numCmp(j),3) * coeff(:,1:numCmp(j),3)' + mu(3,:))';
                Ttot = (score(:,1:numCmp(j),5) * coeff(:,1:numCmp(j),5)' + mu(5,:))'+...
                    (score(:,1:numCmp(j),6) * coeff(:,1:numCmp(j),6)' + mu(6,:))';
                S = (score(:,1:numCmp(j),4) * coeff(:,1:numCmp(j),4)' + mu(4,:))';
                
                A = pi*(Ltoa-L0).*S + Etot.*Ttot;
                rhoInv = pi*(Ltoa-L0)./A;
                e = 100*abs(rho-rhoInv)./rho;
                % We calculate the average error from the all the data
                errRho(:,j) = mean(e,2,'omitnan');
            end
            if flag
                figure(fn+2)
                colororder(turbo(numel(numCmp)))
                semilogy(obj.wvl,errRho), grid on
                title('Error in inverted reflectance')
                xlabel('Wavelength (nm)'), ylabel('Rel. error (%)')
                %ylim([0 10^ceil(log(max(prctile(squeeze(err(:,i,:)),98))))])
                xlim([min(obj.wvl),max(obj.wvl)])
                legend(cellfun(@num2str,num2cell(numCmp),'UniformOutput',false))
            end
        end
        
        function gsa = sensitivityAnalysis(obj)
            % Scale all inputs so that they are between 0 and 1:
            Xt = (obj.Xtrain - min(obj.Xtrain))./(max(obj.Xtrain) - min(obj.Xtrain));
            Xt = obj.Xtrain;
            Y = double(h5read(obj.lutFile,'/LUTdata'));
            l = numel(obj.wvl); nvars = numel(obj.outnames);
            warning off
            ii = 1:size(Xt,2); % Features vector
            
            gsa = nan(numel(obj.varnames),numel(obj.wvl),nvars);
            for v=1:nvars
                Yt = Y(1+(v-1)*l:v*l,:);
                % Selected features vector and associated errors to added
                % features:
                features = nan(size(Xt,2),numel(obj.wvl));
                featErr = features;
                % Start ranking of features
                for k=1:numel(obj.varnames) % for each new feature
                    err = nan(numel(obj.varnames)-k+1,numel(obj.wvl));
                    iblock = 1; nblocks = 1;
                    % Since specific spectral regions are sensitive to
                    % specific variables, the full spectral range is divided
                    % into blocks (i.e. collection of features from previous
                    % step)
                    while iblock<=nblocks
                        % For each block, calculate the errors associated to
                        % adding a new feature
                        if k==1 % First iteration
                            sel = []; vars = ii; iwvl = 1:numel(obj.wvl);
                        else
                            % Calculate the blocks by extracting the unique
                            % values of the selected featues until the step k-1
                            blocks = unique(features(1:k-1,:)','rows');
                            [auxblk,ia,ic] = unique(sort(blocks,2),'rows');
                            nblocks = size(auxblk,1);
                            sel = auxblk(iblock,:);
                            vars = ii;
                            % Deterime the relevant wavelengths for the current
                            % block:
                            iwvl = true(1,numel(obj.wvl));
                            aux  = blocks(ic==ic(ia(iblock)),:);
                            for kk=1:size(aux,2)
                                vars(vars==aux(1,kk)) = [];
                                if size(aux,1)>1
                                    iwvl = iwvl & logical(sum(features(kk,:)==aux(:,kk)));
                                else
                                    iwvl = iwvl & features(kk,:)==aux(:,kk);
                                end
                            end
                        end
                        % Evaluate the interpolation error by adding an extra
                        % feature on top of the already selected in the current
                        % block
                        for i=1:numel(vars)
                            X = Xt(:,[sel,vars(i)])';
                            out = myinterpn(X,Yt,X,'polyfit');
                            % For all the validation points, calculate the
                            % average relative error between true and
                            % interpolated transfer functions:
                            %aux = 100*abs(Yt-out)./Yt;
                            aux = abs(Yt-out);
                            aux(isinf(aux)) = nan; % replace inf by nan
                            aux = mean(aux,2,'omitnan');
                            aux(isinf(aux) | isnan(aux)) = 0;
                            err(i,iwvl) = aux(iwvl)';
                        end
                        % Select feature that gives the minimum relative error
                        % for the current block (i.e. wavelengths)
                        if k<numel(obj.varnames)
                            [featErr(k,iwvl),j] = min(err(:,iwvl));
                            features(k,iwvl) = vars(j);
                        else
                            featErr(k,iwvl) = err(:,iwvl);
                            features(k,iwvl) = vars;
                        end
                        iblock = iblock+1;
                    end
                end
                % For reference, the "no features" condition corresponds to
                % calculate the error with respect to the average spectrum
                %aux = 100*abs(Yt-mean(Yt,2))./Yt;
                aux = abs(Yt-mean(Yt,2));
                aux(isinf(aux)) = nan; % replace inf by nan
                aux = mean(aux,2,'omitnan'); aux(isinf(aux)) = 0;
                featErr = [aux';featErr];
                % The GSA is the normalized difference when adding a new
                % feature:
                w = abs(diff(featErr)); w = 100*w./sum(w); w(isnan(w))=0;
                auxgsa = nan(numel(obj.varnames),numel(obj.wvl));
                for j=1:numel(obj.wvl)
                    auxgsa(features(:,j),j) = w(:,j);
                end
                gsa(:,:,v) = auxgsa;
            end
        end
        
    end
    
    methods (Access = private)
        function features = featureSelection(obj,Xt,Yt)
            %FEATURESELECTION Feature Selection algortihm
            %   Ranks the input variables by importance based on the
            %   accuracy of a polynomial fitting, recursively adding more
            %   input variables. Once ranked, the Spectral Information
            %   Criterion (SIC) method is applied to automatically select
            %   the minimum number of features.
            warning off
            ii = 1:size(Xt,2); % Features vector
            
            features = []; % Selected features vector
            featErr = nan(1,size(Xt,2)); % Associated errors to features
            k = 1;
            % Start ranking of features
            while numel(ii)>0
                err = nan(numel(ii),1);
                for i=1:numel(ii)
                    X = Xt(:,[features,ii(i)])';
                    out = myinterpn(X,Yt,X,'polyfit');
                    % We calculate the average relative error between
                    % true and interpolated transfer functions:
                    aux = mean(100*abs(Yt-out)./Yt,2); aux(isinf(aux)) = 0;
                    err(i) = mean(aux(aux<1000),'omitnan');
                end
                [featErr(k),j] = min(err);
                features(k) = ii(j);
                ii(j) = []; k = k+1;
            end
            % Now we select the number of features based on the SIC method:
            idfeat = obj.SICmethod(featErr,98.5);
            features = sort(features(1:idfeat));
        end
        
        function ID = SICmethod(obj,V,varargin)
            %SICMETHOD Spectral Information Criterion (SIC) method
            %   Applies the SIC method for automatic feature selection.
            %   [ID,CUMFUM] = SICMETHOD(V) provides the index ID of 
            %   selected features ranked based on the input error in V.
            %   The features are selected with a default for a confidence
            %   interval of of 95%. SICMETHOD(V,CI) allows the user to 
            %   choose the confidence interval (CI), in %, between 50% and
            %   100%. SICMETHOD(V,CI,N) allows to set the number of 
            %   Montecarlo executions for statistical representativity of 
            %   the results (by default N=1e5).
            %   Created by Dr. Luca Martino (Univ. Rey Juan Carlos I)
            if nargin<3
                CI = 0.95; % default confidence interval
            else
                CI = varargin{1}/100;
                if CI<0.5,   CI = 0.5; % minimum allowed value
                elseif CI>1, CI = 1;   % maximum allowed value
                end
            end
            if nargin<4, N = 1e5; % default number of montecarlo runs
            else, N = varargin{2}; if N<1000, N = 1000; end
            end
            par = 1:length(V);
            % Calculation of maximum eta value (note: the bigger eta_max is
            % the larger should be the value of N. Typical N values range
            % from 1e5 to 1e6 without a major change in code performance
            % nor accuracy of the results.
            aux_amount = (V(1)-V(2:end))./(1:length(V)-1);
            eta_max = max(aux_amount);
            % Run montecarlo simulations
            % eta0 is a random value with homogeneous distribution between 
            % 0 and eta_max:
            eta = 0 + eta_max*rand(N,1);
            eta = V + eta*(par-1); % Cost function (error + penalizacion modelo)
            eta = eta==min(eta,[],2);
            % Calculation of the position for each montecarlo run
            pos = nan(1,N);
            for k=1:N
                posaux = find(eta(k,:));
                pos(k) = posaux(1)-1;
            end
            % Compute PMF
            wn = nan(1,max(par));
            for i=1:max(par)
                pos2 = find(pos==(i-1));
                wn(i) = length(pos2); % weights
            end
            wn = wn./sum(wn); % Normalized weights
            % Now we calculate the cummulative distribution (CDF) and 
            % determine the elbow position based on CI:
            cumFun  = cumsum(wn);
            pos_elb = find(cumFun>CI);
            ID      = par(pos_elb(1))-1+1;
            % Uncomment below for plotting (only debug!)
            if false
                figure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                subplot(2,1,1), plot(par,V,'o-'), grid on
                xlabel('k'), ylabel('error (%)')
                subplot(2,1,2), stem(par,wn), grid on
                xlabel('k'), ylabel('Weigths (-)')
                figure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                plot(par,cumsum(wn),'o-'), hold on, grid on
                plot([1 length(V)],[CI CI],'k--')
                xlabel('k'), ylabel('SI (-)')
            end
        end
        
        function [obj,Y,l] = constructMfModel(obj,Y,l,mfLayer)
            %CONSTRUCTMFMODEL Construct multifidelity model
            %   This function calculates the low-fidelity model and defines
            %   the magnitude that links two subsequent fidelities
            
            % Calculate regression of low-fidelity (LF) model:
            if mfLayer==1 %lowest-fidelity layer
                switch obj.conf.multifidelity.LFregressor
                    case 'polyfit' % Polynomial n-dimensional surface fitting
                        %[~,obj.fit] = myinterpn(obj.Xtrain',Y,[],'polyfit');
                        %auxY = myinterpn(obj.Xtrain','polyfit',obj.fit);
                        obj.fit = obj.emupolyfit(obj.Xtrain',Y,'train');
                        auxY = obj.emupolyfit(obj.Xtrain',obj.fit,'execution')';
                    case 'gpr' % Multi-fidelity GP
                        % As a LF model we use a simple GP with sqexp kernel
                        % and only 2 PCA components for dimensionality
                        % reduction
                        lf_conf.numCmp = 2; lf_conf.method = 'alg';
                        lf_conf.kernel = 'squaredexponential';
                        lf_conf.multifidelity = false;

                        obj.fit = algEmulator(obj.lutFile,lf_conf);
                        auxY = obj.fit.emulate(obj.Xtrain);
                        obj.fit.gpuGatherFlag = false;
                        auxY = auxY(:,:)';
                end
            else % Intermediate fidelity
                lf_conf.numCmp = obj.conf.numCmp; lf_conf.method = 'alg';
                lf_conf.kernel = obj.conf.kernel;
                lf_conf.multifidelity.LFregressor = ...
                    obj.conf.multifidelity.LFregressor;
                lf_conf.multifidelity.method = ...
                    obj.conf.multifidelity.method;
                lf_conf.multifidelity.nLayers = mfLayer-1;
                
                obj.fit = algEmulator(obj.lutFile,lf_conf);
                auxY = obj.fit.emulate(obj.Xtrain);
                obj.fit.gpuGatherFlag = false;
                auxY = auxY(:,:)';
            end
            % Define multi-fidelity magnitude to train by emulation:
            switch obj.conf.multifidelity.method
                case 'delta', Y = Y - auxY; % Substract LF model from data
                case 'gain',  Y = Y./auxY;  % Divide LF model from data
                case 'linear' % Calculate linear regresion LF model vs data
                    v = numel(obj.outnames); % number transfer functions
                    n = size(obj.Xtrain,1);  % number data points
                    B = nan(2*v,n);
                    for i=1:v % For each transfer function
                        for j=1:n % For each training point
                            A = [ones(l,1),auxY(1+(i-1)*l:i*l,j)];
                            B(1+(i-1)*2:i*2,j) = A\Y(1+(i-1)*l:i*l,j);
                        end
                    end
                    % Redefine variables for compatibility with other
                    % emulation functionalities
                    Y = B; l=2; clear A B
            end
        end
        
        function D2 = euclideanDistance(obj,Xt,X,flag)
            %EUCLIDEANDISTANCE Calculation of Euclidean distance
            %   Calculates the Euclidean distance between training (Xt) and
            %   query (X) points. If flag=true, all dimensions are summed
            %   and D2 is a 2-dim matrix (option for 'squaredexponential'
            %   kernel). Otherwise, the Euclidean distance is calculated
            %   for every dimension separatelly, resulting in a 3-dim
            %   matrix (option for 'ardsquaredexponential' kernel)
            if flag
                D2 = sum(Xt.^2,2) + sum(X.^2,2)' - 2*Xt*X';
            else
                [n,d] = size(X);    % number of input conditions & input dimension
                no    = size(Xt,1); % number of training conditions
                % Matlab pagemtimes function allows doing the calculation
                % much faster:
                Xt = reshape(Xt,no,1,d); X = reshape(X,1,n,d);
                D2 = Xt.^2 + X.^2 - 2*pagemtimes(Xt,X);
                % The code below is the previous version, involving loops
                %if prod([no,n,d])>4e6 && gpuDeviceCount>1
                %    % For large data volume we try to use GPU
                %    gpuFlag = true; D2 = nan(no,n,d,"gpuArray");
                %else % Otherwise, normal matrix
                %    D2 = nan(no,n,d);
                %end
                %for i=1:d % Euclidean distance in each input dimension
                %    D2(:,:,i) = Xt(:,i).^2 + X(:,i).^2' - 2*Xt(:,i)*X(:,i)';
                %end
                
            end
        end
        
        function out = emupolyfit(obj,varargin)
            
            mode = varargin{3};
            order = 2; % hard-coded 2nd order polynomial
            
            switch mode
                case 'train'
                    % polyfit is in training mode
                    X  = varargin{1};  Y  = varargin{2};
                    n = size(X,1);  % n-D input space
                    [k,m] = size(Y);
                    warning off
                    % For illustration purposes, if n=2, the polynomial 
                    % would be:
                    % p(x1,x2;k) = b00 + b10*x1 + p01*x2 + p11*x1*x2 + p20*x1^2 + p02*x2
                    % where x1=X(1,:) and x2=X(2,:) and bxx are the polynomial
                    % coefficients, which are a priori different in every
                    % output dimension (k) (i.e. the matrix B).
                    % The polynomial coefficients are obtained by least-squares
                    % solution of the linear problem Y=A*b, where A is:
                    % A = [ones(m,1) x1 x1^2 x2 x2^2 x1*x2]
                    A = ones(m,1);
                    % First we calculate the individual terms:
                    for d=1:n % dimension input space (X)
                        for o=1:order % order polynomial
                            A = [A, X(d,:)'.^o];
                        end
                    end
                    % And now we add the crossed terms:
                    if n>1
                        for o=2:order
                            comb = nchoosek(1:n,o);
                            for c=1:size(comb,1)
                                A = [A, prod(X(comb(c,:),:))'];
                            end
                        end
                    end
                    %%%
                    % Now we can solve the linear problem by least-squares:
                    out = nan(size(A,2),k);
                    for i=1:k
                        % The least-squares is solved for each of the k outputs
                        y = Y(i,:)';
                        out(:,i) = A\y; % Solution of: Y = A*b
                    end
                    warning on
                case 'execution'%  polyfit method in execution mode
                    XI = varargin{1};  B = varargin{2};
                    gpuFlag = false;
                    if nargin==5, gpuFlag = varargin{4}; end
                    [n,p] = size(XI); % output p points and n-D input space
                    % We have to calculate the matrix A for the query
                    % points:
                    A = ones(p,1);
                    % First we calculate the individual terms:
                    for d=1:n % dimension input space (XI)
                        for o=1:order % order polynomial
                            A = [A, XI(d,:)'.^o];
                        end
                    end
                    % And now we add the crossed terms:
                    if n>1
                        for o=2:order
                            comb = nchoosek(1:n,o);
                            for c=1:size(comb,1)
                                A = [A, prod(XI(comb(c,:),:))'];
                            end
                        end
                    end
                    % Finally we can apply the fitting:
                    if gpuFlag, A = gpuArray(A); B = gpuArray(B); end
                    out = (A*B);
            end
        end
    end
end

