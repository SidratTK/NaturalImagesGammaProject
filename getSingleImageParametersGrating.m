

function [gaborParams] = getSingleImageParametersGrating(imageHSV,imageAxesDeg,rfCenterDeg,rfRadiusDeg,checkRadiusDeg)
imgVal = imageHSV(:,:,3);      % val layer

michConThresh = 0.6;
ovThresh      = 0.75; 
sfLims        = [0.5 8];

% 1. get Michelson contrast in rf patch
[conMichelsonCheckRad] = getMichelCon(imgVal,rfCenterDeg,rfRadiusDeg,imageAxesDeg.xAxisDeg,imageAxesDeg.yAxisDeg);

% 2. get 2d FFT spectra to see any peaks of spatial frequency
[~,radiusVector,imgRadialPSD,oriVarVsF,oriMaxVsF] = getSpectraSingleImageFun(imgVal,rfCenterDeg,checkRadiusDeg,imageAxesDeg);
[~,peakSfs,~,~,~] = getImageSpectraPeaks(radiusVector,imgRadialPSD,oriVarVsF,oriMaxVsF,sfLims);  % 
sfOfInterest= peakSfs;

% 3. if there is high contrast within RF or there is a peak in the SF
% spectrum, look closer at the RF with Gabor filters and find fitting params
if ~isempty(sfOfInterest) || conMichelsonCheckRad>=michConThresh
    if isempty(sfOfInterest), sfOfInterest=[0.5 1 2 4];   % high contrast but no peak in Spectrum
    end
    thetasDeg = (0:22.5:157.5); 
    radbysig  = 3;   % rad is aperature of gauss beyond which values are 0.
    [gaborParams]= getBestGaborFit(imgVal,imageAxesDeg,rfCenterDeg,rfRadiusDeg,sfOfInterest,thetasDeg,radbysig);
    gaborParams.sfOfInterest = sfOfInterest; % the SFs that were checked
    if gaborParams.oriVar >= ovThresh && gaborParams.spatialFreqCPD>=sfLims(1) && gaborParams.spatialFreqCPD<=sfLims(2)
          gaborParams.categoryGrating=true;
    else, gaborParams.categoryGrating=false;
    end
else,     gaborParams.categoryGrating=false;
end

gaborParams.michelsonConInPatch = conMichelsonCheckRad;

end

function [imageDFT,radiusVector,imgRadialPSD,oriVarVsF,oriMaxVsF] = getSpectraSingleImageFun(imgVal,rfCenterDeg,checkRadiusDeg,imageAxesDeg)
        imgVal = imgVal-(mean(imgVal(:)));   % remove dc if any? 
        % gaussian mask to only keep the features at center & fade the rest with smooth edges.
        params= [rfCenterDeg checkRadiusDeg checkRadiusDeg 0 1 0]; % center azi ele, sd major minor, angle of rotation wrt coord system, scaling factor, additive constant
        [~,gaussianEnvelope,~,~] = gauss2D(params,imageAxesDeg.xAxisDeg,imageAxesDeg.yAxisDeg,[]);
        imgValMasked = imgVal .* gaussianEnvelope;
        [imageDFT,radiusVector,imgRadialPSD,oriVarVsF,oriMaxVsF] = getImageDFT(imgValMasked,imageAxesDeg);  % get 2D fourier xfm of image layers & its 1D radial avg.     
        oriMaxVsF(oriMaxVsF<0)=oriMaxVsF(oriMaxVsF<0)+180; % lower quads to upper quads
end 
function [imageDFT,radare,imagePSDrad,oriVarVsF,oriMaxVsF]= getImageDFT(imageIn,imageAxesDeg)
        [hIm,wIm] = size(imageIn); % 2D image
        % get frequency domain axes:
        dFy = 1/(imageAxesDeg.yAxisDeg(end)- imageAxesDeg.yAxisDeg(1));  % inv of spatial spread (deg) 
        dFx = 1/(imageAxesDeg.xAxisDeg(end)- imageAxesDeg.xAxisDeg(1));  % is freq domain resolution (per deg)
        Fs_x= wIm.*dFx;               Fs_y= hIm.*dFy;     % image pixels per degree - sampling freq
        Fx  = -Fs_x/2:dFx:Fs_x/2-dFx; Fy  = -Fs_y/2:dFy:Fs_y/2-dFy;
        imageDFT = fftshift(fft2(imageIn));                % 2D FFT
        [radare,imagePSDrad,oriVarVsF,oriMaxVsF] = getRadAvg(imageDFT,Fx,Fy); % get 1D power spectrum by doing radial averaging
end 
function [radare, radialSpectrum, oriVar, oriMax] = getRadAvg(fft2d, Fx, Fy)
        % Compute radially average power spectrum. My method evolved from following refs:
        % Image analyst's answer: https://nl.mathworks.com/matlabcentral/answers/340538-how-to-average-the-2d-spectrum-of-an-image-from-fft2-to-get-1d-spectrum
        % https://nl.mathworks.com/matlabcentral/fileexchange/23636-radially-averaged-power-spectrum-of-2d-real-valued-matrix
        % https://dsp.stackexchange.com/questions/37957/power-spectrum-of-2d-image-result-interpretation
        % https://mathematica.stackexchange.com/questions/88168/averaging-over-a-circle-from-fft-to-plot-the-amplitude-vs-wavelength
        % Our PSD is not square. Moving 1 point in X direction may not be equal to moving 1 point in Y direction.
        % dFx ~=dFy, so equidistant points will form an ellipse, not circle, around the center pixel
        % now calculate the radius (SF) from 0,0 at each pixel & angle (Ori) at each pixel
        imgFp  = (abs(fft2d));
        [X, Y] = meshgrid(Fx, flip(Fy));  % flip Fy so -ve is lower quadrants   
        rhoF = nan(size(imgFp));
        angD = nan(size(imgFp));
        for rows=1:size(imgFp,1)
            for cols=1:size(imgFp,2)
                rhoF(rows,cols)= sqrt( (X(rows,cols))^2 + (Y(rows,cols))^2 );
                angD(rows,cols)= atan2d( Y(rows,cols),X(rows,cols) );
            end
        end
        % make bins/ discs of radii sizes, if dFx==dFy, use either as stepsize. 
        % within each bin, combine & avg pixels to get resp at that SF
        % calculate the variance of response. this is the orientation variance at that SF
        % pick out the angle at which the response is maximum. 
        stepsize= max(Fx(2)-Fx(1),Fy(2)-Fy(1));                % bigger of the two 
        radare  = 0:stepsize:max(max(rhoF));           % make a vector of radii. outer bin edges, same unit as dFx, dFy, Fx, Fy
        [indicesRadial,~] = getMaskIndices(radare,[0 0],Fx,Fy,'diff');  % will have pixel inds at each radius
        radialSpectrum = nan(size(radare));
        oriVar         = nan(size(radare));
        oriMax         = nan(size(radare));
        for r = 1:length(radare)
            if ~isempty(indicesRadial{r})
                radialSpectrum(r) = mean(imgFp(indicesRadial{r}));
                oriVar(r)         = var(imgFp(indicesRadial{r}));%./ (radialSpectrum(r));  % var by mean
                indmax    = find(imgFp(indicesRadial{r})==max(imgFp(indicesRadial{r})));
                oriMax(r) = angD(indicesRadial{r}(indmax(1)));
            end
        end
end  
function [precomputed_indices,aperture]=getMaskIndices(radiusMatrixDeg,rfCenterDeg,xAxisDeg,yAxisDeg,measure)
    numRadii = length(radiusMatrixDeg);
    precomputed_indices=cell(numRadii,1);
    %setting the defaults for calling makeGaborStimulus()
    gaborStim.azimuthDeg=rfCenterDeg(1);
    gaborStim.elevationDeg=rfCenterDeg(2);
    gaborStim.spatialFreqCPD=0; % These do not matter since we are only interested in the aperture
    gaborStim.sigmaDeg=100000;
    gaborStim.orientationDeg=0;
    gaborStim.contrastPC = 100;

    goodPosPreviousRadius=[]; % Keeps the goodPos of the previous radius to perform the 'diff' computation

    for i=1:numRadii        
        gaborStim.radiusDeg=radiusMatrixDeg(i);
        [~,aperture] = makeGaborStimulus(gaborStim,xAxisDeg,yAxisDeg); % generate a circular mask      
        goodPos = find(aperture==1);% Get goodPos, the indices where mask==1

        if strcmp(measure,'diff') 
            goodPosToUse = setdiff(goodPos,goodPosPreviousRadius);% find the indices of the annular disk between the previous radius and current one
            goodPosPreviousRadius=goodPos; % update previous radis goodpos

        elseif strcmp(measure,'abs')% if  measure is not diff 
            goodPosToUse = goodPos;              
        end
        precomputed_indices(i)={goodPosToUse};        
    end
end 
function [imgCat,peakSfs,peakOvs,peakOrs,peakIndsPsd]= getImageSpectraPeaks(radare,radPSDImage,oriVar,oriMax,sflims)
        % checks the image PSD for peaks & the corrsp OriVar value. 
        radPSDImage  = squeeze(radPSDImage);
        diffpsd      = radPSDImage./max(radPSDImage);   % normalise
        [peakIndsPsd,~] = islocalmax(diffpsd,'FlatSelection','center','MinSeparation',5);
        peakIndsPsd = find(peakIndsPsd);
        peakIndsPsd = peakIndsPsd(radare(peakIndsPsd)>=sflims(1) & radare(peakIndsPsd)<=sflims(2));
        peakSfs = radare(peakIndsPsd);
        peakOvs = oriVar(peakIndsPsd); % what are the OriVars at these SFS?
        peakOrs = oriMax(peakIndsPsd); % what are the Oris at these SFS which have max product?
        
        if any(peakSfs>=sflims(1) & peakSfs<=sflims(2)) % && any(peakOvs>=ovThresh)
              imgCat    = true;
        else, imgCat    = false;
        end
end  

function [gaborParams,oriVar] = getBestGaborFit(imgVal,imageAxesDeg,rfCenterDeg,rfRadDeg,sfs,thetasDeg,radbysig)
degppixX= (imageAxesDeg.xAxisDeg(end)-imageAxesDeg.xAxisDeg(1))/length(imageAxesDeg.xAxisDeg);
degppixY= (imageAxesDeg.yAxisDeg(end)-imageAxesDeg.yAxisDeg(1))/length(imageAxesDeg.yAxisDeg);
degppix = min(degppixX,degppixY);  % use smaller. tighter cycles. 
                                   % degppixX,degppixY are approx equal, it wont make a big difference. 
[imgPrd,normofFilt,~] = getRFsumFilt(imgVal,imageAxesDeg,rfCenterDeg,rfRadDeg,sfs,thetasDeg,degppix);
% normalise products
filteredProd =  imgPrd./normofFilt; 
% find max product to get SfOri parms.
[maxProdParams,~,oriVar]   = getMaxProdandOV(filteredProd,sfs,thetasDeg,rfRadDeg);
gaborParams.azimuthDeg     = rfCenterDeg(1);
gaborParams.elevationDeg   = rfCenterDeg(2);
gaborParams.sigmaDeg       = maxProdParams(1);
gaborParams.spatialFreqCPD = maxProdParams(2);
gaborParams.orientationDeg = maxProdParams(3);
gaborParams.oriVar         = oriVar(sfs==maxProdParams(2));
% calculate phase
gaborParams = getPh(imgVal,gaborParams,imageAxesDeg,radbysig); % find Ph 
% calculate size
gaborParams = getSizeChecked(imgVal,gaborParams,imageAxesDeg,3); 
% calculate contrast in the chosen size.
gaborParams.contrastPC = 100;% contrastCalc

end

function [filteredProd,normofFilt,g] = getRFsumFilt(imgVal,imageAxesDeg,rfCenterDeg,rfRadiusDeg,sfs,thetasDeg,degppix)
% 1.
numOrs = length(thetasDeg);
numSfs = length(sfs);

% 2. get filter bank
SFppc = 1./(sfs*degppix);        % pixel/cycle - wavelength
g = gabor(SFppc,thetasDeg,'SpatialFrequencyBandwidth',1,'SpatialAspectRatio',1);

% 3. Apply filters on images:
[prodisAll, ~] =  imgaborfilt(imgVal, g);

% 4. sum it over the RF
params= [rfCenterDeg rfRadiusDeg rfRadiusDeg 0 1 0]; % center azi ele, sd major minor, angle of rotation wrt coord system, scaling factor, additive constant
[~,gaus,~,~] = gauss2D(params,imageAxesDeg.xAxisDeg,imageAxesDeg.yAxisDeg,[]);
oEnergyTemp = zeros(1,length(g));
for gl = 1:length(g)
    oEnergyTemp(gl) = sum(sum(prodisAll(:,:,gl).*gaus)); 
end

% 5. Rearrange
g = reshape(g,1,numSfs,numOrs);
g = flip(g,2);                 % get sfs in increasing order
filteredProd = reshape(oEnergyTemp,1,numSfs,numOrs);
filteredProd = flip(filteredProd,2);

% 6. for normalising later
normofFilt = zeros(1,numSfs,numOrs);
for f = 1:numSfs
    for o = 1:numOrs
        spatialFilter = real(g(1,f,o).SpatialKernel)/2;
        normofFilt(1,f,o)= sqrt(sum(sum(spatialFilter.*spatialFilter))); 
    end
end
end
function [maxProdParams,maxProd,oriVar,sfOriVar,maxOvParams] = getMaxProdandOV(filteredProd,sfs,thetas,sizes)
% find the max product with the filter 
[X,Y,Z] = ndgrid(sizes,sfs,thetas);
temp = abs(filteredProd);     % magnitude of product
yD   = temp(:);
[maxProd, ind1] = max(yD); % amp is the amplitude.
maxProdParams(1) = X(ind1);       % size  
maxProdParams(2) = Y(ind1);       % sf
maxProdParams(3) = Z(ind1);       % theta
maxProdParams(4) = nan;

% get Ori Variance
oriVar  =nan(length(sizes),length(sfs));
sfOriVar=nan(length(sizes));
for s = 1:length(sizes)
    for f = 1:length(sfs) 
        temp1 = temp(s,f,:);
        oriVar(s,f)= var(temp1)/(mean(temp1))^2;      % Ori variance/mean2 dimensionless.
    end 
    temp2 = temp(s,:,:);
    sfOriVar(s)= var(temp2(:))/(mean(temp2(:)))^2;    % SF ori variance.  
end  
% at which params is OV max?
[rowind,colind]= find(oriVar == max(oriVar(:)));
maxOvParams(1) = sizes(rowind);
maxOvParams(2) = sfs(colind);
maxOvParams(3) = thetas(temp(rowind,colind,:) == max(temp(rowind,colind,:)));
maxOvParams(4) = nan;
end 
function [gaborParams]= getPh(allVals,gaborParams,imageAxesDeg,radbysig)
% initialise gabor inputs for making a gabor 
gaborStim = gaborParams;
gaborStim.contrastPC = 100;
gaborStim.radiusDeg  = gaborStim.sigmaDeg * radbysig;
usePhases = [0 90]; %  get prod at 2 phases 
useProd   = zeros(size(usePhases));
for pp=1:length(usePhases)
    gaborStim.spatialFreqPhaseDeg=usePhases(pp);
    gaborIs = makeGaborStimulus(gaborStim,imageAxesDeg.xAxisDeg,imageAxesDeg.yAxisDeg,0);
    useProd(pp)  = sum(sum(gaborIs.*allVals));
end 
phout1= atan2d(useProd(2),useProd(1));
if phout1<0, phout1 = phout1+2*pi; end
gaborParams.spatialFreqPhaseDeg = phout1;
end
function [gaborParams,prodOut] = getSizeChecked(allVals,gaborParams,imageAxesDeg,radbysig)
% start with given size & increase till product and Ov keep increasing.
sizesInc = [0.5 0.4 0.3 0.2 0.1];  % in degrees. increment steps
phs      = [gaborParams.spatialFreqPhaseDeg gaborParams.spatialFreqPhaseDeg+90];
gaborStim= gaborParams;
gaborStim.radiusDeg  = gaborStim.sigmaDeg * radbysig; % for aperture
gaborStim.contrastPC = 100;   % 
% make Gabor & get product at given size
useProd = zeros(1,2);
for p=1:2 % 2phases
      gaborStim.spatialFreqPhaseDeg = phs(p);
      gaborIs = makeGaborStimulus(gaborStim,imageAxesDeg.xAxisDeg,imageAxesDeg.yAxisDeg,0);
      normOfFilt = sqrt(sum(sum(gaborIs.*gaborIs)));
      useProd(p) = sum(sum(gaborIs.*allVals))/normOfFilt;
end 
prodOut = sqrt(useProd(1)^2 + useProd(2)^2);  % from 2 phases

usestep = 1;           % go on to check more sizes
counter1= 0;           % insert counter here for check
while ~isempty(usestep)  % keep checking
        sz = gaborParams.sigmaDeg;    
        sz = sz + sizesInc(usestep); % new size
        gaborStim.sigmaDeg = sz; gaborStim.radiusDeg = gaborStim.sigmaDeg * radbysig; 
        for p=1:2 % 2phases
            gaborStim.spatialFreqPhaseDeg = phs(p);
            gaborIs = makeGaborStimulus(gaborStim,imageAxesDeg.xAxisDeg,imageAxesDeg.yAxisDeg,0);
            normOfFilt = sqrt(sum(sum(gaborIs.*gaborIs)));
            useProd(p) = sum(sum(gaborIs.*allVals))/normOfFilt;
        end
        tempPrd2 = sqrt(useProd(1)^2 + useProd(2)^2);  % from 2 phases
        
        if (tempPrd2>=prodOut) || (diff([tempPrd2 prodOut])<=0.01*prodOut)    % if higher product or within 1%
            gaborParams.sigmaDeg = sz;       % new size updated
            prodOut  = tempPrd2; % new product to be sent out.
        else
            usestep  = usestep+1; % if not met, use smaller increments than prev pass
        end
        if counter1==10  || usestep == length(sizesInc)+1 % if limit reached
           usestep   = [];           % time to leave
        end  
        counter1  = counter1+1;
end    
end
function [ConMich,mnLum,sdLum] = getMichelCon(imageV,rfCenterDeg,radiusDeg,xAxisDeg,yAxisDeg)

% which pixels are included in this patch? make a circular mask
gaborStim.azimuthDeg=rfCenterDeg(1);
gaborStim.elevationDeg=rfCenterDeg(2);
gaborStim.spatialFreqCPD=0; % These do not matter since we are only interested in the aperture
gaborStim.sigmaDeg=100000;
gaborStim.orientationDeg=0;
gaborStim.contrastPC = 100;
gaborStim.radiusDeg=radiusDeg;
[~,ellipsePixels] = makeGaborStimulus(gaborStim,xAxisDeg,yAxisDeg); % generate a circular mask      

[rowinds,colinds] = find(ellipsePixels);
numinds = length(rowinds);
% for Michelson contrast. 
chosenPixVal = zeros(1,numinds);
for ind = 1:numinds
    chosenPixVal(ind) =  imageV(rowinds(ind),colinds(ind));
end
maxLum= max(chosenPixVal);
minLum= min(chosenPixVal);

ConMich = (maxLum-minLum)/(maxLum+minLum);

% 3. Luminance mean & std
mnLum = mean(chosenPixVal);
sdLum = std(chosenPixVal);

end



