% This program returns the predicted gamma response for any achromatic
% grating or hue patch from the tuning functions that are obtained experimentally 

% Inputs
% Required
% subjectName: 'alpaH' or 'kesariH'
% stimParams: description of an achromatic or chromatic grating or a patch, with the following fields
%    gratings: spatialFreqCPD(1 2 4 etc), orientationDeg(0 to 180), radiusDeg, contrastPC (0 to 100)
%    hsv patches: hueDeg (0 to 360), radiusDeg, sat (saturation 0 to 1), contrastPC(value. 0 to 100 as percentage)

% Optional
% modelParams: model parameters to construct the tuning functions. If not
% defined, we use the default median values that are saved for each monkey
% scalingFactor: the overall gamma is scaled by scalingFactor(1) and then
% an offset of scalingFactor(2) is added. Default: [1 0]
% if actual data (a vector of values of same length as number of stimuli) is sent in as scaling factor,
% then this data is used to get overall gain and offset on scaled predictions 


function gammaVal = getPredictedGamma(subjectName,stimParams,modelParams,scalingFactor)

if ~exist('modelParams','var');         modelParams=[];                 end
if ~exist('scalingFactor','var');       scalingFactor=[1 0];            end

if isfield(stimParams,'hueDeg')
    gammaVal = getPredictedGammaHue(subjectName,stimParams,modelParams,scalingFactor);
else
    gammaVal = getPredictedGammaGrating(subjectName,stimParams,modelParams,scalingFactor);
end
end
function val = getPredictedGammaGrating(subjectName,stimParams,modelParams,scalingFactor)

if isempty(modelParams)
    if strcmp(subjectName,'alpaH')
        modelParams.sfCenter = 5.0196;
        modelParams.sfSigma = 0.9413;
        modelParams.oriCenter = 110.3286;
        modelParams.oriSpreadK = 1.0302;
        modelParams.sizeSlope = 0.3674;
        modelParams.sizeAtHalfMax = 1.2904;
        modelParams.conSlope = 1.4808;
        modelParams.conAtHalfMax = 22.8935;
        
    elseif strcmp(subjectName,'kesariH')
        modelParams.sfCenter = 2.8280;
        modelParams.sfSigma = 1.0938;
        modelParams.oriCenter =  28.2081;   
        modelParams.oriSpreadK = 0.1552;
        modelParams.sizeSlope = 8.1988;
        modelParams.sizeAtHalfMax = 0.4874;
        modelParams.conSlope = 1.1467;
        modelParams.conAtHalfMax = 27.4072;       
    end
end

f = stimParams.spatialFreqCPD;
o = stimParams.orientationDeg;
s = stimParams.radiusDeg;
c = stimParams.contrastPC;

sfCoeffs   = [modelParams.sfCenter modelParams.sfSigma];
oriCoeffs  = [modelParams.oriCenter modelParams.oriSpreadK];
sizeCoeffs = [modelParams.sizeSlope modelParams.sizeAtHalfMax];
conCoeffs  = [modelParams.conSlope modelParams.conAtHalfMax];

sfVal   = gaussianFunctionOfLogx(f,sfCoeffs);   %        
oriVal  = vmFunctionOf2Radx(o,oriCoeffs);
sizeVal = sigmoidalFunctionOfLogx(s,sizeCoeffs); %     
conVal  = sigmoidalFunctionOfLogx(c,conCoeffs); 

% get predicted gamma - scaled without gain
val = 1.*sfVal.*oriVal.*sizeVal.*conVal + 0;
% get gain and intercept if data available
if length(scalingFactor)>2 && length(scalingFactor(:))==length(val) 
    actualData = scalingFactor(:);
    scalingFactor = fitGandO(actualData,val(:)); % get actual scaling factor
    val = scalingFactor(1).*sfVal.*oriVal.*sizeVal.*conVal + scalingFactor(2); % 
end
end
function val = getPredictedGammaHue(subjectName,stimParams,modelParams,scalingFactor)

if isempty(modelParams)
    if strcmp(subjectName,'alpaH')
        modelParams.centerR = 5.9789;
        modelParams.spreadR = 1.9068;
        modelParams.centerC = 3.0042;
        modelParams.spreadC = 2.5095;
        modelParams.gainCbyR = 0.4335;
        modelParams.sizeSlope = 1.6175;
        modelParams.sizeAtHalfMax = 1.2119;
        modelParams.satSlope = 3.2922;
        modelParams.satAtHalfMax = 0.7477;
        modelParams.valSlope = 0.1821; 
        modelParams.valIntercept = 0.7773;
        
    elseif strcmp(subjectName,'kesariH')
        modelParams.centerR = 6.2326;
        modelParams.spreadR = 4.2599;
        modelParams.centerC = 3.7618;
        modelParams.spreadC = 1.8884;    
        modelParams.gainCbyR = 0.4030;
        modelParams.sizeSlope = 0.9333;
        modelParams.sizeAtHalfMax = 0.2258;
        modelParams.satSlope = 4.2867;
        modelParams.satAtHalfMax = 0.6463; 
        modelParams.valSlope = 5.9836e-04;
        modelParams.valIntercept = 0.9329;     
    end
end

o = stimParams.hueDeg;
s = stimParams.radiusDeg;
st= stimParams.sat;
v = 0.5 * (1 + sin(deg2rad(stimParams.spatialFreqPhaseDeg))*(stimParams.contrastPC/100));

hueCoeffs  = [modelParams.centerR modelParams.spreadR modelParams.centerC modelParams.spreadC  modelParams.gainCbyR];
sizeCoeffs = [modelParams.sizeSlope modelParams.sizeAtHalfMax];
satCoeffs  = [modelParams.satSlope modelParams.satAtHalfMax];
valCoeffs  = [modelParams.valSlope modelParams.valIntercept];

hueVal  = sum2VmFunctionsOfRadx(o,hueCoeffs);
sizeVal = sigmoidalFunctionOfLogx(s,sizeCoeffs);
satVal  = sigmoidalFunctionOfx(st,satCoeffs);
valVal  = linearFunctionOfx(v,valCoeffs);

% get predicted gamma - scaled without gain
val = 1.*hueVal.*sizeVal.*satVal.*valVal + 0;  % 

% get gain and intercept if data available
if length(scalingFactor)>2 && length(scalingFactor(:))==length(val) 
    actualData = scalingFactor(:);
    scalingFactor = fitGandO(actualData,val(:)); % get actual scaling factor
    val = scalingFactor(1).*hueVal.*sizeVal.*satVal.*valVal + scalingFactor(2);
end

end

function [y,gaussianFunctionStr] = gaussianFunctionOfLogx(x,coeff)
if ~exist('x','var'), x = 0; end
if ~exist('coeff','var'), coeff = [0 1]; end  % mu = coeff(1); sigma = coeff(2);
gaussianFunction = @(coeff,x) normpdf(log2(x),log2(coeff(1)),coeff(2))./normpdf(log2(coeff(1)),log2(coeff(1)),coeff(2)) ;  % scaled so max is 1
y = gaussianFunction(coeff,x);
gaussianFunctionStr = func2str(gaussianFunction);
end
function [y,vmFunctionStr] = vmFunctionOf2Radx(x,coeff)
if ~exist('x','var'), x = 0; end
if ~exist('coeff','var'), coeff = [0 1]; end  % thetaPreferred = coeff(1). concentration parameter =coeff(2)
vmFunction= @(coeff,x) circ_vmpdf(2*deg2rad(x),2*deg2rad(coeff(1)),coeff(2))./circ_vmpdf(2*deg2rad(coeff(1)),2*deg2rad(coeff(1)),coeff(2));  %  scaled so max is 1
y = vmFunction(coeff,x);
vmFunctionStr = func2str(vmFunction);
end
function [y,sigmoidFunctionStr] = sigmoidalFunctionOfLogx(x,coeff)
if ~exist('x','var'), x = 1; end
if ~exist('coeff','var'), coeff = [1 0]; end  % slope = coeff(1); offset = coeff(2);
sigmoidFunction = @(coeff,x) (1./(1+10.^(coeff(1)*(log2(coeff(2))-log2(x)))));
y = sigmoidFunction(coeff,x);                % y = 1./(1+10.^(slope*(x-offset)));
sigmoidFunctionStr = func2str(sigmoidFunction);
end
function [y,sigmoidFunctionStr] = sigmoidalFunctionOfx(x,coeff)
if ~exist('x','var'), x = 1; end
if ~exist('coeff','var'), coeff = [1 0]; end  % slope = coeff(1); offset = coeff(2);
sigmoidFunction = @(coeff,x) (1./(1+10.^(coeff(1)*(coeff(2)-x))));
y = sigmoidFunction(coeff,x);                % y = 1./(1+10.^-(slope*(x-offset)));
sigmoidFunctionStr = func2str(sigmoidFunction);
end
function [y,sum2VmFunctionStr] = sum2VmFunctionsOfRadx(x,coeff)
if ~exist('x','var'), x = 0; end
if ~exist('coeff','var'), coeff = [0 1 0 1 0]; end
sum2VmFunction =  @(coeff,x)( circ_vmpdf(deg2rad(x),coeff(1),coeff(2))./circ_vmpdf(coeff(1),coeff(1),coeff(2)) + ...  % keep max at 1.
                    coeff(5).*circ_vmpdf(deg2rad(x),coeff(3),coeff(4))./circ_vmpdf(coeff(3),coeff(3),coeff(4)));
y = sum2VmFunction(coeff,x);
sum2VmFunctionStr = func2str(sum2VmFunction);
end
function [y,linearFunctionStr] = linearFunctionOfx(x,coeff)
if ~exist('x','var'), x = 1; end
if ~exist('coeff','var'), coeff = [1 0]; end
linearFunction = @(coeff,x)(coeff(1).*x + coeff(2));
y = linearFunction(coeff,x);
linearFunctionStr = func2str(linearFunction);
end
% optimise gain & offset
function [fitResult] = fitGandO(yData,xData)

tols    = 1e-16;
options = optimset('Display','off','MaxFunEvals',5e2,'MaxIter',5e2,'TolX',tols,'TolFun',tols,'TolCon',tols );
X0  =  [max(yData)-min(yData)  min(yData)];  % overall gain & offset
Low =  [0   0 ];
Upp =  [Inf min(yData)];
funis = @(p,dd) p(1).*dd + p(2);

[fitResult,~,~] = lsqcurvefit(funis,X0,xData,yData,Low,Upp,options);
end