powerOption = 3; % 1 - ST power, 2 - ST/BL ratio, 3 - ST/BL ratio minus ST/BL ratio in HG

selectOptions.meanThr = [0.05 0.05 0.05];
selectOptions.stdThr = 2*selectOptions.meanThr;
selectOptions.measure = 'diff';
selectOptions.method = 'vector';

radiusMatrixDeg = 0.3:0.3:2;

[experimentalDetails,matchIndex] = getExperimentalDetails;
posList = 5; % Index for which data needs to be saved

correlationsFull = [];
correlationsSelected = [];
numSelectedImages = [];

for i=1:length(posList)
    tmp = experimentalDetails{posList(i)};
    
    for j=1:2 % Two image sets per experimental session except "human" dataset
        subjectName = tmp{1};
        imageFolderName = tmp{4};
        expDate = tmp{5};
        protocolName = tmp{6};
        
        if j==1
            dataType = tmp{2};
            imageIndices = 1:16;
        else
            dataType = tmp{3};
            imageIndices = 17:32;
        end
        
        if ~isempty(dataType)
            disp(['Working on ' subjectName expDate protocolName ', set: ' dataType]);
            [cFull,cSelected,numSI,predictionString] = analyzeData(subjectName,expDate,protocolName,imageFolderName,imageIndices,powerOption,selectOptions,radiusMatrixDeg);
            correlationsFull = cat(2,correlationsFull,cFull);
            correlationsSelected = cat(2,correlationsSelected,cSelected);
            numSelectedImages = cat(2,numSelectedImages,numSI);
        end
    end
end

cutoffSelectedImages = 8;
clf;
for i=1:2
    hPlot = subplot(1,2,i);
    if i==1
        X = correlationsFull;
        N = size(X,2);
        titleStr = ['Full set (N=' num2str(N) ')'];
    else
        goodPos = find(numSelectedImages>=cutoffSelectedImages);
        X = correlationsSelected(:,goodPos);
        N = size(X,2);
        titleStr = ['Selected set (cutoff=' num2str(cutoffSelectedImages) ', N=' num2str(N) ')'];
    end
    
    mData = mean(X,2);
    sData = std(X,[],2)/sqrt(N);

    bar(mData,'Parent',hPlot);
    hold(hPlot,'on');
    errorbar(hPlot,mData,sData,'ko');
    set(hPlot,'XTickLabel',[]);
    set(hPlot,'XTickLabel',predictionString);
    ylim(hPlot,[-0.5 1]);
    
    title(hPlot,titleStr);
end