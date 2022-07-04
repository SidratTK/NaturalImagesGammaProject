% Modified from displaySingleChannelGRF which is part of CommonPrograms
% (https://github.com/supratimray/CommonPrograms). Please make sure that
% CommonPrograms is added to Matlab's add path.

% Apart from CommonPrograms, Chronux must be installed and added to
% Matlab's add path if you want to do time-frequency analysis that use
% multi-tapering method.

% Several options have been added for EEG analysis which are not needed
% right now. But these may be useful in future when we have comparable EEG
% data, and therefore have not been removed.

% Run this function as:
% displaySingleChannelNaturalImages('alpaH','240817','GRF_002')

function displaySingleChannelNaturalImages(subjectName,expDate,protocolName,folderSourceString,gridType,gridLayout,badTrialNameStr,useCommonBadTrialsFlag)

if ~exist('folderSourceString','var');  folderSourceString='';          end
if ~exist('gridType','var');            gridType='Microelectrode';      end
if ~exist('gridLayout','var');          gridLayout=2;                   end
if ~exist('badTrialNameStr','var');     badTrialNameStr = '';           end
if ~exist('useCommonBadTrialsFlag','var'); useCommonBadTrialsFlag = 1;  end

if isempty(folderSourceString)
    folderSourceString = fileparts(pwd);
end
    
folderName = fullfile(folderSourceString,'data',subjectName,gridType,expDate,protocolName);

% Get folders
folderExtract = fullfile(folderName,'extractedData');
folderSegment = fullfile(folderName,'segmentedData');
folderLFP = fullfile(folderSegment,'LFP');
folderSpikes = fullfile(folderSegment,'Spikes');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Display main options %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fonts
fontSizeSmall = 10; fontSizeMedium = 12; fontSizeLarge = 16;
% Panel position details
allPanelsHeight = 0.225; allPanelsStartHeight = 0.725;
electrodePanelWidth = 0.2; electrodeStartPos = 0.025;
rfPanelWidth = 0.15; rfStartPos = 0.25;
parametersPanelWidth = 0.2; parametersStartPos = 0.425;
timingPanelWidth = 0.2; timingStartPos = 0.625;
plotOptionsPanelWidth = 0.15; plotOptionsStartPos = 0.825;
backgroundColor = 'w';

%%%%%%%%%%%%%%%%%%%%%%%% Get details of images datset %%%%%%%%%%%%%%%%%%%%%
[allExperimentalDetails,matchIndex] = getExperimentalDetails(expDate);
if isempty(matchIndex)
    error('Incorrect expDate');
else
    experimentalDetails = allExperimentalDetails{matchIndex};
    if ~isequal(experimentalDetails{1},subjectName); error('subjectName does not match'); end
    if ~isequal(experimentalDetails{6},protocolName); error('protocolName does not match'); end
    set1Name = experimentalDetails{2};
    set2Name = experimentalDetails{3};
    imageFolderName = experimentalDetails{4};
end
rawImageFolder = fullfile(folderSourceString,'data','images',imageFolderName);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Plot Electrode Array %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load Spike and LFP Information
[analogChannelsStored,timeVals,~,analogInputNums] = loadlfpInfo(folderLFP);
[neuralChannelsStored,sourceUnitIDs] = loadspikeInfo(folderSpikes);

% Get RF Data
rfData = load(fullfile(folderSourceString,'data','rfData',subjectName,[subjectName gridType 'RFData.mat']));

% Work with only highRMSElectrodes
analogChannels = intersect(analogChannelsStored,rfData.highRMSElectrodes);
% Work only with sorted units
sortedPos = find(sourceUnitIDs==1);
spikeChannels = neuralChannelsStored(sortedPos);
sourceUnitIDs = sourceUnitIDs(sortedPos);

% Get electrode array information
electrodeGridPos = [electrodeStartPos allPanelsStartHeight electrodePanelWidth allPanelsHeight];
[hElectrodes,colorNamesAll,electrodeArray] = showElectrodeLocationsInColor(electrodeGridPos,[],0,0,1,gridType,subjectName,gridLayout);

% Find colors for the selected electrodes
numAnalogChannels = length(analogChannels);
colorNamesAnalog = cell(1,numAnalogChannels);
for ii=1:numAnalogChannels
    colorNamesAnalog{ii} = colorNamesAll{analogChannels(ii)==electrodeArray};
end
numSpikeChannels = length(spikeChannels);
colorNamesSpikes = cell(1,numSpikeChannels);
for ii=1:numSpikeChannels
    colorNamesSpikes{ii} = colorNamesAll{spikeChannels(ii)==electrodeArray};
end

% Gray out bad channels
badAnalogChannels = setdiff(electrodeArray(:),analogChannels);
showElectrodeLocations([],badAnalogChannels,[0.5 0.5 0.5],hElectrodes,1,0,gridType,subjectName);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Show RF Map %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rfMapPos = [rfStartPos allPanelsStartHeight+0.025 rfPanelWidth allPanelsHeight-0.025];
hRFMapPlot = subplot('Position',rfMapPos,'box','on');

hold(hRFMapPlot,'on');
for ii=1:numAnalogChannels
    plot(hRFMapPlot,rfData.rfStats(analogChannels(ii)).meanAzi,rfData.rfStats(analogChannels(ii)).meanEle,'marker','+','color',colorNamesAnalog{ii});
end
xlabel(hRFMapPlot,'Degrees'); ylabel(hRFMapPlot,'Degrees'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters panel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameterBoxHeight = 0.2; parameterBoxGap=0; parameterTextWidth = 0.6;
hParameterPanel = uipanel('Title','Parameters','fontSize', fontSizeLarge, ...
    'Unit','Normalized','Position',[parametersStartPos allPanelsStartHeight parametersPanelWidth allPanelsHeight]);

% Analog channel
[analogChannelStringList,analogChannelStringArray] = getAnalogStringFromValues(analogChannels,analogInputNums);
uicontrol('Parent',hParameterPanel,'Unit','Normalized', ...
    'Position',[0 1-(parameterBoxHeight+parameterBoxGap) parameterTextWidth parameterBoxHeight],...
    'Style','text','String','Analog Channel','FontSize',fontSizeMedium);
hAnalogChannel = uicontrol('Parent',hParameterPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, 'Position', ...
    [parameterTextWidth 1-(parameterBoxHeight+parameterBoxGap) 1-parameterTextWidth parameterBoxHeight], ...
    'Style','popup','String',analogChannelStringList,'FontSize',fontSizeMedium);

% Spike channel
uicontrol('Parent',hParameterPanel,'Unit','Normalized', ...
    'Position',[0 1-2*(parameterBoxHeight+parameterBoxGap) parameterTextWidth parameterBoxHeight],...
    'Style','text','String','Spike Channel','FontSize',fontSizeMedium);
    
if ~isempty(spikeChannels)
    spikeChannelString = getNeuralStringFromValues(spikeChannels,sourceUnitIDs);
    hSpikeChannel = uicontrol('Parent',hParameterPanel,'Unit','Normalized', ...
        'BackgroundColor', backgroundColor, 'Position', ...
        [parameterTextWidth 1-2*(parameterBoxHeight+parameterBoxGap) 1-parameterTextWidth parameterBoxHeight], ...
        'Style','popup','String',spikeChannelString,'FontSize',fontSizeMedium);
else
    hSpikeChannel = uicontrol('Parent',hParameterPanel,'Unit','Normalized', ...
        'Position', [parameterTextWidth 1-2*(parameterBoxHeight+parameterBoxGap) 1-parameterTextWidth parameterBoxHeight], ...
        'Style','text','String','Not found','FontSize',fontSizeMedium);
end

% Analysis Type
analysisTypeString = 'ERP|Firing Rate|Raster|FFT|delta FFT|FFT_ERP|delta FFT_ERP|TF|delta TF';
uicontrol('Parent',hParameterPanel,'Unit','Normalized', ...
    'Position',[0 1-3*(parameterBoxHeight+parameterBoxGap) parameterTextWidth parameterBoxHeight], ...
    'Style','text','String','Analysis Type','FontSize',fontSizeMedium);
hAnalysisType = uicontrol('Parent',hParameterPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, 'Position', ...
    [parameterTextWidth 1-3*(parameterBoxHeight+parameterBoxGap) 1-parameterTextWidth parameterBoxHeight], ...
    'Style','popup','String',analysisTypeString,'FontSize',fontSizeMedium);

% Set Number
if ~isempty(set2Name)
    setNums = [set1Name '|' set2Name];
    numSets=2;
else
    setNums = set1Name;
    numSets=1;
end
uicontrol('Parent',hParameterPanel,'Unit','Normalized', ...
    'Position',[0 1-4*(parameterBoxHeight+parameterBoxGap) parameterTextWidth parameterBoxHeight], ...
    'Style','text','String','Set Name','FontSize',fontSizeMedium);
hSetNum = uicontrol('Parent',hParameterPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, 'Position',...
    [parameterTextWidth 1-4*(parameterBoxHeight+parameterBoxGap) 1-parameterTextWidth parameterBoxHeight], ...
    'Style','popup','String',setNums,'FontSize',fontSizeMedium);

% Reference scheme
referenceChannelStringList = ['None|AvgRef|' analogChannelStringList];
referenceChannelStringArray = [{'None'} {'AvgRef'} analogChannelStringArray];

uicontrol('Parent',hParameterPanel,'Unit','Normalized', ...
    'Position',[0 1-5*(parameterBoxHeight+parameterBoxGap) parameterTextWidth parameterBoxHeight], ...
    'Style','text','String','Reference','FontSize',fontSizeMedium);
hReferenceChannel = uicontrol('Parent',hParameterPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, 'Position',...
    [parameterTextWidth 1-5*(parameterBoxHeight+parameterBoxGap) 1-parameterTextWidth parameterBoxHeight], ...
    'Style','popup','String',referenceChannelStringList,'FontSize',fontSizeMedium);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Timing panel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
timingHeight = 0.125; timingTextWidth = 0.5; timingBoxWidth = 0.25;
hTimingPanel = uipanel('Title','Timing','fontSize', fontSizeLarge, ...
    'Unit','Normalized','Position',[timingStartPos allPanelsStartHeight timingPanelWidth allPanelsHeight]);

signalRange = [-0.5 1];
fftRange = [0 100];
baseline = [-0.25 0];
stimPeriod = [0.25 0.5];

% Signal Range
uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'Position',[0 1-timingHeight timingTextWidth timingHeight], ...
    'Style','text','String','Parameter','FontSize',fontSizeMedium);

uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'Position',[timingTextWidth 1-timingHeight timingBoxWidth timingHeight], ...
    'Style','text','String','Min','FontSize',fontSizeMedium);

uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'Position',[timingTextWidth+timingBoxWidth 1-timingHeight timingBoxWidth timingHeight], ...
    'Style','text','String','Max','FontSize',fontSizeMedium);

% Stim Range
uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'Position',[0 1-3*timingHeight timingTextWidth timingHeight], ...
    'Style','text','String','Stim Range (s)','FontSize',fontSizeSmall);
hStimMin = uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, ...
    'Position',[timingTextWidth 1-3*timingHeight timingBoxWidth timingHeight], ...
    'Style','edit','String',num2str(signalRange(1)),'FontSize',fontSizeSmall);
hStimMax = uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, ...
    'Position',[timingTextWidth+timingBoxWidth 1-3*timingHeight timingBoxWidth timingHeight], ...
    'Style','edit','String',num2str(signalRange(2)),'FontSize',fontSizeSmall);

% FFT Range
uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'Position',[0 1-4*timingHeight timingTextWidth timingHeight], ...
    'Style','text','String','FFT Range (Hz)','FontSize',fontSizeSmall);
hFFTMin = uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, ...
    'Position',[timingTextWidth 1-4*timingHeight timingBoxWidth timingHeight], ...
    'Style','edit','String',num2str(fftRange(1)),'FontSize',fontSizeSmall);
hFFTMax = uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, ...
    'Position',[timingTextWidth+timingBoxWidth 1-4*timingHeight timingBoxWidth timingHeight], ...
    'Style','edit','String',num2str(fftRange(2)),'FontSize',fontSizeSmall);

% Baseline
uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'Position',[0 1-5*timingHeight timingTextWidth timingHeight], ...
    'Style','text','String','Basline (s)','FontSize',fontSizeSmall);
hBaselineMin = uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, ...
    'Position',[timingTextWidth 1-5*timingHeight timingBoxWidth timingHeight], ...
    'Style','edit','String',num2str(baseline(1)),'FontSize',fontSizeSmall);
hBaselineMax = uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, ...
    'Position',[timingTextWidth+timingBoxWidth 1-5*timingHeight timingBoxWidth timingHeight], ...
    'Style','edit','String',num2str(baseline(2)),'FontSize',fontSizeSmall);

% Stim Period
uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'Position',[0 1-6*timingHeight timingTextWidth timingHeight], ...
    'Style','text','String','Stim period (s)','FontSize',fontSizeSmall);
hStimPeriodMin = uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, ...
    'Position',[timingTextWidth 1-6*timingHeight timingBoxWidth timingHeight], ...
    'Style','edit','String',num2str(stimPeriod(1)),'FontSize',fontSizeSmall);
hStimPeriodMax = uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, ...
    'Position',[timingTextWidth+timingBoxWidth 1-6*timingHeight timingBoxWidth timingHeight], ...
    'Style','edit','String',num2str(stimPeriod(2)),'FontSize',fontSizeSmall);

% Y Range
uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'Position',[0 1-7*timingHeight timingTextWidth timingHeight], ...
    'Style','text','String','Y Range','FontSize',fontSizeSmall);
hYMin = uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, ...
    'Position',[timingTextWidth 1-7*timingHeight timingBoxWidth timingHeight], ...
    'Style','edit','String','0','FontSize',fontSizeSmall);
hYMax = uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, ...
    'Position',[timingTextWidth+timingBoxWidth 1-7*timingHeight timingBoxWidth timingHeight], ...
    'Style','edit','String','1','FontSize',fontSizeSmall);

% Z Range
uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'Position',[0 1-8*timingHeight timingTextWidth timingHeight], ...
    'Style','text','String','Z Range','FontSize',fontSizeSmall);
hZMin = uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, ...
    'Position',[timingTextWidth 1-8*timingHeight timingBoxWidth timingHeight], ...
    'Style','edit','String','0','FontSize',fontSizeSmall);
hZMax = uicontrol('Parent',hTimingPanel,'Unit','Normalized', ...
    'BackgroundColor', backgroundColor, ...
    'Position',[timingTextWidth+timingBoxWidth 1-8*timingHeight timingBoxWidth timingHeight], ...
    'Style','edit','String','1','FontSize',fontSizeSmall);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Options %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotOptionsHeight = 0.15;
hPlotOptionsPanel = uipanel('Title','Plotting Options','fontSize', fontSizeLarge, ...
    'Unit','Normalized','Position',[plotOptionsStartPos allPanelsStartHeight plotOptionsPanelWidth allPanelsHeight]);

uicontrol('Parent',hPlotOptionsPanel,'Unit','Normalized', ...
    'Position',[0 5*plotOptionsHeight 1 plotOptionsHeight], ...
    'Style','pushbutton','String','cla','FontSize',fontSizeMedium, ...
    'Callback',{@cla_Callback});

uicontrol('Parent',hPlotOptionsPanel,'Unit','Normalized', ...
    'Position',[0 4*plotOptionsHeight 1 plotOptionsHeight], ...
    'Style','togglebutton','String','hold on','FontSize',fontSizeMedium, ...
    'Callback',{@holdOn_Callback});

uicontrol('Parent',hPlotOptionsPanel,'Unit','Normalized', ...
    'Position',[0 3*plotOptionsHeight 1 plotOptionsHeight], ...
    'Style','pushbutton','String','rescale Z','FontSize',fontSizeMedium, ...
    'Callback',{@rescaleZ_Callback});

uicontrol('Parent',hPlotOptionsPanel,'Unit','Normalized', ...
    'Position',[0 2*plotOptionsHeight 1 plotOptionsHeight], ...
    'Style','pushbutton','String','rescale Y','FontSize',fontSizeMedium, ...
    'Callback',{@rescaleY_Callback});

uicontrol('Parent',hPlotOptionsPanel,'Unit','Normalized', ...
    'Position',[0 plotOptionsHeight 1 plotOptionsHeight], ...
    'Style','pushbutton','String','rescale X','FontSize',fontSizeMedium, ...
    'Callback',{@rescaleData_Callback});

uicontrol('Parent',hPlotOptionsPanel,'Unit','Normalized', ...
    'Position',[0 0 1 plotOptionsHeight], ...
    'Style','pushbutton','String','plot','FontSize',fontSizeMedium, ...
    'Callback',{@plotData_Callback});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get plots and message handles
uicontrol('Unit','Normalized','Position',[0 0.95 1 0.05],...
    'Style','text','String',[subjectName expDate protocolName],'FontSize',fontSizeLarge);

% Plot handles
[~,~,~,~,fValsUnique] = loadParameterCombinations(folderExtract);
numPlots = length(fValsUnique)/numSets;
hImagesPlot = getPlotHandles(1,numPlots,[0.025 0.6 0.95 0.1],0.002);
hImagePatchesPlot = getPlotHandles(1,numPlots,[0.025 0.45 0.95 0.1],0.002);
hImagePatchPredictionPlot = getPlotHandles(1,numPlots,[0.025 0.3 0.95 0.1],0.002);
hDataPlot = getPlotHandles(1,numPlots,[0.025 0.05 0.95 0.2],0.002);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function plotData_Callback(~,~)
        
        analysisType = get(hAnalysisType,'val');
        setNum = get(hSetNum,'val');
        blRange = [str2double(get(hBaselineMin,'String')) str2double(get(hBaselineMax,'String'))];
        stRange = [str2double(get(hStimPeriodMin,'String')) str2double(get(hStimPeriodMax,'String'))];
        referenceChannelString = referenceChannelStringArray{get(hReferenceChannel,'val')};

        fValsToUse = fValsUnique(numPlots*(setNum-1) + (1:numPlots));
        
        %%%%%%%%%%%%%%%%%%%%%%%% Plot Neural Data %%%%%%%%%%%%%%%%%%%%%%%%%
        if analysisType == 2 || analysisType == 3 % Spike Data
            channelPos = get(hSpikeChannel,'val');
            channelNumber = spikeChannels(channelPos);
            unitID = sourceUnitIDs(channelPos);
            plotColor = colorNamesSpikes{channelPos};
            plotSpikeData1Channel(hDataPlot,channelNumber,fValsToUse,folderName,analysisType,timeVals,plotColor,unitID,badTrialNameStr);
        else
            analogChannelPos = get(hAnalogChannel,'val');
            analogChannelString = analogChannelStringArray{analogChannelPos};
            
            if analogChannelPos<=length(colorNamesAnalog)
                plotColor = colorNamesAnalog{analogChannelPos};
                channelNumber = analogChannels(analogChannelPos);
            else
                plotColor = 'k';
                channelNumber=[];
            end
            plotLFPData1Channel(hDataPlot,analogChannelString,fValsToUse,folderName,...
                analysisType,timeVals,plotColor,blRange,stRange,referenceChannelString,badTrialNameStr,useCommonBadTrialsFlag);
        end

        if analysisType<=3 || analysisType>=8 % ERP or spikes, or TF
            xMin = str2double(get(hStimMin,'String'));
            xMax = str2double(get(hStimMax,'String'));
        else
            xMin = str2double(get(hFFTMin,'String'));
            xMax = str2double(get(hFFTMax,'String'));
        end

        if analysisType<=8
            rescaleData(hDataPlot,xMin,xMax,getYLims(hDataPlot));
        else
            yMin = str2double(get(hFFTMin,'String'));
            yMax = str2double(get(hFFTMax,'String'));
            yRange = [yMin yMax];
            rescaleData(hDataPlot,xMin,xMax,yRange);

            zRange = getZLims(hDataPlot);
            set(hZMin,'String',num2str(zRange(1))); set(hZMax,'String',num2str(zRange(2)));
            rescaleZPlots(hDataPlot,zRange);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%% Plot the images %%%%%%%%%%%%%%%%%%%%%%%%%%
        plotImageData(hImagesPlot,hImagePatchesPlot,rawImageFolder,fValsToUse,channelNumber,subjectName,plotColor);
        
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function rescaleZ_Callback(~,~)

        analysisType = get(hAnalysisType,'val');
        
        if analysisType>=8
            zRange = [str2double(get(hZMin,'String')) str2double(get(hZMax,'String'))];
            rescaleZPlots(hDataPlot,zRange);
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function rescaleY_Callback(~,~)

        analysisType = get(hAnalysisType,'val');
        
        if analysisType<=3 || analysisType>=8 % ERP or spikes
            xMin = str2double(get(hStimMin,'String'));
            xMax = str2double(get(hStimMax,'String'));
        else
            xMin = str2double(get(hFFTMin,'String'));
            xMax = str2double(get(hFFTMax,'String'));
        end

        yLims = [str2double(get(hYMin,'String')) str2double(get(hYMax,'String'))];
        rescaleData(hDataPlot,xMin,xMax,yLims);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function rescaleData_Callback(~,~)

        analysisType = get(hAnalysisType,'val');

        if analysisType<=3 || analysisType>=8 % ERP or spikes or TFs
            xMin = str2double(get(hStimMin,'String'));
            xMax = str2double(get(hStimMax,'String'));
        else    
            xMin = str2double(get(hFFTMin,'String'));
            xMax = str2double(get(hFFTMax,'String'));
        end

        if analysisType<=8
            rescaleData(hDataPlot,xMin,xMax,getYLims(hDataPlot));
        else
            yMin = str2double(get(hFFTMin,'String'));
            yMax = str2double(get(hFFTMax,'String'));
            yRange = [yMin yMax];
            rescaleData(hDataPlot,xMin,xMax,yRange);
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function holdOn_Callback(source,~)
        holdOnState = get(source,'Value');    
        holdOnGivenPlotHandle(hDataPlot,holdOnState);

        function holdOnGivenPlotHandle(plotHandles,holdOnState)
            
            [numRows,numCols] = size(plotHandles);
            if holdOnState
                for i=1:numRows
                    for j=1:numCols
                        set(plotHandles(i,j),'Nextplot','add');

                    end
                end
            else
                for i=1:numRows
                    for j=1:numCols
                        set(plotHandles(i,j),'Nextplot','replace');
                    end
                end
            end
        end 
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function cla_Callback(~,~)
        
        claGivenPlotHandle(hImagesPlot);
        claGivenPlotHandle(hImagePatchesPlot);
        claGivenPlotHandle(hImagePatchPredictionPlot);
        claGivenPlotHandle(hDataPlot);
   
        function claGivenPlotHandle(plotHandles)
            [numRows,numCols] = size(plotHandles);
            for i=1:numRows
                for j=1:numCols
                    cla(plotHandles(i,j));
                end
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main function that plots the data
function plotLFPData1Channel(plotHandles,channelString,fList,folderName,...
analysisType,timeVals,plotColor,blRange,stRange,referenceChannelString,badTrialNameStr,useCommonBadTrialsFlag)

folderExtract = fullfile(folderName,'extractedData');
folderSegment = fullfile(folderName,'segmentedData');
folderLFP = fullfile(folderSegment,'LFP');

parameterCombinations = loadParameterCombinations(folderExtract);
numPlots = size(plotHandles,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get Signal %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear analogData
x=load(fullfile(folderLFP,channelString));
analogData=x.analogData;

%%%%%%%%%%%%%%%%%%%%%%%%%% Change Reference %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmpi(referenceChannelString,'None')
    % Do nothing
elseif strcmp(referenceChannelString,'AvgRef')
    disp('Changing to average reference');
    x = load(fullfile(folderLFP,'AvgRef.mat'));
    analogData = analogData - x.analogData;
else
    disp('Changing to bipolar reference');
    x = load(fullfile(folderLFP,referenceChannelString));
    analogData = analogData - x.analogData;
end

% Get bad trials
badTrialFile = fullfile(folderSegment,['badTrials' badTrialNameStr '.mat']);
if ~exist(badTrialFile,'file')
    disp('Bad trial file does not exist...');
    badTrials=[]; allBadTrials=[];
else
    [badTrials,allBadTrials] = loadBadTrials(badTrialFile);    
end

if ~useCommonBadTrialsFlag
    badTrials = allBadTrials{str2double(channelString(5:end))};
end

disp([num2str(length(badTrials)) ' bad trials']);

%%%%%%%%%%%%%%%%%%%%%%% Take a common baseline for TF plots %%%%%%%%%%%%%%%
Fs = round(1/(timeVals(2)-timeVals(1)));
movingwin = [0.25 0.025];
params.tapers   = [1 1];
params.pad      = -1;
params.Fs       = Fs;
params.fpass    = [0 200];
params.trialave = 1; %averaging across trials

if analysisType == 9 % deltaTF
    clear goodPos
    goodPos = parameterCombinations{1,1,1,2*numPlots+1};
    goodPos = setdiff(goodPos,badTrials);
    
    [S,timeTF] = mtspecgramc(analogData(goodPos,:)',movingwin,params);
    xValToPlot = timeTF+timeVals(1)-1/Fs;
    
    blPos = intersect(find(xValToPlot>=blRange(1)),find(xValToPlot<blRange(2)));
    logS = log10(S);
    blPower = mean(logS(blPos,:),1);
    logSBLAllConditions = repmat(blPower,length(xValToPlot),1);
end

for i=1:numPlots
    clear goodPos
    goodPos = parameterCombinations{1,1,1,fList(i)};
    goodPos = setdiff(goodPos,badTrials);
    
    if isempty(goodPos)
        disp('No entries for this combination..');
    else
        disp(['image' num2str(fList(i)) ', n=' num2str(length(goodPos))]);
        
        if round(diff(blRange)*Fs) ~= round(diff(stRange)*Fs)
            disp('baseline and stimulus ranges are not the same');
        else
            range = blRange;
            rangePos = round(diff(range)*Fs);
            blPos = find(timeVals>=blRange(1),1)+ (1:rangePos);
            stPos = find(timeVals>=stRange(1),1)+ (1:rangePos);
            xs = 0:1/diff(range):Fs-1/diff(range);
        end
        
        if analysisType == 1        % compute ERP
            clear erp
            erp = mean(analogData(goodPos,:),1); %#ok<*NODEF>
            plot(plotHandles(i),timeVals,erp,'color',plotColor);

        elseif analysisType == 2  ||   analysisType == 3 % compute Firing rates
            disp('Use plotSpikeData instead of plotLFPData...');
            
        elseif analysisType == 4  ||   analysisType == 5
            fftBL = abs(fft(analogData(goodPos,blPos),[],2));
            fftST = abs(fft(analogData(goodPos,stPos),[],2));
            
            if analysisType == 4
                plot(plotHandles(i),xs,log10(mean(fftBL)),'g');
                hold(plotHandles(i),'on');
                plot(plotHandles(i),xs,log10(mean(fftST)),'k');
                hold(plotHandles(i),'off');
            end
            
            if analysisType == 5
                plot(plotHandles(i),xs,log10(mean(fftST))-log10(mean(fftBL)),'color',plotColor);
                hold(plotHandles(i),'on');
                plot(plotHandles(i),xs,zeros(1,length(xs)),'color','k');
                hold(plotHandles(i),'off');
            end
            
        elseif analysisType == 6 || analysisType == 7
            fftERPBL = abs(fft(mean(analogData(goodPos,blPos),1)));
            fftERPST = abs(fft(mean(analogData(goodPos,stPos),1)));
            
            if analysisType == 6
                plot(plotHandles(i),xs,log10(fftERPBL),'g');
                set(plotHandles(i),'Nextplot','add');
                plot(plotHandles(i),xs,log10(fftERPST),'k');
                set(plotHandles(i),'Nextplot','replace');
            end
            
            if analysisType == 7
                plot(plotHandles(i),xs,log10(fftERPST)-log10(fftERPBL),'color',plotColor);
            end
            
        elseif analysisType == 8 || analysisType == 9  % TF analysis
            colormap jet;
            [S,timeTF,freqTF] = mtspecgramc(analogData(goodPos,:)',movingwin,params);
            xValToPlot = timeTF+timeVals(1)-1/Fs;
            if (analysisType==8)
                pcolor(plotHandles(i),xValToPlot,freqTF,log10(S'));
                shading(plotHandles(i),'interp');
            else
                blPos = intersect(find(xValToPlot>=blRange(1)),find(xValToPlot<blRange(2)));
                logS = log10(S);
                pcolor(plotHandles(i),xValToPlot,freqTF,10*(logS-logSBLAllConditions)');

                %blPower = mean(logS(blPos,:),1);
                %logSBL = repmat(blPower,length(xValToPlot),1);
                %pcolor(plotHandles(i),xValToPlot,freqTF,10*(logS-logSBL)');
                shading(plotHandles(i),'interp');
            end
        end
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotSpikeData1Channel(plotHandles,channelNumber,fList,folderName,analysisType,timeVals,plotColor,unitID,badTrialNameStr)

folderExtract = fullfile(folderName,'extractedData');
folderSegment = fullfile(folderName,'segmentedData');
folderSpikes = fullfile(folderSegment,'Spikes');

parameterCombinations = loadParameterCombinations(folderExtract);
numPlots = size(plotHandles,2);

% Get the data
clear spikeData
x=load(fullfile(folderSpikes,['elec' num2str(channelNumber) '_SID' num2str(unitID) '.mat']));
spikeData=x.spikeData;

% Get bad trials
badTrialFile = fullfile(folderSegment,['badTrials' badTrialNameStr '.mat']);
if ~exist(badTrialFile,'file')
    disp('Bad trial file does not exist...');
    badTrials=[];
else
    badTrials = loadBadTrials(badTrialFile);
    disp([num2str(length(badTrials)) ' bad trials']);
end

for i=1:numPlots
    clear goodPos
    goodPos = parameterCombinations{1,1,1,fList(i)};
    goodPos = setdiff(goodPos,badTrials);
    
    if isempty(goodPos)
        disp('No entries for this combination..')
    else
        disp(['image' num2str(fList(i)) ', n=' num2str(length(goodPos))]);
        
        if analysisType == 2
            [psthVals,xs] = getPSTH(spikeData(goodPos),10,[timeVals(1) timeVals(end)]);
            plot(plotHandles(i),xs,psthVals,'color',plotColor);
        else
            X = spikeData(goodPos);
            axes(plotHandles(i)); %#ok<LAXES>
            rasterplot(X,1:length(X),'k');
        end
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function yLims = getYLims(plotHandles)

[numRows,numCols] = size(plotHandles);
% Initialize
yMin = inf;
yMax = -inf;

for row=1:numRows
    for column=1:numCols
        % get positions
        axis(plotHandles(row,column),'tight');
        tmpAxisVals = axis(plotHandles(row,column));
        if tmpAxisVals(3) < yMin
            yMin = tmpAxisVals(3);
        end
        if tmpAxisVals(4) > yMax
            yMax = tmpAxisVals(4);
        end
    end
end

yLims=[yMin yMax];
end
function zLims = getZLims(plotHandles)

[numRows,numCols] = size(plotHandles);
% Initialize
zMin = inf;
zMax = -inf;

for row=1:numRows
    for column=1:numCols
        % get positions
        tmpAxisVals = caxis(plotHandles(row,column));
        if tmpAxisVals(1) < zMin
            zMin = tmpAxisVals(1);
        end
        if tmpAxisVals(2) > zMax
            zMax = tmpAxisVals(2);
        end
    end
end

zLims=[zMin zMax];
end
function rescaleData(plotHandles,xMin,xMax,yLims)

[numRows,numCols] = size(plotHandles);
labelSize=12;
for i=1:numRows
    for j=1:numCols
        axis(plotHandles(i,j),[xMin xMax yLims]);
        if (i==numRows && rem(j,2)==1)
            if j~=1
                set(plotHandles(i,j),'YTickLabel',[],'fontSize',labelSize);
            end
        elseif (rem(i,2)==0 && j==1)
            set(plotHandles(i,j),'XTickLabel',[],'fontSize',labelSize);
        else
            set(plotHandles(i,j),'XTickLabel',[],'YTickLabel',[],'fontSize',labelSize);
        end
    end
end

% Remove Labels on the four corners
%set(plotHandles(1,1),'XTickLabel',[],'YTickLabel',[]);
%set(plotHandles(1,numCols),'XTickLabel',[],'YTickLabel',[]);
%set(plotHandles(numRows,1),'XTickLabel',[],'YTickLabel',[]);
%set(plotHandles(numRows,numCols),'XTickLabel',[],'YTickLabel',[]);
end
function rescaleZPlots(plotHandles,zLims)
[numRow,numCol] = size(plotHandles);

for i=1:numRow
    for j=1:numCol
        caxis(plotHandles(i,j),zLims);
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [outString,outArray] = getAnalogStringFromValues(analogChannelsStored,analogInputNums)
outString='';
count=1;
for i=1:length(analogChannelsStored)
    outArray{count} = ['elec' num2str(analogChannelsStored(i))]; %#ok<AGROW>
    outString = cat(2,outString,[outArray{count} '|']);
    count=count+1;
end
if ~isempty(analogInputNums)
    for i=1:length(analogInputNums)
        outArray{count} = ['ainp' num2str(analogInputNums(i))];
        outString = cat(2,outString,[outArray{count} '|']);
        count=count+1;
    end
end
end
function outString = getNeuralStringFromValues(neuralChannelsStored,SourceUnitIDs)
outString='';
for i=1:length(neuralChannelsStored)
    outString = cat(2,outString,[num2str(neuralChannelsStored(i)) ', SID ' num2str(SourceUnitIDs(i)) '|']);
end 
end
%%%%%%%%%%%%c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load Data
function [analogChannelsStored,timeVals,goodStimPos,analogInputNums] = loadlfpInfo(folderLFP) %#ok<*STOUT>
x=load(fullfile(folderLFP,'lfpInfo.mat'));
analogChannelsStored=x.analogChannelsStored;
goodStimPos=x.goodStimPos;
timeVals=x.timeVals;

if isfield(x,'analogInputNums')
    analogInputNums=x.analogInputNums;
else
    analogInputNums=[];
end
end
function [neuralChannelsStored,SourceUnitID] = loadspikeInfo(folderSpikes)
fileName = fullfile(folderSpikes,'spikeInfo.mat');
if exist(fileName,'file')
    x=load(fileName);
    neuralChannelsStored=x.neuralChannelsStored;
    SourceUnitID=x.SourceUnitID;
else
    neuralChannelsStored=[];
    SourceUnitID=[];
end
end
function [parameterCombinations,aValsUnique,eValsUnique,sValsUnique,...
    fValsUnique,oValsUnique,cValsUnique,tValsUnique] = loadParameterCombinations(folderExtract)

p = load(fullfile(folderExtract,'parameterCombinations.mat'));

parameterCombinations=p.parameterCombinations;
aValsUnique=p.aValsUnique;
eValsUnique=p.eValsUnique;

if ~isfield(p,'sValsUnique')
    sValsUnique = p.rValsUnique/3;
else
    sValsUnique=p.sValsUnique;
end

fValsUnique=p.fValsUnique;
oValsUnique=p.oValsUnique;

if ~isfield(p,'cValsUnique')
    cValsUnique=100;
else
    cValsUnique=p.cValsUnique;
end

if ~isfield(p,'tValsUnique')
    tValsUnique=0;
else
    tValsUnique=p.tValsUnique;
end
end
function [badTrials,allBadTrials] = loadBadTrials(badTrialFile)
x=load(badTrialFile);
badTrials=x.badTrials;
if isfield(x,'allBadTrials')
    allBadTrials=x.allBadTrials;
else
    allBadTrials=badTrials;
end
end
%%%%%%%%%%%%c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display Images
function plotImageData(hImagesPlot,hImagePatches,rawImageFolder,fValsToUse,channelNumber,subjectName,colorName)
patchSizeDeg=2;
plottingDetails.displayPlotsFlag=1;

for i=1:length(fValsToUse)
    imageFileName = fullfile(rawImageFolder,['Image' num2str(fValsToUse(i)) '.tif']);
    plottingDetails.hImagePlot=hImagesPlot(i);
    plottingDetails.hImagePatches=hImagePatches(i);
    plottingDetails.colorNames=colorName;
    getImagePatches(imageFileName,channelNumber,subjectName,'',patchSizeDeg,plottingDetails);
    if i>1
        set(hImagesPlot(i),'XTicklabel',[],'YTicklabel',[]);
        set(hImagePatches(i),'XTicklabel',[],'YTicklabel',[]);
    else
        xlabel(hImagesPlot(i),'Degrees'); ylabel(hImagesPlot(i),'Degrees');
        xlabel(hImagePatches(i),'Degrees'); ylabel(hImagePatches(i),'Degrees');
    end
end
end