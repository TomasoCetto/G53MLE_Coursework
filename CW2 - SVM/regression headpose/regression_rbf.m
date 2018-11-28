

clear;  % ***Clear all previous data before executing this code***

disp("*Clear all previous data");



% prepare data

load('facialPoints.mat');
load('headpose.mat');
load('Perm.mat');

inputs = reshape(points,[66*2,8955])';   % samples x features

targets = pose(:,6);                       % samples x classes



% cross validation initial

k = 10;

numOfExamples = (k-1)*size(inputs,1)/k; % numOfExamples = 135

numOfFeatures = size(inputs,2);         % inputWidth = 132

foldLength = size(inputs,1)/k;          % foldLength = 15

%P = randperm(size(inputs,1));           % random permutation containing all index of single data point



% variables

global doDisp;  

doDisp = false;  % display flag



tic;    % start timer




%Outerloop returns

rms_errors = zeros(1,10);

predictionsMatrix = cell(1,10);

scoreMatrix = cell(1,10);

supportVectors = zeros(1,10);

SVMRBFPredictionsMat = []


%outerloop

for i = 1:10

    

    [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs , targets);  

    currentModelParameters = innerLoop(trainingInputs,trainingTargets,3);

    

    %test the model

    %currentModel = fitcsvm(trainingInputs,trainingTargets, 'Standardize',true,'KernelFunction', 'rbf', 'KernelScale',currentModelParameters.sigmaValue, 'BoxConstraint',currentModelParameters.cValue); 
    currentModel = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', currentModelParameters.sigmaValue, 'BoxConstraint', currentModelParameters.cValue, 'Epsilon', currentModelParameters.epsilonValue);

    

    %calculate f1 of this fold

    predictions = predict(currentModel,testingInputs);
    
    rms = rmsCal(predictions, testingTargets);
    
    rms_errors(i) = rms;
    
    tPrint("rms = " + rms_errors(i));

    predictionsMatrix{i} = predictions;

    supportVectors(i) = size(currentModel.SupportVectors,1);

    SVMRBFPredictions = [SVMRBFPredictionsMat predictions];

end

SVMRBFPredictions = SVMRBFPredictionsMat(:);

fprintf("\n");

disp("time stamp: " + toc + " sec");    % stamp total training duration





%                           End of process





%   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -

%   -   -   -   -   -   - ???  FUNCTIONS ???    -   -   -   -   -   -   -

%   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -



%inner loop function

function [bestHyperParameter] = innerLoop(inputs,targets,k)

    

    numOfExamples = (k-1)*size(inputs,1)/k; % numOfExamples = 135

    foldLength = size(inputs,1)/k;          % foldLength = 15

    P = randperm(size(inputs,1));           % random permutation containing all index of single data point



    iteration = 1;

    hyperparameterInformation = cell(1,1);

    

    %Box constraint values

    %cValues = [0.01, 0.05, 0.1, 0.5 , 1 , 5 , 10, 50, 100, 500, 1000, 5000, 10000];
    cValues = [1];

    

    %sigma values

    %sigmaValues = [0.01, 0.05,  1 , 5 ];
    sigmaValues = [1];

    

    %polynomial order

    pOrder = [2,3,4,5];

    % epsilon 
    
    epsilonValues = [1];


    for sigmaIndex=1:length(sigmaValues)

        for cIndex=1:length(cValues)
            
            for eIndex=1:length(epsilonValues)
                
                rms_errors = zeros(1,k); 
                
                for i=1:k     % each iteration performs one time of training and CV



                    % retrieve training and testing dataset for i fold

                    [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs, targets);  



                    % training SVM

                    %SVM = fitcsvm(trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', sigmaValues(sigmaIndex), 'BoxConstraint', cValues(cIndex));
                    SVM = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', sigmaValues(sigmaIndex), 'BoxConstraint', cValues(cIndex), 'Epsilon', epsilonValues(eIndex));

                    display("sigma values =" + sigmaValues(sigmaIndex))

                    display("c values = " + cValues(cIndex))
                    
                    display("epsilon values = " + epsilonValues(eIndex))

                    tPrint("finish " + i + "-fold training SVM: " + tic + " sec");

                    sv = size(SVM.SupportVectors,1);

                    tPrint("sv = " + sv + "(" + size(SVM.SupportVectors,1)/size(SVM.SupportVectors,2)*100 + "%)");



                    % prediction and evaluation. 

                    predictions = predict(SVM,testingInputs);
                    
                    rms = rmsCal(predictions, testingTargets);
                    rms_errors(i) = rms;
                    
                    tPrint("rms_errors = " + rms_errors(i));



                end     % end of fold cross validation 


            % determine mean values
            meanRMS = mean(rms_errors);

           

            % store hyperparameters into a structure

            % struct.pOrder = 2;

            struct.cValue = cValues(cIndex);

            struct.sigmaValue = sigmaValues(sigmaIndex);
            
            struct.epsilonValue = epsilonValues(eIndex);

            struct.supportVectors = size(SVM.SupportVectors,1);

            struct.supportVectorsRatio = struct.supportVectors / size(inputs,1);

            struct.rms = meanRMS;

            hyperparameterInformation{iteration} = struct;

            

            %hyperparameterInformation

            iteration = iteration + 1;

            end

        end

    end



    % select the best rms
    bestRMS = 100;
    for i=1:length(hyperparameterInformation)
        currentRMS = hyperparameterInformation{i}.rms;
        if currentRMS < bestRMS
            bestRMS = currentRMS;
            bestHyperParameter = hyperparameterInformation{i};
        end
    end


        disp("The best hyperparameter regarding to rms: ");

        disp(bestHyperParameter);



    

end



%Start of confusion matrix



function cm = confusion_matrix(outputs, labels)

% cm = zeros(2);

tp=0;tn=0;fp=0;fn=0;

% length(outputs);

for i=1:length(outputs)

    if (labels(i) == 1) && (outputs(i)==1)

        tp=tp+1;

    elseif (labels(i) == 0) && (outputs(i)==0)

        tn=tn+1;

    elseif (labels(i) == 1) && (outputs(i)==0)

        fn=fn+1;

    else

        fp=fp+1;

    end

end

cm = [tp, fn; fp, tn];

end



% Start of CV Partition



function [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs, targets)

    validPerm = P((i-1)*foldLength+1:i*foldLength); % extract the indexes of validation data

    % the remaining are indexes of training data

    if i==1

        trainPerm = P(foldLength+1:end);

    elseif i==k

        trainPerm = P(1:numOfExamples);

    else

        trainPerm1 = P(1:(i-1)*foldLength);

        trainPerm2 = P(i*foldLength+1:end);

        trainPerm = [trainPerm1,trainPerm2];

    end

    % Set up Division of Data for Training, Validation, Testing

    % find the values of features and labels with their corresponding indexes

    trainingTargets = targets( trainPerm, :);

    trainingInputs = inputs( trainPerm, :);

    % find the values of features and labels with their corresponding indexes

    testingTargets = targets( validPerm, :);

    testingInputs = inputs( validPerm, :);

end



% Start of printing

function [] = tPrint(str)   % tracing print

    global doDisp;

    if doDisp

        disp(str);

    end

end     % end of tPrint

function [rms] = rmsCal(outputs, testingTargets)
%calculate the rms error between predictions and targets and store it
    total = 0;
    differences = outputs - testingTargets;
    for j=1:length(differences)
        square = power(differences(j), 2);
        total = total + square;
    end
    ms = (1/length(testingTargets))*total;
    rms = sqrt(ms);
end     % end of rms