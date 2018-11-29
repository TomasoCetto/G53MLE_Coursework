clear;  % ***Clear all previous data before executing this code***

% prepare data
load('facialPoints.mat');
load('headpose.mat');
% load('PermutationRegression.mat');

inputs = reshape(points,[66*2,8955])';   % samples x features
targets = pose(:,6);                       % samples x classes
P = randperm(size(inputs,1));
% cross validation initial

k = 10;
numOfFeatures = size(inputs,2);         % inputWidth = 132
foldLength = size(inputs,1)/k;          % foldLength = 15

%Outerloop returns
rms_errors = zeros(1,10);
predictionsMatrix = cell(1,10);
scoreMatrix = cell(1,10);
supportVectors = zeros(1,10);
SVMrbfPredictionsMat = [];
allModels = {};
predictionCat = [];

%Outerloop      can use parfor loop to run in parellel, thus saving time
for i = 1:k
    [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, i, P, k, inputs , targets);  
    currentModelParameters = innerLoop(trainingInputs,trainingTargets,3);
    %test the model
    currentModel = fitrsvm(trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', currentModelParameters.sigmaValue, 'BoxConstraint', currentModelParameters.cValue, 'Epsilon', currentModelParameters.epsilonValue);
    allModels{i} = currentModel;
    %calculate f1 of this fold
    predictions = predict(currentModel,testingInputs);
    rms = rmsCal(predictions, testingTargets);
    rms_errors(i) = rms;
    predictionsMatrix{i} = predictions;
    supportVectors(i) = size(currentModel.SupportVectors,1);

end
predictionCat = vertcat(predictionCat,predictionsMatrix{:});


%inner loop function
function [bestHyperParameter] = innerLoop(inputs,targets,k)
    foldLength = round(size(inputs,1)/k);          % foldLength = 15
    P = randperm(size(inputs,1));           % random permutation containing all index of single data point
    iteration = 1;
    hyperparameterInformation = cell(1,1);

    cValues = [0.1, 1, 3, 10, 50];
    sigmaValues = [0.25, 1, 2, 5, 10];
    epsilonValues = [0.5 1, 3, 10];

    for sigmaIndex=1:length(sigmaValues)
        for cIndex=1:length(cValues)
            for eIndex=1:length(epsilonValues)
                rms_errors = zeros(1,k); 
                for i=1:k     % each iteration performs one time of training and CV
                    % retrieve training and testing dataset for i fold
                    [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, i, P, k, inputs, targets);  
                    % training SVM
                    SVM = fitrsvm(trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', sigmaValues(sigmaIndex), 'BoxConstraint', cValues(cIndex), 'Epsilon', epsilonValues(eIndex));
                    sv = size(SVM.SupportVectors,1);
                    % prediction and evaluation. 
                    predictions = predict(SVM,testingInputs);
                    rms = rmsCal(predictions, testingTargets);
                    rms_errors(i) = rms;
                end     % end of fold cross validation 

            % determine mean values
            meanRMS = mean(rms_errors);

            % store hyperparameters into a structure
            struct.cValue = cValues(cIndex);
            struct.sigmaValue = sigmaValues(sigmaIndex);
            struct.epsilonValue = epsilonValues(eIndex);
            struct.supportVectors = size(SVM.SupportVectors,1);
            struct.supportVectorsRatio = struct.supportVectors / size(inputs,1);
            struct.rms = meanRMS;
            hyperparameterInformation{iteration} = struct;

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

end

%Start of confusion matrix
function cm = confusion_matrix(outputs, labels)
    tp=0;tn=0;fp=0;fn=0;
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
function [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, i, P, k, inputs, targets)
    if i == k
        validPerm = P(foldLength*(k-1)+1:end);
    else
        validPerm = P((i-1)*foldLength+1:i*foldLength); % extract the indexes of validation data
    end
    % the remaining are indexes of training data
    if i==1
        trainPerm = P(foldLength+1:end);
    elseif i==k
        trainPerm = P(1:foldLength*(k-1));
    else
        trainPerm1 = P(1:(i-1)*foldLength);
        trainPerm2 = P(i*foldLength+1:end);
        trainPerm = [trainPerm1,trainPerm2];
    end

    trainingTargets = targets(trainPerm, :);
    trainingInputs = inputs(trainPerm, :);
    % find the values of features and labels with their corresponding indexes
    testingTargets = targets(validPerm, :);
    testingInputs = inputs(validPerm, :);
end


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
