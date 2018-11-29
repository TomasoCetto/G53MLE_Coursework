clear;  % Clear all previous data before executing this code

% prepare data
load('facialPoints.mat');
load('headpose.mat');
inputs = reshape(points,[66*2,8955])';   % samples x features
targets = pose(:,6);                     % samples x class

% cross-validation initial variables
k = 10; % 10 fold
numOfExamples = (k-1)*size(inputs,1)/k; 
numOfFeatures = size(inputs,2);        
foldLength = round(size(inputs,1)/k);  
P = randperm(size(inputs,1));

% Outerloop (10 cross-validation loop) return variables
rms_errors = zeros(1,10);
meanRMS = 0;
predictionsMatrix = cell(1,10);
supportVectors = zeros(1,10);
meanSV = 0;
predictionCat = [];
SVMs = {};
bestHypers = {};

% outerloop: 10 cross validation
for i = 1:k
    
    % retieving training and test subdataset
    [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, i, P, k, inputs , targets);  

    % Do inner cross validation and return the best hyperparamters
    currentModelParameters = innerLoop(trainingInputs,trainingTargets,3);
    bestHypers{i} = currentModelParameters;
    % train the model with the best hyperparameter
    currentModel = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'linear', 'BoxConstraint', currentModelParameters.cValue, 'Epsilon', currentModelParameters.epsilonValue);
    SVMs{i} = currentModel;

    % test the model, do prediction, and calculate root mean square (RMS) for this fold
    predictions = predict(currentModel,testingInputs);
    rms = rmsCal(predictions, testingTargets);
    rms_errors(i) = rms;
    predictionsMatrix{i} = predictions;
    supportVectors(i) = size(currentModel.SupportVectors,1);

end

% 10 cross validation results: predictions, average RMS and average number
% of support vectors
predictionCat = vertcat(predictionCat, predictionsMatrix{:});   % concatenate all predictions from all folds
meanRMS = mean(rms_errors);
meanSV = mean(supportVectors);


%inner-loop (inner cross validation) function
function [bestHyperParameter] = innerLoop(inputs,targets,k)

    % initial variables for 3 cross validation
    numOfExamples = (k-1)*size(inputs,1)/k; % numOfExamples = 135
    foldLength = round(size(inputs,1)/k);          % foldLength = 15
    P = randperm(size(inputs,1));           % random permutation containing all index of single data point
    
    % record
    hyperparameterInformation = cell(1,1);
    
    % other variables
    iteration = 1;

    % Box constraint values
    cValues = [10 20];
    % Epsilon 
    epsilonValues = [1 1.5];

    for cIndex=1:length(cValues)
        for eIndex=1:length(epsilonValues)
            
            % record errors from all 3 fold cross validation
            rms_errors = zeros(1,k); 
            
            % Do 3 inner fold validation
            for i=1:k     % each iteration performs one time of training and CV
                
                % retrieve training and testing dataset for i fold
                [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, i, P, k, inputs, targets);  

                % training SVM
                SVM = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'linear', 'BoxConstraint', cValues(cIndex), 'Epsilon', epsilonValues(eIndex));
                sv = size(SVM.SupportVectors,1);

                % prediction and evaluation. 
                predictions = predict(SVM,testingInputs);
                rms = rmsCal(predictions, testingTargets);
                rms_errors(i) = rms;

            end

        % determine average RMS
        meanRMS = mean(rms_errors);

        % store hyperparameters, number of support vectors, and the average RMS into a structure
        struct.cValue = cValues(cIndex);
        struct.epsilonValue = epsilonValues(eIndex);
        struct.supportVectors = size(SVM.SupportVectors,1);
        struct.supportVectorsRatio = struct.supportVectors / size(trainingInputs,1);
        struct.rms = meanRMS;
        hyperparameterInformation{iteration} = struct;

        iteration = iteration + 1;

        end
    end

    % select the best hyperparameter from the lowest RMS
    bestRMS = 100;
    for i=1:length(hyperparameterInformation)
        currentRMS = hyperparameterInformation{i}.rms;
        if currentRMS < bestRMS
            bestRMS = currentRMS;
            bestHyperParameter = hyperparameterInformation{i};
        end
    end
end

% Partition for cross validation
function [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, i, P, k, inputs, targets)

    % Indice of validation data
    if i == k
        validPerm = P(foldLength*(k-1)+1:end);
    else
        validPerm = P((i-1)*foldLength+1:i*foldLength); % extract the indexes of validation data
    end
    
    % the remaining are indice of training data
    if i==1
        trainPerm = P(foldLength+1:end);
    elseif i==k
        trainPerm = P(1:foldLength*(k-1));
    else
        trainPerm1 = P(1:(i-1)*foldLength);
        trainPerm2 = P(i*foldLength+1:end);
        trainPerm = [trainPerm1,trainPerm2];
    end

    % find the values of features and labels with their corresponding
    % indice
    trainingTargets = targets( trainPerm, :);
    trainingInputs = inputs( trainPerm, :);

    % find the values of features and labels with their corresponding
    % indice
    testingTargets = targets( validPerm, :);
    testingInputs = inputs( validPerm, :);
end

% Root mean square (RMS) calculation
function [rms] = rmsCal(outputs, testingTargets)
    total = 0;
    differences = outputs - testingTargets;
    for j=1:length(differences)
        square = power(differences(j), 2);
        total = total + square;
    end
    ms = (1/length(testingTargets))*total;
    rms = sqrt(ms);
end
