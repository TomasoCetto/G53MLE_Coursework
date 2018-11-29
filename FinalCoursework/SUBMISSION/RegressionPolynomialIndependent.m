clear;  % ***Clear all previous data before executing this code***

% prepare data
load('facialPoints.mat');
load('headpose.mat');

inputs = reshape(points,[66*2,8955])';   % samples x features
targets = pose(:,6);                       % samples x classes
P = randperm(size(inputs,1));

% cross validation initial
k = 10;
numOfFeatures = size(inputs,2);         % inputWidth = 132
foldLength = round(size(inputs,1)/k);          % foldLength = 896

%Outerloop returns
rms_errors = zeros(1,10);
predictionsMatrix = cell(1,10);
scoreMatrix = cell(1,10);
supportVectors = zeros(1,10);
SVMPolynomialPredictionsMat = [];
allModels = {};
predictionCat = [];

%outerloop: can use parfor loop to run them in parallel
for i = 1:k

    [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, i, P, k, inputs , targets);  
    [bestCValue, bestEpsilonValue, bestPOrder] = innerLoop(trainingInputs,trainingTargets,3);
    %test the model
    
    currentModel = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', bestPOrder, 'BoxConstraint', bestCValue, 'Epsilon', bestEpsilonValue);
    %calculate f1 of this fold
    allModels{i} = currentModel;
    predictions = predict(currentModel,testingInputs);
    rms = rmsCal(predictions, testingTargets);
    rms_errors(i) = rms;
    predictionsMatrix{i} = predictions;
    supportVectors(i) = size(currentModel.SupportVectors,1);
end
predictionCat = vertcat(predictionCat,predictionsMatrix{:});


%inner loop function
function [bestCValue, bestEpsilonValue, bestPOrder] = innerLoop(inputs,targets,k)
    foldLength = size(inputs,1)/k;         
    newP = randperm(size(inputs,1));         
    
    % %determine best epsilon
    cValues = [0.1, 5 , 20];
    cValueArray = [];
    for cIndex=1:length(cValues)
        rms_errors = zeros(1,k); 
        for i=1:k     % each iteration performs one time of training and CV --> taking 10 minutes
            % retrieve training and testing dataset for i fold
            [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, i, newP, k, inputs, targets);  

            %train model 
            SVM = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2, 'BoxConstraint', cValues(cIndex), 'Epsilon', 1);
            sv = size(SVM.SupportVectors,1);
            % prediction and evaluation. 
            predictions = predict(SVM,testingInputs);
            rms = rmsCal(predictions, testingTargets);
            rms_errors(i) = rms;        

        end     % end of fold cross validation 

        %return best cvalue
        cValueArray(cIndex) = mean(rms_errors);
    end
    
    %find best cValue
    [bestRmsValue, bestCIndex] = min(cValueArray);
        
    epsilonValues = [0.5, 3, 10]
    epsilonValueArray = []
    for eIndex=1:length(epsilonValues)
        rms_errors = zeros(1,k); 
        for i=1:k     % each iteration performs one time of training and CV --> taking 10 minutes
            % retrieve training and testing dataset for i fold
            [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, i, newP, k, inputs, targets);  

            %train model 
            SVM = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2, 'BoxConstraint', cValues(bestCIndex), 'Epsilon', epsilonValues(eIndex));
            sv = size(SVM.SupportVectors,1);
            % prediction and evaluation. 
            predictions = predict(SVM,testingInputs);
            rms = rmsCal(predictions, testingTargets);
            rms_errors(i) = rms;        
        end     % end of fold cross validation 
       
        %return best epsilon
        epsilonValueArray(eIndex) = mean(rms_errors);
    end  
    
    %find best cValue
    [bestRmsValue, bestEIndex] = min(epsilonValueArray);
    
    pOrder = [2,3];
    pOrderArray = [];
    for pIndex=1:length(pOrder)
        rms_errors = zeros(1,k); 
        for i=1:k     % each iteration performs one time of training and CV --> taking 10 minutes
            % retrieve training and testing dataset for i fold
            [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, i, P, k, inputs, targets);  

            %train model 
            SVM = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', pOrder(pIndex), 'BoxConstraint', cValues(bestCIndex), 'Epsilon', epsilonValues(bestEIndex));
            sv = size(SVM.SupportVectors,1);
            % prediction and evaluation. 
            predictions = predict(SVM,testingInputs);
            rms = rmsCal(predictions, testingTargets);
            rms_errors(i) = rms;        
        end     % end of fold cross validation 

        %return best epsilon
        pOrderArray(pIndex) = mean(rms_errors);
    end  

    [bestRmsValue, bestPIndex] = min(pOrderArray);
    
    bestCValue = cValues(bestCIndex);
    bestPOrder = pOrder(bestPIndex);
    bestEpsilonValue = epsilonValues(bestEIndex);
    
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