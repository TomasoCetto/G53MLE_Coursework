clear;  % ***Clear all previous data before executing this code***
disp("*Clear all previous data");

% prepare data
load('facialPoints.mat');
load('headpose.mat');
load('PermutationRegression.mat');

inputs = reshape(points,[66*2,8955])';      % samples x features
targets = pose(:,6);                        % samples x classes

% cross validation initial
k = 10;
numOfExamples = (k-1)*size(inputs,1)/k;     % numOfExamples = 896*9
numOfFeatures = size(inputs,2);             % inputWidth = 132
foldLength = round(size(inputs,1)/k);          % foldLength = 896

% variables
global doDisp;  
doDisp = false;  % display flag
tic;    % start timer

%Outerloop returns
rms_errors = zeros(1,10);
predictionsMatrix = cell(1,10);
scoreMatrix = cell(1,10);
supportVectors = zeros(1,10);
SVMPolynomialPredictionsMat = []
allModels = {}

%outerloop
parfor i = 1:k

    [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs , targets);  
    display("cv partition done");
    [bestCValue, bestEpsilonValue, bestPOrder] = innerLoop(trainingInputs,trainingTargets,3);

    %test the model
    
    currentModel = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', bestPOrder, 'BoxConstraint', bestCValue, 'Epsilon', bestEpsilonValue);
    %calculate f1 of this fold
    allModels{i} = currentModel;

    predictions = predict(currentModel,testingInputs);

    rms = rmsCal(predictions, testingTargets);

    rms_errors(i) = rms;

    tPrint("rms = " + rms_errors(i));

    predictionsMatrix{i} = predictions;

    supportVectors(i) = size(currentModel.SupportVectors,1);

    SVMPolynomialPredictionsMat = [SVMPolynomialPredictionsMat predictions];
    
    display("FoldNumber " + i + "completed");
end

SVMPolynomialPredictionsMat = SVMPolynomialPredictionsMat(:);

fprintf("\n");



disp("time stamp: " + toc + " sec");    % stamp total training duration


%   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -



%   -   -   -   -   -   - ???  FUNCTIONS ???    -   -   -   -   -   -   -



%   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -


%inner loop function

function [bestCValue, bestEpsilonValue, bestPOrder] = innerLoop(inputs,targets,k)

    numOfExamples = (k-1)*size(inputs,1)/k; 
    foldLength = size(inputs,1)/k;         
    P = randperm(size(inputs,1));         
    
    % %determine best epsilon
    cValues = [0.1, 5 , 20]
    cValueArray = [];
    for cIndex=1:length(cValues)
        rms_errors = zeros(1,k); 
        for i=1:k     % each iteration performs one time of training and CV --> taking 10 minutes
            display("inner k 3-fold started");
            % retrieve training and testing dataset for i fold
            [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs, targets);  
            display("inner cv partition done");

            %train model 
            SVM = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2, 'BoxConstraint', cValues(cIndex), 'Epsilon', 1);
            display("finsihed training and in fold : " + k + " of CValues")
            
            sv = size(SVM.SupportVectors,1);

            tPrint("sv = " + sv + "(" + size(SVM.SupportVectors,1)/size(SVM.SupportVectors,2)*100 + "%)");

            % prediction and evaluation. 

            predictions = predict(SVM,testingInputs);

            rms = rmsCal(predictions, testingTargets);

            rms_errors(i) = rms;        

            tPrint("rms_errors = " + rms_errors(i));

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
            disp("inner k 3-fold started");
            % retrieve training and testing dataset for i fold
            [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs, targets);  
            disp("inner cv partition done");

            %train model 
            SVM = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2, 'BoxConstraint', cValues(bestCIndex), 'Epsilon', epsilonValues(eIndex));

            display("finsihed training and in fold : " + k + " of epsilon")

            sv = size(SVM.SupportVectors,1);

            tPrint("sv = " + sv + "(" + size(SVM.SupportVectors,1)/size(SVM.SupportVectors,2)*100 + "%)");

            % prediction and evaluation. 

            predictions = predict(SVM,testingInputs);

            rms = rmsCal(predictions, testingTargets);

            rms_errors(i) = rms;        

            tPrint("rms_errors = " + rms_errors(i));

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
            disp("inner k 3-fold started");
            % retrieve training and testing dataset for i fold
            [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs, targets);  
            disp("inner cv partition done");

            %train model 
            SVM = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', pOrder(pIndex), 'BoxConstraint', cValues(bestCIndex), 'Epsilon', epsilonValues(bestEIndex));

            display("finsihed training and in fold : " + k + " of pOrder")

            sv = size(SVM.SupportVectors,1);

            tPrint("sv = " + sv + "(" + size(SVM.SupportVectors,1)/size(SVM.SupportVectors,2)*100 + "%)");

            % prediction and evaluation. 

            predictions = predict(SVM,testingInputs);

            rms = rmsCal(predictions, testingTargets);

            rms_errors(i) = rms;        

            tPrint("rms_errors = " + rms_errors(i));

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

    % Set up Division of Data for Training, Validation, Testing

    % find the values of features and labels with their corresponding indexes

    trainingTargets = targets( trainPerm, :);
    trainingInputs = inputs( trainPerm, :);
    % find the values of features and labels with their corresponding indexes
    testingTargets = targets( validPerm, :);
    testingInputs = inputs( validPerm, :);
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
