
clear;  % ***Clear all previous data before executing this code***
disp("*Clear all previous data");

% prepare data
load('facialPoints.mat');
load('headpose.mat');
global inputs;  % declare global variable to be accessed from anywhere
global targets; % declare global variable to be accessed from anywhere
inputs = reshape(points,[66*2,8955])';   % samples x features
targets = pose(:,6);                       % samples x classes

% cross validation initial
k = 3;
numOfExamples = (k-1)*size(inputs,1)/k; % numOfExamples = 135
numOfFeatures = size(inputs,2);         % inputWidth = 132
foldLength = size(inputs,1)/k;          % foldLength = 15
P = randperm(size(inputs,1));           % random permutation containing all index of single data point

% Record variable
rms_errors = zeros(1,k);                % accuracies from each fold
%hyperparameterInformation = cell(1,10); % info of hyperparameters of each iteration
hyperparameterInformation = []; % info of hyperparameters of each iteration

% variables
global doDisp;  
doDisp = true;  % display flag

% SMV values
pOrder = 1;         
minKScale = 0.001;  % Kernel Scale: min-max = 1e-3 to 1e3
maxKScale = 1000;
stepKScale = 0.5;
stepKScale2 = 200;
%kernelScale = minKScale;
minC = 1;       % c value: min-max = 1e-5 to 1e5
maxC = 1;
stepC = 1000;
%cValue = minC;
%epsilon = 0;
minEps = 1;
maxEps = 1;
stepEps = 0.5;
% Evaluation variables
precision = 0;
recall = 0;
f1 = 0;
rms = 0;

tic;    % start timer
iteration = 1;  % initial iteration
for cValue = minC: stepC: maxC
    kernelScale = minKScale;    % initial kernelScale
    while kernelScale <= maxKScale
        epsilon = minEps;                % initial epsilon
        while epsilon <= maxEps
            % storage variables for each fold
            f1s = zeros(1,k);
            recalls = zeros(1,k);
            precisions = zeros(1,k);
            for i=1:k     % each iteration performs one time of training and CV

                % retrieve training and testing dataset for i fold
                [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k);  

                % training SVM
                %SVM = fitcsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', kernelScale, 'BoxConstraint', cValue);
                SVM = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', kernelScale, 'BoxConstraint', cValue, 'Epsilon', epsilon);
                tPrint("finish " + i + "-fold training SVM: " + tic + " sec");
                sv = size(SVM.SupportVectors,1);
                %tPrint("sv = " + sv + "(" + size(SVM.SupportVectors,1)/size(SVM.SupportVectors,2)*100 + "%)");

                % prediction and evaluation
                predictions = predict(SVM,testingInputs);
                rms = rmsCal(predictions, testingTargets);
                rms_errors(i) = rms;

            end     % end of fold cross validation 

            % determine mean values
            meanRMS = mean(rms_errors);

            % store hyperparameters into a structure
            tPrint("rms errors = [" + num2str(rms_errors) + "]");
            struct.rms = meanRMS;
            struct.epsilon = epsilon;
            struct.cValue = cValue;
            % struct.pOrder = 2;
            struct.kernelScale = kernelScale;
            struct.supportVectors = size(SVM.SupportVectors,1);
            struct.supportVectorsRatio = struct.supportVectors / numOfExamples;
            struct.predicted = predictions;
            hyperparameterInformation{iteration} = struct;

            epsilon = epsilon + stepEps;
            tPrint("C Value: " + cValue + " | " + "Kernel Scale: " + kernelScale + " | " +  "epsilon: " + epsilon);
            fprintf("|");
            iteration = iteration + 1;
        end     % end of epsilon loop
        
        % config variables before starting a new loop
        if kernelScale > 1
           stepKScale = stepKScale2; 
        end
        kernelScale = kernelScale + stepKScale;
        
    end     % end of kernel scale loop
end     % end of c-value loop
fprintf("\n");
disp("time stamp: " + toc + " sec");    % stamp total training duration

% find the best hyperparameter
bestRMS = 100;
for i=1:length(hyperparameterInformation)
    currentRMS = hyperparameterInformation{i}.rms;
    if currentRMS < bestRMS
        bestRMS = currentRMS;
        bestHyperParameter = hyperparameterInformation{i};
    end
end
disp("The best hyperparameter regarding to RMS: ");
disp(bestHyperParameter);


%                           End of process


%   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
%   -   -   -   -   -   - ???  FUNCTIONS ???    -   -   -   -   -   -   -
%   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -


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
end     % end of confusion_matrix

function [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k)
    global inputs;
    global targets;
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
end     % end of myCVPartition

function accuracy = accuracyCal(predictions, testingTargets, foldLength)    % calculate accuracy
    differences = abs(gsubtract(predictions,testingTargets));
    accuracy = 1 - sum(differences)/foldLength;
end     % end of accuracyCal

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