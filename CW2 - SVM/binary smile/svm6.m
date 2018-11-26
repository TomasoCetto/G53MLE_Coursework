
clear;  % ***Clear all previous data before executing this code***
disp("*Clear all previous data");

% prepare data
load('facialPoints.mat');
load('labels.mat');
global inputs;  % declare global variable to be accessed from anywhere
global targets; % declare global variable to be accessed from anywhere
inputs = reshape(points,[66*2,150])';   % samples x features
targets = labels;                       % samples x classes

% cross validation initial
k = 3;
numOfExamples = (k-1)*size(inputs,1)/k; % numOfExamples = 135
numOfFeatures = size(inputs,2);         % inputWidth = 132
foldLength = size(inputs,1)/k;          % foldLength = 15
P = randperm(size(inputs,1));           % random permutation containing all index of single data point

% Record variable
accuracies = zeros(1,k);                % accuracies from each fold
hyperparameterInformation = cell(1,10); % info of hyperparameters of each iteration

% variables
global doDisp;  
doDisp = false;  % display flag

% SMV values
pOrder = 1;         
minKScale = 0.001;  % Kernel Scale: min-max = 1e-3 to 1e3
maxKScale = 1000;
stepKScale = 0.1;
kernelScale = minKScale;
minC = 1;       % c value: min-max = 1e-5 to 1e5
maxC = 10000;
stepC = 500;
cValue = minC;
% Evaluation variables
precision = 0;
recall = 0;
f1 = 0;

tic;    % start timer
iteration = 1;  % initial iteration
for cValue = minC: stepC: maxC
    kernelScale = minKScale;    % initial kernelScale
    while kernelScale < maxKScale
        % storage variables for each fold
        f1s = zeros(1,k);
        recalls = zeros(1,k);
        precisions = zeros(1,k);
        for i=1:k     % each iteration performs one time of training and CV

            % retrieve training and testing dataset for i fold
            [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k);  

            % training SVM
            SVM = fitcsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', kernelScale, 'BoxConstraint', cValue);
            tPrint("finish " + i + "-fold training SVM: " + tic + " sec");
            sv = size(SVM.SupportVectors,1);
            tPrint("sv = " + sv + "(" + size(SVM.SupportVectors,1)/size(SVM.SupportVectors,2)*100 + "%)");

            % prediction and evaluation
            predictions = predict(SVM,testingInputs);
            confusion = confusion_matrix(predictions, testingTargets);
            recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2));
            precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));
            f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)));
            tPrint("f1 = " + f1s(i));

            % calculating performance, record average accuracy
            accuracies(i) = accuracyCal(predictions, testingTargets, foldLength);

        end     % end of fold cross validation 

        % determine mean values
        f1 = mean(f1s);
        precision = mean(precisions);
        recall = mean(recalls);
        
        % store hyperparameters into a structure
        tPrint("accuracies = [" + num2str(accuracies) + "]");
        average = mean(accuracies);
        tPrint("avevrage accuracy = " + average);
        struct.cValue = cValue;
        % struct.pOrder = 2;
        struct.kernelScale = kernelScale;
        struct.supportVectors = size(SVM.SupportVectors,1);
        struct.averageValue = average;
        struct.predicted = predictions;
        struct.f1 = f1;
        struct.precision = precision;
        struct.recall = recall;
        hyperparameterInformation{iteration} = struct;

        % config variables before starting a new loop
        if kernelScale > 1
           stepKScale = 100; 
        end
        kernelScale = kernelScale + stepKScale;
        iteration = iteration + 1;
        tPrint("C Value: " + cValue + " | Kernel Scale: " + kernelScale);
        fprintf("|");
    end     % end of kernel-scale loop
end     % end of c-value loop
fprintf("\n");
disp("time stamp: " + toc + " sec");    % stamp total training duration

% find the best hyperparameter
bestf1 = 0;
bestAcc = 0;
bestSV = numOfExamples;
x = [];
y = [];
z = [];
for i=1:length(hyperparameterInformation)
    currentf1 = hyperparameterInformation{i}.f1;
    if currentf1 > bestf1
        bestf1 = hyperparameterInformation{i}.f1;
        bestHyperParameter = hyperparameterInformation{i};
    end
    currentAcc = hyperparameterInformation{i}.averageValue;
    if currentAcc > bestAcc
        bestAcc = hyperparameterInformation{i}.averageValue;
        bestHyperParameter_acc = hyperparameterInformation{i};
    end
    currentSV = hyperparameterInformation{i}.supportVectors;
    if currentSV < bestSV
        bestSV = hyperparameterInformation{i}.supportVectors;
        bestHyperParameter_sv = hyperparameterInformation{i};
    end
end
disp("The best hyperparameter regarding to f1: ");
disp(bestHyperParameter);
disp("The best hyperparameter regarding to average accuracy: ");
disp(bestHyperParameter_acc);
disp("The best hyperparameter regarding to support vectors: ");
disp(bestHyperParameter_sv);


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
