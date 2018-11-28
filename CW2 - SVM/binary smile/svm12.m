% this code is just to backup the crossvalidaion with innerfold

load('facialPoints.mat');
load('labels.mat');

inputs = reshape(points,[66*2,150])';   % samples x features
targets = labels;                       % samples x classes

% cross validation initial
k = 10;
innerK = 3;
numOfExamples = (k-1)*size(inputs,1)/k; % numOfExamples = 135
numOfFeatures = size(inputs,2);         % inputWidth = 132
foldLength = size(inputs,1)/k;          % foldLength = 15
innerFoldLength = (k-1)*foldLength/innerK   % innerFoldLength = 45
P = randperm(size(inputs,1));           % random permutation containing all index of single data point

for i=1:k     % each iteration performs one time of training and CV

    % retrieve training and testing dataset for i fold
    % [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k);  
    [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(inputs, targets, foldLength, numOfExamples, i, P, k);
    
    innerP = randperm(numOfExamples);
    innerRecalls = zeros(1,innerK);
    innerPrecisions = zeros(1,innerK);
    innerF1s = zeros(1,innerK);
    innerSvs = zeros(1,innerK);
    for j=1:innerK          % this is the inner fold crossvalidation
        [innerTrainingInputs, innerTrainingTargets, innerTestingInputs, innerTestingTargets] = myCVPartition(trainingInputs, trainingTargets, innerFoldLength, innerFoldLength*(innerK-1)/k, j, innerP, innerK);
        SVM = fitcsvm(innerTrainingInputs, innerTrainingTargets, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', kernelScale, 'BoxConstraint', cValue);
        % tPrint("finish " + i + "-fold training SVM: " + tic + " sec");
        innerSvs(j) = size(SVM.SupportVectors,1);
        % tPrint("sv = " + sv + "(" + size(SVM.SupportVectors,1)/size(SVM.SupportVectors,2)*100 + "%)");
        innerPredictions = predict(SVM,innerTestingInputs);
        innerConfusion = confusion_matrix(innerPredictions, innerTestingInputs);
        innerRecalls(j) = innerConfusion(1,1)/(innerConfusion(1,1)+innerConfusion(1,2));
        innerPrecisions(j) = innerConfusion(1,1)/(innerConfusion(1,1)+innerConfusion(2,1));
        innerF1s(j) = 2*((innerPrecisions(j)*innerRecalls(j))/(innerPrecisions(j)+innerRecalls(j)));
        % tPrint("f1 = " + f1s(i));

    end

    f1s(i) = mean(innerF1s);

    % training SVM
    % SVM = fitcsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', kernelScale, 'BoxConstraint', cValue);
    % tPrint("finish " + i + "-fold training SVM: " + tic + " sec");
    % sv = size(SVM.SupportVectors,1);
    % tPrint("sv = " + sv + "(" + size(SVM.SupportVectors,1)/size(SVM.SupportVectors,2)*100 + "%)");

    % prediction and evaluation
    % predictions = predict(SVM,testingInputs);
    % confusion = confusion_matrix(predictions, testingTargets);
    % recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2));
    % precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));
    % f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)));
    % tPrint("f1 = " + f1s(i));

    % calculating performance, record average accuracy
    % accuracies(i) = accuracyCal(predictions, testingTargets, foldLength);

end     % end of fold cross validation 


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


function [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(inputs, targets,foldLength, numOfExamples, i, P, k)
    % global inputs;
    % global targets;
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