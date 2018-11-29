clear;
disp("*Clear all previous data");
% prepare data

load('facialPoints.mat');
load('labels.mat');

inputs = reshape(points,[66*2,150])';   % samples x features
targets = labels;                       % samples x classes

% cross validation initial

k = 10;
numOfExamples = (k-1)*size(inputs,1)/k; 
numOfFeatures = size(inputs,2);        
foldLength = size(inputs,1)/k;          
P = randperm(size(inputs,1));         

% variables

global doDisp;  
doDisp = false;  % display flag

iteration = 1;  % initial iteration

%Outerloop returns

recalls = zeros(1,10);
precisions = zeros(1,10);
f1s = zeros(1,10);
predictionsMatrix = cell(1,10);
scoreMatrix = cell(1,10);
supportVectors = zeros(1,10);
predictionCat = [];

%outerloop
for i = 1:10

    [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs , targets);  
    currentModelParameters = innerLoop(trainingInputs,trainingTargets,3);

    %test the model

    currentModel = fitcsvm(trainingInputs,trainingTargets, 'Standardize',true,'KernelFunction', 'polynomial', 'PolynomialOrder',currentModelParameters.pOrder, 'BoxConstraint',currentModelParameters.cValue); 

    %calculate f1 of this fold

    [predictions, score] = predict(currentModel,testingInputs);
    confusion = confusion_matrix(predictions, testingTargets);
    recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2));
    precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));
    f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)));
    predictionsMatrix{i} = predictions;
    scoreMatrix{i} = score;
    supportVectors(i) = size(currentModel.SupportVectors,1);

end

predictionCat = vertcat(predictionCat, predictionsMatrix{:});

%inner loop function

function [bestHyperParameter] = innerLoop(inputs,targets,k)

    numOfExamples = (k-1)*size(inputs,1)/k; % numOfExamples = 135
    foldLength = size(inputs,1)/k;          % foldLength = 15
    P = randperm(size(inputs,1));           % random permutation containing all index of single data point

    iteration = 1;

    hyperparameterInformation = cell(1,5);

    %Box constraint values
    cValues = [0.01, 0.05, 0.1, 0.5 , 1 , 5 , 10, 50, 100, 500, 1000];

    %polynomial order
    pOrder = [2,3,4];

    for pIndex=1:length(pOrder)

        for cIndex=1:length(cValues)

            f1s = zeros(1,k);
            recalls = zeros(1,k);
            precisions = zeros(1,k);

            for i=1:k     % each iteration performs one time of training and CV

                % retrieve training and testing dataset for i fold
                [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs, targets);  

                % training SVM

                SVM = fitcsvm(trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', pOrder(pIndex), 'BoxConstraint', cValues(cIndex));
                sv = size(SVM.SupportVectors,1);

                % prediction and evaluation. 

                predictions = predict(SVM,testingInputs);
                confusion = confusion_matrix(predictions, testingTargets);
                recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2));
                precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));
                f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)));

            end     % end of fold cross validation 

            % determine mean f1 value
            f1 = mean(f1s);

            % store hyperparameters into a structure
            struct.cValue = cValues(cIndex);
            struct.sigmaValue = sigmaValues(sigmaIndex);
            struct.supportVectors = size(SVM.SupportVectors,1);
            struct.supportVectorsRatio = struct.supportVectors / size(inputs,1);
            struct.f1 = f1;
            hyperparameterInformation{iteration} = struct;

            iteration = iteration + 1;

        end

    end



    % select the best f1
    bestf1 = 0;

    for i=1:length(hyperparameterInformation)

        currentf1 = hyperparameterInformation{i}.f1;

        if currentf1 > bestf1

            bestf1 = hyperparameterInformation{i}.f1;
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
