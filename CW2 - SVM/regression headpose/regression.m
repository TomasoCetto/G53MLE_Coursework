load('facialPoints.mat');
load('headpose.mat');

labels = pose(:,6);
targets = labels';
inputs = reshape(points, [66*2,8955])';

hiddenLayers = 10;
k = 3;

numOfExamples = (k-1)*size(inputs,1)/k;   % numOfExamples = 135
numOfFeatures = size(inputs,2);            % inputWidth = 132
foldLength = size(inputs,1)/k;          % foldLength = 15
P = randperm(size(inputs,1));           % random permutation containing all index of single data point

perf = zeros(c.NumTestSets,1);

hyperparameterInformation = cell(1,40);

rms_errors = zeros(1,k);

errs = zeros(1,k);

for i=1:k
    %obtain indexes for the train/test split
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

    % initialize the features and labels sets of training data
    trainingInputs = zeros(numOfExamples,numOfFeatures);
    trainingTargets = zeros(1, numOfExamples);
    % find the values of features and labels with their corresponding indexes
    for j=1:numOfExamples
        trainingTargets(j) = targets(trainPerm(j));
        trainingInputs(j,:) = inputs(trainPerm(j),:);
    end

    % SVM = fitcsvm(trainingInputs,trainingTargets,'KernelFunction','linear','BoxConstraint', 1);
    % SVM = fitcsvm(trainingInputs,trainingTargets,'KernelFunction','rbf', 'KernelScale', 30,'BoxConstraint', 1, 'Epsilon', 0.3, 'OptimizeHyperparameters','BoxConstraint');
    SVM = fitrsvm(trainingInputs,trainingTargets,'Standardize',true, 'KernelFunction','rbf', 'KernelScale', 30,'BoxConstraint', 1, 'Epsilon', 0.3);
    
    % find the values of features and labels with their corresponding indexes
    testingInputs = zeros(foldLength,numOfFeatures);
    testingTargets = zeros(1, foldLength);
    for j=1:foldLength
        testingTargets(j) = targets(validPerm(j));
        testingInputs(j,:) = inputs(validPerm(j),:);
    end
    
    %Test the networks performance on test data
    outputs = predict(SVM,testingInputs)';
    % getting the output with testing inputs
    sv = size(SVM.SupportVectors,1);

    %calculate the rms error between predictions and targets and store it
    total = 0;
    differences = outputs - testingTargets;
    for j=1:length(differences)
        sqr = power(differences(j), 2);
        total = total + sqr;
    end
    ms = (1/foldLength)*total;
    rms = sqrt(ms);

    rms_errors(i) = rms;
    % err = immse(outputs,testingTargets);
    % errs(i) = sqrt(err);
    
    %calculate and store network performance
    % performance = perform(nets{i},testingTargets,outputs);
    % perf(i) = performance;
    
end

    struct.cValue = cValue;
    struct.pOrder = 2;
    struct.supportVectors = size(SVM.SupportVectors,1);
    struct.averageValue = mean(rms_errors);
    struct.predicted = outputs;
    hyperparameterInformation{1,iteration} = struct;
    iteration = iteration + 1;
    rms_errors
    errs
% accuracies
