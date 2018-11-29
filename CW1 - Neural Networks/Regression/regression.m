clear all

load('facialPoints.mat');
load('headpose.mat');
load('Perm.mat')

labels = pose(:,6);
labels = labels';
points = reshape(points, [66*2,8955]);

hiddenLayers = 10;
k = 10;

trainLength = 8955 - 896;   % trainLength = 135
inputWidth = size(points,1);            % inputWidth = 132
foldLength = 896;          % foldLength = 15
% P = P           % random permutation containing all index of single data point
perf = zeros(k,1);

%create the net
net = patternnet(10,'trainlm','mse');
net.trainParam.epochs = 1000;


%create a cell of 10 neural nets
numNN = 10;
nets = cell(1,numNN);

rms_errors = zeros(1,k);

RegPrediction = [];

for i=1:k-1
    validPerm = P((i-1)*foldLength+1:i*foldLength); % extract the indexes of validation data

    % the remaining are indexes of training data
    if i==1
        trainPerm = P(foldLength+1:end);
    elseif i==k
        trainPerm = P(1:trainLength);
    else
        trainPerm1 = P(1:(i-1)*foldLength);
        trainPerm2 = P(i*foldLength+1:end);
        trainPerm = [trainPerm1,trainPerm2];
    end
    
    % initialize the features and labels sets of training data
    trainingInputs = zeros(inputWidth,trainLength);
    trainingTargets = zeros(1, trainLength);
    
    % find the values of features and labels with their corresponding indexes
    for j=1:trainLength
        trainingTargets(j) = labels(trainPerm(j));
        trainingInputs(:,j) = points(:,trainPerm(j));
    end
    
    % Set up Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 20/100;
    net.divideParam.testRatio = 0/100;
    net.divideParam.lr = 0.01;
    
    %Create a neural network, train it on the training data for this fold
    [nets{i},tr] = train(net,trainingInputs,trainingTargets);
    
    fprintf('Training %d/%d\n', i, k)

    testingInputs = zeros(inputWidth,foldLength);
    testingTargets = zeros(1, foldLength);
    for j=1:foldLength
        testingTargets(j) = labels(validPerm(j));
        testingInputs(:,j) = points(:,validPerm(j));
    end
    
    %Test the networks performance on test data
    outputs = nets{i}(testingInputs);
    RegPrediction = [RegPrediction, outputs];
    
    %calculate the rms error between predictions and targets and store it
    rms =(1/(2*length(testingTargets)))*sum(power((outputs - testingTargets),2));
    rms_errors(i) = rms;
    %calculate and store network performance
    performance = perform(nets{i},testingTargets,outputs);
    perf(i) = performance;
    %View the net
 	% view(nets{i});
    
    
    % Plots
    % Uncomment these lines to enable various plots.
    %figure, plotperform(tr)
    % figure, plottrainstate(tr)
    %figure, plotconfusion(testingTargets,outputs);
    % crossval(testingTargets,outputs)
    % figure, ploterrhist(errors)
    % err(i) = performance;
    %% 
    
end

newFoldLength = 891
newTrainLength = 8064

validPerm = P(8065:end); % extract the indexes of validation data

trainPerm = P(1:newTrainLength);

% initialize the features and labels sets of training data
trainingInputs = zeros(inputWidth,trainLength);
trainingTargets = zeros(1, trainLength);

% find the values of features and labels with their corresponding indexes
for j=1:newTrainLength
    trainingTargets(j) = labels(trainPerm(j));
    trainingInputs(:,j) = points(:,trainPerm(j));
end

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 0/100;
net.divideParam.lr = 0.01;

%Create a neural network, train it on the training data for this fold
[nets{10},tr] = train(net,trainingInputs,trainingTargets);

fprintf('Training %d/%d\n', i, k)

testingInputs = zeros(inputWidth,newFoldLength);
testingTargets = zeros(1, newFoldLength);
for j=1:newFoldLength
    testingTargets(j) = labels(validPerm(j));
    testingInputs(:,j) = points(:,validPerm(j));
end

%Test the networks performance on test data
outputs = nets{10}(testingInputs);
RegPrediction = [RegPrediction, outputs];

%calculate the rms error between predictions and targets and store it
rms =(1/(2*length(testingTargets)))*sum(power((outputs - testingTargets),2));
rms_errors(10) = rms;
%calculate and store network performance
performance = perform(nets{10},testingTargets,outputs);
perf(10) = performance;
%View the net
% view(nets{i});


RegPrediction;
rms_errors

% accuracies
