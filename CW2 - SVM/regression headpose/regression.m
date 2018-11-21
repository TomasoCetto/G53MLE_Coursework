load('facialPoints.mat');
load('headpose.mat');

labels = pose(:,6);
labels = labels';
points = reshape(points, [66*2,8955]);

hiddenLayers = 10;
k = 10;

c = cvpartition(length(points),'KFold', k); %split the data into 10 folds
perf = zeros(c.NumTestSets,1);

%create the net
net = patternnet(10,'trainlm','mse');
net.trainParam.epochs = 1000;


%create a cell of 10 neural nets
numNN = 10;
nets = cell(1,numNN);

rms_errors = zeros(1,k);

for i=1:c.NumTestSets
    %obtain indexes for the train/test split
    trIDX = training(c,i);
    teIDX = test(c,i);
    
    %obtain training inputs and associated labels
    trainingInputs = points(:,trIDX);
    trainingTargets = labels(:,trIDX);
    
    %%obtain test inputs and associated labels
    testingInputs = points(:,teIDX);
    testingTargets = labels(:,teIDX);
    
    % Set up Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 20/100;
    net.divideParam.testRatio = 0/100;
    net.divideParam.lr = 0.01;
    
    %Create a neural network, train it on the training data for this fold
    [nets{i},tr] = train(net,trainingInputs,trainingTargets);
    
    fprintf('Training %d/%d\n', i, k)
    
    %Test the networks performance on test data
    outputs = nets{i}(testingInputs);
    
    %calculate the rms error between predictions and targets and store it
    rms =(1/(2*length(testingTargets)))*sum(power((outputs - testingTargets),2));
    rms_errors(i) = rms
    %calculate and store network performance
    performance = perform(nets{i},testingTargets,outputs);
    perf(i) = performance;
    %View the net
 	view(nets{i});
    
    
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

% accuracies
