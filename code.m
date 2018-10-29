load('labels.mat')
load('facialPoints.mat')

points = reshape(points, [66*2,150]);
labels = labels';

hiddenLayers = 10;
k = 10;

c = cvpartition(length(points),'KFold', k);
perf = zeros(c.NumTestSets,1);

%create the net
net = patternnet(10,'trainlm','mse');
net.trainParam.epochs = 2000;


%create a cell of 10 neural nets
numNN = 10;
nets = cell(1,numNN);

accuracies = zeros(1,k);

for i=1:c.NumTestSets
    trIDX = training(c,i);
    teIDX = test(c,i);
    
    trainingInputs = points(:,trIDX);
    trainingTargets = labels(:,trIDX);
    
    testingInputs = points(:,teIDX);
    testingTargets = labels(:,teIDX);
    
    
    %Create 10 neural nets using this folds data only
    [nets{i},tr] = train(net,trainingInputs,trainingTargets);
    
    % Set up Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 20/100;
    net.divideParam.testRatio = 0/100;
    net.divideParam.lr = 0.01;
    
    fprintf('Training %d/%d\n', i, k)
    
    %Test the networks performance on test data
    outputs = nets{i}(testingInputs);
    errors = gsubtract(testingTargets,outputs);
    performance = perform(nets{i},testingTargets,outputs)
    perf(i) = performance;
    %View the net
 	  view(nets{i});
    
    r = round(outputs);
    % s = sim(net,validInputs);
    differences = r - testingTargets
    % differences = gsubtract(r,testingTargets)
    differences = abs(differences);
    accuracy = 1 - (sum(differences)/c.TestSize(1))
    accuracies(i) = accuracy
    % Plots
    % Uncomment these lines to enable various plots.
    %figure, plotperform(tr)
    % figure, plottrainstate(tr)
    figure, plotconfusion(testingTargets,outputs);
    % crossval(testingTargets,outputs)
    % figure, ploterrhist(errors)
    % err(i) = performance;
    %% 
    
end

% accuracies
