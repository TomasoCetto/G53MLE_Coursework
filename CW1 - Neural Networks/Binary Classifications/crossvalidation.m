function cv = crossvalidation(points,labels,net,nets,k)

c = cvpartition(length(points),'KFold', k);
perf = zeros(c.NumTestSets,1);

accuracies = zeros(1,k)

for i=1:c.NumTestSets
    trIDX = training(c,i);
    teIDX = test(c,i);
    
    trainingInputs = points(:,trIDX);
    trainingTargets = labels(:,trIDX);
    
    testingInputs = points(:,teIDX);
    testingTargets = labels(:,teIDX);
    
    
    %Create 10 neural nets using this folds data only
    [nets{i},tr] = train(net,trainingInputs,trainingTargets);
    
    fprintf('Training %d/%d\n', i, k)
    
    %Test the networks performance on test data
    outputs = nets{i}(testingInputs);
    % errors = gsubtract(testingTargets,outputs);
    performance = perform(nets{i},testingTargets,outputs)
    perf(i) = performance;
    %View the net
 	% view(nets{i});
    
    r = round(outputs);
    % s = sim(net,validInputs);
    differences = r - testingTargets
    % differences = gsubtract(r,testingTargets)
    differences = abs(differences);
    accuracy = 1 - (sum(differences)/c.TestSize(1));
    accuracies(1,i) = accuracy
    % Plots
    % Uncomment these lines to enable various plots.
    %figure, plotperform(tr)
    % figure, plottrainstate(tr)
    % figure, plotconfusion(testingTargets,outputs);
    % crossval(testingTargets,outputs)
    % figure, ploterrhist(errors)
    % err(i) = performance;
    %% 
    
end

cv = accuracies;

% average = sum(accuracy)/k
