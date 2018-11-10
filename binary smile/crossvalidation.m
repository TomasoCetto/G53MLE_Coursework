load('facialPoints.mat');
load('labels.mat');

inputs = reshape(points,[66*2, 150])';
targets = labels';

function cv = crossvalidation(inputs,targets)

    c = cvpartition(length(inputs),'KFold', k);

    trees = cell(1,10)

    for i=1:c.NumTestSets
        trIDX = training(c,i);
        teIDX = test(c,i);

        trainingInputs = inputs(:,trIDX);
        trainingTargets = targets(:,trIDX);

        testingInputs = inputs(:,teIDX);
        testingTargets = targets(:,teIDX);


        %Create 10 trained trees
        [trees{i},tr] = decisionTreeLearning(trainingInputs,trainingTargets);
        %tree = trees{i}
        %use the trained tree to classify my data 
        outputArray = [];
        for i=1:length(testingInputs)
            value = evaluateOneSample(trees{i},testingInputs(i,:));
            outputArray = [outputArray,value];
        end


        %recall and precision rates 
    end
end



function output = evaluateOneSample(tree, input)
    if tree.kids == []
        output = tree.class
        return
    else if input(:,tree.attribute) > tree.threshold
            evaluateOneSample(tree.kids{1},input)
    else
            evaluateOneSample(tree.kids{2},input)
        end
     
end
    
