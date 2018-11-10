load('facialPoints.mat');
load('labels.mat');

inputs = reshape(points,[66*2, 150])';
targets = labels';

[p, r, f] = crossvalidations(inputs, targets);

function [precisions, recalls, f1s] = crossvalidations(inputs,targets)

    k = 10;
    c = cvpartition(length(inputs),'KFold', k);

    trees = cell(1,10)
    recalls = zeros(1,k);
    precisions = zeros(1,k);
    f1s = zeros(1,k);

    for i=1:c.NumTestSets
        trIDX = training(c,i);
        teIDX = test(c,i);

        trainingInputs = inputs(trIDX,:);
        trainingTargets = targets(trIDX);

        testingInputs = inputs(teIDX,:);
        testingTargets = targets(teIDX);


        %Create 10 trained trees
        [trees{i},tr] = decisionTreeLearning(trainingInputs,trainingTargets);
        %tree = trees{i}
        %use the trained tree to classify my data 
        outputArray = [];
        for i=1:length(testingInputs)
            value = evaluateOneSample(trees{i},testingInputs(i,:));
            outputArray = [outputArray,value];
        end

        confusion = confusion_matrix(outputArray, testingTargets);

        recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2));
        precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));
        f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)));
        %recall and precision rates 
    end
    recalls
    precisions
    f1s
end



function output = evaluateOneSample(tree, input)
    if tree.kids == []
        output = tree.class
        return
    elseif input(:,tree.attribute) > tree.threshold
            evaluateOneSample(tree.kids{1},input)
    else
            evaluateOneSample(tree.kids{2},input)
        % end
    end
end
    

function cm = confusion_matrix(outputs, labels)
    % cm = zeros(2);
    tp=0;tn=0;fp=0;fn=0;
    for i=1:length(outputs)
        if (labels(i) == 1) && (outputs(i)==1)
            tp=tp+1;
        elseif (labels(i) == 0) && (outputs(i)==0)
            tn=tn+1;
        elseif (labels(i) == 1) && (outputs(i)==0)
            fn=fn+1;
        else
            fp=fp+1
        end
    end
    cm = [tp, fn; fp, tn];
end