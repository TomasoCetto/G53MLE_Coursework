function [tree, bestGain] = DecisionTreeLearning(indices)
    %UNTITLED3 Summary of this function goes here
    %   Detailed explanation goes here

    %fprintf("DTL: start\n");

    % init variables
    global features;
    global labels;
    tree = initTreeStruct();
    bestAttribute = 0;
    bestThreshold = 0;
    bestGain = 0;

    % check indices
    if size(indices, 2) > size(features,1) || size(indices, 2) > size(labels,1)
        fprintf("DTL: over indices.\n");
        return;
    end
    
    % check sample's size => TODO: move out
    if size(features(indices,:),1) ~= size(labels(indices),1)
       fprintf("DTL: features and labels do not match.\n");
       return;
    end

    % check same labels +> TODO: move out
    if sum(labels(indices)) == size(labels(indices),1) || sum(labels(indices)) == 0  
        %fprintf("DTL: all same labels\n");
        tree.class = sum(labels(indices))/size(labels(indices),1);
        return;  
    end
    
    % DO ChooseAttribute
    [bestAttribute, bestThreshold, bestGain] = ChooseAttribute(indices);
    
    % split
    [leftTreeIdc, rightTreeIdc] = splitSample(indices, bestThreshold, bestAttribute);
    ChildrenIdc = {leftTreeIdc rightTreeIdc};
    kidNum = size(ChildrenIdc,2);
%     kidNum = 1;
    subtrees = {};
    for kidCount = 1: kidNum
        % Check empty sample
        if size(ChildrenIdc{kidCount},2) == 0
            fprintf("DTL: do major value\n");
            tree.class = majority_value(ChildrenIdc{kidCount});
            return;
        else
            subtrees{kidCount} = DecisionTreeLearning(ChildrenIdc{kidCount});
        end
    end
    
    % return tree
    tree.kids = subtrees;
    tree.op = "x" + num2str(bestAttribute) + " < " + num2str(bestThreshold) + "   x" + num2str(bestAttribute) + " >= " + num2str(bestThreshold);
    tree.attribute = bestAttribute;
    tree.threshold = bestThreshold;
    %fprintf("DTL: end\n");
    return;
end

%   -   -   -   Fucntions-   -   -   -

function label = majority_value(indices)
    global labels;
    n=0;
    p=0;
    for i=1:length(labels(indices))
        if labels(indices(i)) == 1
            p = p + 1;
        else
            n = n +1;
        end
    end
    if n<p
        label=1;
    else
        label=0;
    end
end
