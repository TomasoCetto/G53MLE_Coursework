function [predicts] = DecisionTreeSim(tree, indices)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

    
    %fprintf("DTS: start\n");
    % init variables
    global labels;
    predicts = zeros(size(labels,1),1);
    % dummy
%     predicts(1,1) = 1;
%     predicts(2,1) = 1;
%     return
    
    % check if tree = leaf by kids = {emptry}
    if size(tree.kids, 2) == 0
        for labelNum = 1: size(indices, 2)
            predicts(indices(labelNum),1) = tree.class;
        end
        %fprintf("DTS: end at leaf\n");
        return
    end
    
    % Split
    [leftIdc rightIdc] = splitSample(indices, tree.threshold, tree.attribute);
    ChildrenIdc = {leftIdc rightIdc};
    kidNum = size(ChildrenIdc, 2);
    for kidCount = 1: kidNum
        predicts = predicts | DecisionTreeSim(tree.kids{kidCount}, ChildrenIdc{kidCount});
    end
    
    %fprintf("DTS: end\n");
end

