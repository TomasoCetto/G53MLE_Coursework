function [bestFeature, bestThreshold, bestGain] = ChooseAttribute(indices)
    %UNTITLED6 Summary of this function goes here
    %   Detailed explanation goes here

    %fprintf("ChooseAttribute: start\n");

    % initail variables ---
    global features;
    maxSample = size(features(indices, :), 1);
    maxFeature = size(features(indices, :), 2);
    [pParent, nParent] = posNegCount(indices);
    % ---

    % storage
    bestFeature = 0;
    bestThreshold = 0.0;
    bestGain = 0.0;

    % Choose  the Best Attribute

        for featureCount = 1: maxFeature    % feature loop
           for sampleCount = 1: maxSample      % sample loop
               testThreshold = features(indices(sampleCount), featureCount);   
               % split 
               [leftIdc, rightIdc] = splitSample(indices, testThreshold, featureCount);
               % count positive and negative
               pChildren = [0 0];   % [left right]
               nChildren = [0 0];
               [pChildren(1), nChildren(1)] = posNegCount(leftIdc);
               [pChildren(2), nChildren(2)] = posNegCount(rightIdc);
               % check gain
               testGain = Gain(pParent, nParent, pChildren, nChildren); % TODO: move calulation in parent out
               if testGain > bestGain
                    bestGain = testGain;
                    bestFeature = featureCount;
                    bestThreshold = testThreshold;
               end
           end
        end

    %Choose the Best Attribute ----


        %fprintf("ChooseAttribute: end\n");

end

%   -   -   -   Fucntions-   -   -   -

function [pCount , nCount] = posNegCount(indices)
    global labels;
    pCount = 0;
    nCount = 0;
    for i = 1: size(indices, 2)
        if labels(indices(i)) == 1
            pCount = pCount + 1;
        else
            nCount = nCount + 1;
        end
    end
end