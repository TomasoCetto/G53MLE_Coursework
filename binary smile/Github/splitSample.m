function [leftTreeIdc, rightTreeIdc] = splitSample(indices, threshold, featureNum)
    global features;
    leftTreeIdc = [];     % open array**
    rightTreeIdc = [];    % open array**
    leftCount = 0;
    rightCount = 0;
    for sampleCount = 1: size(indices, 2)
        if(features(indices(sampleCount), featureNum) < threshold)  % go left
            leftCount = leftCount + 1;
            leftTreeIdc(leftCount) = indices(sampleCount);
        else        % go right
            rightCount = rightCount + 1;
            rightTreeIdc(rightCount) = indices(sampleCount);
        end
    end
end