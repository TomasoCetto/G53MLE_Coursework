% Load & Prepare Data
global labels;
global features;
load("facialPoints.mat");
load("labels.mat");
points = reshape(points, [150,66*2]);
features = points;

% Initial vaiables
kf = 10;

% storage
treeArr = createStructArr(kf);
predictArr = zeros(size(labels,1), kf);
truePos = [];
trueNeg = [];
falsePos = [];
falseNeg = [];

% Cross valication
% trainIdc = [1:3 51:51];
% testIdc = [4 5 52 53];
% trainIdc = [1:150];
% cv = cvpartition(size(features,1), 'kfold', kf);
% prepare training and test indices
CrossValIndices = crossvalind('Kfold', size(features,1), kf);   %group dataset into 10 subdataset randomly and indicate them by indices

for fold = 1 : kf
    
    % prepare training and test indices
    trainIdc = [];          %list of indices pointing to training data in the dataset
    testIdc = [];           %list of indices pointing to test/validation data in the dataset
    for j=1:size(features,1)     %make lists of index
        if CrossValIndices(j)==fold
            testIdc = horzcat(testIdc, j);      %put a training dataset index in the end of the training list
        else
            trainIdc = horzcat(trainIdc, j);    %put a test dataset index in the end of the test list
        end
    end
    
    % Training
    fprintf("start learing: %d\n", fold);
    treeArr(fold) = DecisionTreeLearning(trainIdc);
%     treeArr(fold) % show tree
    DrawDecisionTree(treeArr(fold), "fold" + num2str(fold));
    
    fprintf("finish learing\n")
    
    % buildin function
    dt1 = fitctree(features(trainIdc,:), labels(trainIdc));
    view(dt1, 'mode', 'graph');

    % Testing
    fprintf("start testing: %d\n", fold);
    predictArr(:, fold) = DecisionTreeSim(treeArr(fold), testIdc);
    % show comparison of labels and predictions
%     [labels(testIdc) predictArr(testIdc,fold)]  
%     fprintf("labels vs predictions %s\n", 11173);
    % recall and Precision Rate
%     [truePos, trueNeg, falsePos, falseNeg, recogRate] = recogTabel( );

    truePos(fold) = sum(predictArr( testIdc, fold) & labels( testIdc, 1));
    trueNeg(fold) = sum(~(predictArr( testIdc, fold) | labels( testIdc, 1)));
    falsePos(fold) = sum( ( predictArr( testIdc, fold) | labels( testIdc, 1) ) & ( ~predictArr( testIdc, fold) | ~labels( testIdc, 1) ) );
    oneprdctIdc = testIdc(1,find(predictArr(testIdc,fold) == 1));
    onelabelIdc = testIdc(1,find(labels(testIdc) == 1));
    falsePos(fold) = sum( ( predictArr( oneprdctIdc, fold) | labels( oneprdctIdc, 1) ) & ( ~predictArr( oneprdctIdc, fold) | ~labels( oneprdctIdc, 1) ) );
    falseNeg(fold) = sum( ( predictArr( onelabelIdc, fold) | labels( onelabelIdc, 1) ) & ( ~predictArr( onelabelIdc, fold) | ~labels( onelabelIdc, 1) ) );  
    fprintf("true positive\ttrue negative\tfalse positive\tfalse negative\n")
%     fprintf("%d\t\t\t\t%d\t\t\t\t%d\t\t\t\t%d\n", truePos, trueNeg, falsePos, falseNeg);
    fprintf("%d\t\t\t\t%d\t\t\t\t%d\t\t\t\t%d\n", truePos(fold), trueNeg(fold), falsePos(fold), falseNeg(fold));
    
    fprintf("finish testing\n");

end % end of fold

    % load ionosphere;
    % Mdl = fitctree(X,Y);
    % view(Mdl, 'mode', 'graph');
    fprintf("end process\n");

% - -   -   -   -   end whole process -   -   -   -   -



function structArr = createStructArr(n)
    for i = 1: n
       structArr(i) = initTreeStruct(); 
    end
end

% TODO:
function [truePos, trueNeg, falsePos, falseNeg, recogRate] = recogTabel( )
    truePos = sum(predictArr( testIdc, fold) & labels( testIdc, 1));
    trueNeg = sum(~(predictArr( testIdc, fold) | labels( testIdc, 1)));
    falsePos = sum( ( predictArr( testIdc, fold) | labels( testIdc, 1) ) & ( ~predictArr( testIdc, fold) | ~labels( testIdc, 1) ) );
    oneprdctIdc = testIdc(1,find(predictArr(testIdc,fold) == 1));
    onelabelIdc = testIdc(1,find(labels(testIdc) == 1));
    falsePos = sum( ( predictArr( oneprdctIdc, fold) | labels( oneprdctIdc, 1) ) & ( ~predictArr( oneprdctIdc, fold) | ~labels( oneprdctIdc, 1) ) );
    falseNeg = sum( ( predictArr( onelabelIdc, fold) | labels( onelabelIdc, 1) ) & ( ~predictArr( onelabelIdc, fold) | ~labels( onelabelIdc, 1) ) );  
    fprintf("true positive\ttrue negative\tfalse positive\tfalse negative\n")
%     fprintf("%d\t\t\t\t%d\t\t\t\t%d\t\t\t\t%d\n", truePos, trueNeg, falsePos, falseNeg);
    fprintf("%d\t\t\t\t%d\t\t\t\t%d\t\t\t\t%d\n", truePos(fold), trueNeg(fold), falsePos(fold), falseNeg(fold));
end


