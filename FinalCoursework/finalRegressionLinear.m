clear;  % ***Clear all previous data before executing this code***
disp("*Clear all previous data");

% prepare data
load('facialPoints.mat');
load('headpose.mat');
load('PermutationRegression.mat');      % P = permutation
inputs = reshape(points,[66*2,8955])';   % samples x features
targets = pose(:,6);                     % samples x class

% cross validation initial
k = 10;
numOfExamples = (k-1)*size(inputs,1)/k; 
numOfFeatures = size(inputs,2);        
foldLength = round(size(inputs,1)/k);          

% variables
global doDisp;  
doDisp = false;  % display flag
global duration;
duration = 0;   % timer

%tic;    % start timer

%Outerloop returns
rms_errors = zeros(1,10);
meanRMS = 0;
predictionsMatrix = cell(1,10);
scoreMatrix = cell(1,10);
supportVectors = zeros(1,10);
meanSV = 0;
SVMLinearPredictionsMat = [];
SVMs = {};
bestHypers = {};

% outerloop: 10 cross validation
for i = 1:10
    
    % retieving training and test subdataset
    [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs , targets);  

    % Do inner cross validation and return the best hyperparamters
    currentModelParameters = innerLoop(trainingInputs,trainingTargets,3);
    bestHypers{i} = currentModelParameters;
    
    % train the model with the best hyperparameter
    tic;
    display("start " + i + "-fold training SVM of OuterLoop");    
    display("best: c value = " + currentModelParameters.cValue);
    display("best: epsilon values = " + currentModelParameters.epsilonValue);
    currentModel = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'linear', 'BoxConstraint', currentModelParameters.cValue, 'Epsilon', currentModelParameters.epsilonValue);
    SVMs{i} = currentModel;
    display("finish " + i + "-fold training SVM of OuterLoop: " + toc + " sec");
    duration = duration + toc;

    % test the model, do prediction, and calculate rms of this fold
    predictions = predict(currentModel,testingInputs);
    rms = rmsCal(predictions, testingTargets);
    rms_errors(i) = rms;
    tPrint("rms = " + rms_errors(i));
    predictionsMatrix{i} = predictions;
    supportVectors(i) = size(currentModel.SupportVectors,1);
    tPrint("sv = " + supportVectors(i));

    SVMLinearPredictionsMat = [SVMLinearPredictionsMat predictions];

end

SVMLinearPredictions = SVMLinearPredictionsMat(:);

fprintf("\n");
disp("time stamp: " + duration + " sec");    % stamp total training duration

meanRMS = mean(rms_errors);
meanSV = mean(supportVectors);
disp("mean rms for 10 cross validation = " + meanRMS);
disp("mean support vectors for 10 cross validation = " + meanSV);


%                           End of process





%   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -

%   -   -   -   -   -   - ???  FUNCTIONS ???    -   -   -   -   -   -   -

%   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -



%inner loop function

function [bestHyperParameter] = innerLoop(inputs,targets,k)

    % initial variables for 3 cross validation
    numOfExamples = (k-1)*size(inputs,1)/k; % numOfExamples = 135
    foldLength = size(inputs,1)/k;          % foldLength = 15
    P = randperm(size(inputs,1));           % random permutation containing all index of single data point
    
    % variables
    iteration = 1;
    global duration;
    
    % record
    hyperparameterInformation = cell(1,1);

    %Box constraint values
    cValues = [0.1 20 100];
    % epsilon 
    epsilonValues = [1 1.5 2];

    for cIndex=1:length(cValues)
        
        for eIndex=1:length(epsilonValues)
            
            % record errors fron 3 folds
            rms_errors = zeros(1,k); 
            
            % Do 3 inner fold validation
            for i=1:k     % each iteration performs one time of training and CV
                
                % retrieve training and testing dataset for i fold
                [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs, targets);  

                % training SVM
                tic;
                display("start " + i + "-fold training SVM of InnerLoop");   
                disp("C value = " + cValues(cIndex));
                disp("epsilon = " + epsilonValues(eIndex));
                SVM = fitrsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'linear', 'BoxConstraint', cValues(cIndex), 'Epsilon', epsilonValues(eIndex));
                display("finish " + i + "-fold training SVM of InnerLoop: " + toc + " sec");
                duration = duration + toc;

                sv = size(SVM.SupportVectors,1);
                tPrint("sv = " + sv + "(" + size(SVM.SupportVectors,1)/size(SVM.SupportVectors,2)*100 + "%)");



                % prediction and evaluation. 
                predictions = predict(SVM,testingInputs);
                rms = rmsCal(predictions, testingTargets);
                rms_errors(i) = rms;

                tPrint("rms_errors = " + rms_errors(i));



            end     % end of fold cross validation 


        % determine mean values
        meanRMS = mean(rms_errors);

        % store hyperparameters into a structure
        struct.cValue = cValues(cIndex);
        struct.epsilonValue = epsilonValues(eIndex);
        struct.supportVectors = size(SVM.SupportVectors,1);
        struct.supportVectorsRatio = struct.supportVectors / size(trainingInputs,1);
        struct.rms = meanRMS;
        hyperparameterInformation{iteration} = struct;

        iteration = iteration + 1;

        end % end of epsilon loop

    end     % end of box constraint loop

    % select the best rms
    bestRMS = 100;
    for i=1:length(hyperparameterInformation)
        currentRMS = hyperparameterInformation{i}.rms;
        if currentRMS < bestRMS
            bestRMS = currentRMS;
            bestHyperParameter = hyperparameterInformation{i};
        end
    end


        disp("The best hyperparameter regarding to rms: ");
        disp(bestHyperParameter);


end



%Start of confusion matrix



function cm = confusion_matrix(outputs, labels)

% cm = zeros(2);

tp=0;tn=0;fp=0;fn=0;

% length(outputs);

for i=1:length(outputs)

    if (labels(i) == 1) && (outputs(i)==1)

        tp=tp+1;

    elseif (labels(i) == 0) && (outputs(i)==0)

        tn=tn+1;

    elseif (labels(i) == 1) && (outputs(i)==0)

        fn=fn+1;

    else

        fp=fp+1;

    end

end

cm = [tp, fn; fp, tn];

end



% Start of CV Partition



function [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs, targets)

    if i == k
        validPerm = P(foldLength*(k-1)+1:end);
    else
        validPerm = P((i-1)*foldLength+1:i*foldLength); % extract the indexes of validation data
    end
    % the remaining are indexes of training data

    if i==1
        trainPerm = P(foldLength+1:end);
    elseif i==k
        trainPerm = P(1:foldLength*(k-1));
    else
        trainPerm1 = P(1:(i-1)*foldLength);

        trainPerm2 = P(i*foldLength+1:end);

        trainPerm = [trainPerm1,trainPerm2];

    end

    % Set up Division of Data for Training, Validation, Testing

    % find the values of features and labels with their corresponding indexes

    trainingTargets = targets( trainPerm, :);

    trainingInputs = inputs( trainPerm, :);

    % find the values of features and labels with their corresponding indexes

    testingTargets = targets( validPerm, :);

    testingInputs = inputs( validPerm, :);

end



% Start of printing

function [] = tPrint(str)   % tracing print

    global doDisp;

    if doDisp

        disp(str);

    end

end     % end of tPrint

function [rms] = rmsCal(outputs, testingTargets)
%calculate the rms error between predictions and targets and store it
    total = 0;
    differences = outputs - testingTargets;
    for j=1:length(differences)
        square = power(differences(j), 2);
        total = total + square;
    end
    ms = (1/length(testingTargets))*total;
    rms = sqrt(ms);
end     % end of rms
