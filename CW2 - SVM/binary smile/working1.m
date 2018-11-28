
clear;  % ***Clear all previous data before executing this code***

disp("*Clear all previous data");



% prepare data

load('facialPoints.mat');

load('labels.mat');

inputs = reshape(points,[66*2,150])';   % samples x features

targets = labels;                       % samples x classes



% 10 cross validation initial

k = 10;

numOfExamples = (k-1)*size(inputs,1)/k; % numOfExamples = 135

numOfFeatures = size(inputs,2);         % inputWidth = 132

foldLength = size(inputs,1)/k;          % foldLength = 15

P = randperm(size(inputs,1));           % random permutation containing all index of single data point



% variables

global doDisp;  

doDisp = false;  % display flag




tic;    % start timer



%Outerloop returns

recalls = zeros(1,10);

precisions = zeros(1,10);

f1s = zeros(1,10);

predictionsMatrix = cell(1,10);

scoreMatrix = cell(1,10);

supportVectors = zeros(1,10);



%outerloop

for i = 1:10

    



    [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs , targets);  

    currentModelParameters = innerLoop(trainingInputs,trainingTargets,3);

    

    %test the model

    currentModel = fitcsvm(trainingInputs,trainingTargets, 'Standardize',true,'KernelFunction', 'rbf', 'KernelScale',currentModelParameters.kernelScale, 'BoxConstraint',currentModelParameters.cValue); 

    

    %calculate f1 of this fold

    [predictions, score] = predict(currentModel,testingInputs);

    confusion = confusion_matrix(predictions, testingTargets);

    recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2));

    precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));

    f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)));

    tPrint("f1 = " + f1s(i));

    predictionsMatrix{i} = predictions;

    scoreMatrix{i} = score;

    supportVectors(i) = size(currentModel.SupportVectors,1);

    score

    predictions

    

end



fprintf("\n");

disp("time stamp: " + toc + " sec");    % stamp total training duration





%                           End of process





%   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -

%   -   -   -   -   -   - ???  FUNCTIONS ???    -   -   -   -   -   -   -

%   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -



%inner loop function

function [bestHyperParameter] = innerLoop(inputs,targets,k)

    

    numOfExamples = (k-1)*size(inputs,1)/k; % numOfExamples = 135

    foldLength = size(inputs,1)/k;          % foldLength = 15

    P = randperm(size(inputs,1));           % random permutation containing all index of single data point



    iteration = 1;

%     hyperparameterInformation = cell(1,10);
     hyperparameterInformation = [];

    

    %Box constraint values

    cValueStart = 1;

    cValueEnd = 10;

    cValueStep = cValueStart;

    

    %sigma values

    kernelScaleStart = 1;

    kernelScaleEnd = 10;

    kernelScaleStep = kernelScaleStart;

    

    %polynomial order

    pOrderStart = 2;

    pOrderEnd = 5;
 
    % p order scale = 1 as default
    

    cValueCount = 1;

    kernelScaleCount = 1;

        

    %for kernelScale=kernelScaleStart:kernelScaleStep:kernelScaleEnd

    kernelScale = kernelScaleStart;

    while kernelScale <= kernelScaleEnd

    %for cValue=cValueStart:cValueStep:cValueEnd

        cValue = cValueStart;

        while cValue <= cValueEnd

            f1s = zeros(1,k);

            recalls = zeros(1,k);

            precisions = zeros(1,k);

            for i=1:k     % each iteration performs one time of training and CV



                % retrieve training and testing dataset for i fold

                [trainingInputs, trainingTargets, testingInputs, testingTargets] = myCVPartition(foldLength, numOfExamples, i, P, k, inputs, targets);  



                % training SVM

                SVM = fitcsvm( trainingInputs, trainingTargets, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', kernelScale, 'BoxConstraint', cValue);

                tPrint("finish " + i + "-fold training SVM: " + tic + " sec");

                sv = size(SVM.SupportVectors,1);

                tPrint("sv = " + sv + "(" + size(SVM.SupportVectors,1)/size(SVM.SupportVectors,2)*100 + "%)");



                % prediction and evaluation. 

                predictions = predict(SVM,testingInputs)

                confusion = confusion_matrix(predictions, testingTargets)

                recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2));

                precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));

%                 if precisions(i) + recalls(i) == 0

%                     f1s(i) = 0

%                 else

                    f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)))

%                 end

                tPrint("f1 = " + f1s(i));



            end     % end of fold cross validation 



            % determine mean f1 value

            f1 = mean(f1s);

           



            % store hyperparameters into a structure

            % struct.pOrder = 2;

            struct.cValue = cValue;

            struct.kernelScale = kernelScale;

            struct.supportVectors = size(SVM.SupportVectors,1);

            struct.supportVectorsRatio = struct.supportVectors / size(inputs,1);

            struct.f1 = f1;
            struct.nobest = false;

            hyperparameterInformation{iteration} = struct;



           % config variables before starting a new loop

%             stepKScale = changeScale(kernelScale);

%             kernelScale = kernelScale + stepKScale;

%             tPrint("C Value: " + cValue + " | c value step: " + stepKScale);

%             tPrint("Kernel Scale: " + kernelScale + " | kernelScale step: " + stepKScale);

%             fprintf("|");

            iteration = iteration + 1;

            cValueCount = cValueCount +1

            

            if cValueCount == 10

                cValueStep = cValueStep*10;

                cValueCount = 0;

            end

            cValue = cValue + cValueStep;

            display(cValue)

            display(cValueStep)

            

        end

        

        kernelScaleCount = kernelScaleCount +1;

        if kernelScaleCount == 10

                kernelScaleStep = kernelScaleStep*10;

                kernelScaleCount = 0;

        end

        kernelScale = kernelScale + kernelScaleStep

    end



    % select the best f1

    bestf1 = 0;
    for i=1:length(hyperparameterInformation)

        currentf1 = hyperparameterInformation{1}.f1;

        if currentf1 >= bestf1

            bestf1 = hyperparameterInformation{i}.f1;

            bestHyperParameter = hyperparameterInformation{1,i};
        elseif isnan(currentf1)
            bestHyperParameter = hyperparameterInformation{1,i};
            bestHyperParameter.nobest = true;
        end

    end
    

%     disp("The best hyperparameter regarding to f1: ");

%     disp(bestHyperParameter);



    

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

    validPerm = P((i-1)*foldLength+1:i*foldLength); % extract the indexes of validation data

    % the remaining are indexes of training data

    if i==1

        trainPerm = P(foldLength+1:end);

    elseif i==k

        trainPerm = P(1:numOfExamples);

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



function [step] = changeScale(scale)

        if scale < 0.0099999999

            step = 0.001; 

        elseif scale < 0.09999999999

            step = 0.01;

        elseif scale < 0.99999999999

            step = 0.1;

        elseif scale < 9.99999999

            step = 1;

        elseif scale < 99.99999999

            step = 10;

        elseif scale < 999.999999999

            step = 100;

        else

            step = 1000;

        end

end     % end of changeScale