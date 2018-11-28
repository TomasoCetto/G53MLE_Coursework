function [bestHyperParameter] = innerLoop(inputs,targets,k, minKScale,maxKScale,minC,maxC)

    % SMV values
    %pOrder = 1;         
    % minKScale = 0.001;  % Kernel Scale: min-max = 1e-3 to 1e3
    % maxKScale = 1000;
    % minC = 0.001;       % c value: min-max = 1e-5 to 1e5
    % maxC = 1000;
    
    numOfExamples = (k-1)*size(inputs,1)/k; % numOfExamples = 90
    %numOfFeatures = size(inputs,2);         % inputWidth = 132
    foldLength = size(inputs,1)/k;          % foldLength = 45
    P = randperm(size(inputs,1));           % random permutation containing all index of single data point

    iteration = 1;
    hyperparameterInformation = cell(1,10);
    %cValue = minC;
%     while cValue <= maxC
    for cValue=1:maxC
        for kernelScale = 1:maxKScale
%         kernelScale = minKScale;    % initial kernelScale
%         while kernelScale <= maxKScale
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
                predictions = predict(SVM,testingInputs);
                confusion = confusion_matrix(predictions, testingTargets);
                recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2));
                precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));
                f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)));
                tPrint("f1 = " + f1s(i));

            end     % end of fold cross validation 

            % determine mean f1 value
            f1 = mean(f1s);

            % store hyperparameters into a structure
            % struct.pOrder = 2;
            struct.cValue = cValue;
            struct.kernelScale = kernelScale;
            struct.supportVectors = size(SVM.SupportVectors,1);
            struct.supportVectorsRatio = struct.supportVectors / numOfExamples;
            struct.f1 = f1;
            hyperparameterInformation{iteration} = struct;

           % config variables before starting a new loop
%             stepKScale = changeScale(kernelScale);
%             kernelScale = kernelScale + stepKScale;
%             tPrint("C Value: " + cValue + " | c value step: " + stepKScale);
%             tPrint("Kernel Scale: " + kernelScale + " | kernelScale step: " + stepKScale);
%             fprintf("|");
             iteration = iteration + 1;

        end
    end
    
    hyperparameterInformation

    % select the best f1
    bestf1 = 0;
    for i=1:length(hyperparameterInformation)
        currentf1 = hyperparameterInformation{1}.f1;
        if currentf1 > bestf1
            bestf1 = hyperparameterInformation{i}.f1;
            bestHyperParameter = hyperparameterInformation{1,i};
        end
    end
    
%     disp("The best hyperparameter regarding to f1: ");
%     disp(bestHyperParameter);

    
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

