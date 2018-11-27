load('facialPoints.mat');
load('labels.mat');


inputs = reshape(points,[66*2,150])';
targets = labels';

k = 3;
numOfExamples = (k-1)*size(inputs,1)/k;   % numOfExamples = 135
numOfFeatures = size(inputs,2);            % inputWidth = 132
% trainLength = (k-1)*length(inputs)/k;   % trainLength = 135
% inputWidth = size(inputs,1);            % inputWidth = 132
foldLength = size(inputs,1)/k;          % foldLength = 15

accuracies = zeros(1,k);

hyperparameterInformation = cell(1,40);

iteration = 1;

P = randperm(size(inputs,1));           % random permutation containing all index of single data point
pOrder = 2;

precision = 0;
recall = 0;
f1 = 0;
cValue = 1;

% for pOrder=1:10
% for cValue = 0.1:0.02:0.2
for kernelScale = 0.1:0.1

  f1s = zeros(1,k);
  recalls = zeros(1,k);
  precisions = zeros(1,k);
  for i=1:k                               % each iteration performs one time of training and CV

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

      % initialize the features and labels sets of training data
      trainingInputs = zeros(numOfExamples,numOfFeatures);
      trainingTargets = zeros(1, numOfExamples);

      % find the values of features and labels with their corresponding indexes
      for j=1:numOfExamples
          trainingTargets(j) = targets(trainPerm(j));
          trainingInputs(j,:) = inputs(trainPerm(j),:);
      end

      % Set up Division of Data for Training, Validation, Testing    

      %SVM = fitcsvm(trainingInputs,trainingTargets,'Standardize',true,'KernelFunction','rbf','KernelScale',kernelScale,'BoxConstraint', cValue);
      SVM = fitcsvm(trainingInputs,trainingTargets,'Standardize',true,'KernelFunction','polynomial', 'PolynomialOrder', pOrder, 'BoxConstraint', cValue);

      % find the values of features and labels with their corresponding indexes
      testingInputs = zeros(foldLength,numOfFeatures);
      testingTargets = zeros(1, foldLength);
      for j=1:foldLength
          testingTargets(j) = targets(validPerm(j));
          testingInputs(j,:) = inputs(validPerm(j),:);
      end

      predictions = predict(SVM,testingInputs)
      confusion = confusion_matrix(predictions, testingTargets)
      recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2))
      precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1))
      f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)))
      % getting the output with testing inputs
      sv = size(SVM.SupportVectors,1);
      % r = round(predictions)

      % calculating performance, record average accuracy
      differences = abs(gsubtract(predictions,testingTargets'));

      accuracy = 1 - sum(differences)/foldLength;
      accuracies(i) = accuracy;
      
  %     view(net)
  end

  f1 = mean(f1s);
  precision = mean(precisions);
  recall = mean(recalls);

  disp(accuracies);
  average = mean(accuracies);
  struct.cValue = cValue;
  % struct.pOrder = 2;
  struct.kernelScale = kernelScale;
  struct.supportVectors = size(SVM.SupportVectors,1);
  struct.averageValue = average;
  struct.predicted = predictions;
  struct.f1 = f1;
  struct.precision = precision;
  struct.recall = recall;
  % struct.confusionMatrix{1,}
  hyperparameterInformation{iteration} = struct;

  iteration = iteration + 1;
  % averageMatrix = [averageMatrix, average];
end


% averageMatrix';
% hyperparameterInformation

% hyperparameterInformation;
bestf1 = 0;
% bestHyperParameter;
iteration
% for i=1:length(hyperparameterInformation)
%     currentf1 = hyperparameterInformation{i}.f1;
%     if currentf1 > bestf1
%         bestf1 = hyperparameterInformation{i}.f1;
%         bestHyperParameter = hyperparameterInformation{i}
%     end
% end

% bestHyperParameter;

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