load('facialPoints.mat');
load('labels.mat');


inputs = reshape(points,[66*2,150])';
inputs = normalize(inputs);
targets = labels';

k = 3;
numOfExamples = (k-1)*size(inputs,1)/k;   % numOfExamples = 135
numOfFeatures = size(inputs,2);            % inputWidth = 132
% trainLength = (k-1)*length(inputs)/k;   % trainLength = 135
% inputWidth = size(inputs,1);            % inputWidth = 132
foldLength = size(inputs,1)/k;          % foldLength = 15
P = randperm(size(inputs,1));           % random permutation containing all index of single data point

accuracies = zeros(1,k);

bestC = 0.1;
bestPOrder = 0;
hyperparameterInformation = cell(1,40);
averageMatrix = [];
iteration = 1;
prediction = [];

for pOrder=3:3

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

      % SVM = fitcsvm(trainingInputs,trainingTargets,'KernelFunction','linear','BoxConstraint', 1);
      SVM = fitcsvm(trainingInputs,trainingTargets,'KernelFunction','polynomial', 'PolynomialOrder', pOrder, 'BoxConstraint', 1, 'OptimizeHyperparameter', 'BoxConstraint');

      % find the values of features and labels with their corresponding indexes
      testingInputs = zeros(foldLength,numOfFeatures);
      testingTargets = zeros(1, foldLength);
      for j=1:foldLength
          testingTargets(j) = targets(validPerm(j));
          testingInputs(j,:) = inputs(validPerm(j),:);
      end

      predictions = predict(SVM,testingInputs)
      % getting the output with testing inputs
      sv = size(SVM.SupportVectors,1);
      % r = round(predictions)

      % calculating performance, record average accuracy
      differences = abs(gsubtract(predictions,testingTargets'));

      accuracy = 1 - sum(differences)/foldLength;
      accuracies(i) = accuracy;
      
  %     view(net)
  end

  disp(accuracies);
  average = mean(accuracies);
  struct.cValue = 1;
  struct.pOrder = pOrder;
  struct.supportVectors = size(SVM.SupportVectors,1);
  struct.averageValue = average;
  struct.predicted = prediction;
  hyperparameterInformation{1,iteration} = struct;
  iteration = iteration + 1;
  averageMatrix = [averageMatrix, average];
end


averageMatrix';
hyperparameterInformation