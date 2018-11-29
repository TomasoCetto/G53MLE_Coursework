% In this file, we manually split the data to be used in validation to k fold
 % this file shows how we did it in the binary classification task and it is similiar in other tasks as well
 % Even if this works, we prefer using the function cvpartition() since its not difficult to use
 % we submit it to demonstrate our understanding of cvpartation()
 % the performance and model demostrated was not 

clear all
load('facialPoints.mat');
load('labels.mat');
load('Permutation.mat')


inputs = reshape(points,[66*2,150]);
targets = labels';
 
% Create a Fitting Network
hiddenLayerSize = [4];

% Divide the data
k = 10;
trainLength = (k-1)*length(inputs)/k;   % trainLength = 135
inputWidth = size(inputs,1);            % inputWidth = 132
foldLength = length(inputs)/k;          % foldLength = 15
P = Per;          % random permutation containing all index of single data point
% errors = zeros(1,k);
nets = cell(1,k);
accuracies = zeros(1,k);

f1s = zeros(1,k);
recalls = zeros(1,k);
precisions = zeros(1,k);

for i=1:k                               % each iteration performs one time of training and CV
    
    validPerm = P((i-1)*foldLength+1:i*foldLength); % extract the indexes of validation data

    % the remaining are indexes of training data
    if i==1
		trainPerm = P(foldLength+1:end);
	elseif i==k
		trainPerm = P(1:trainLength);
	else
		trainPerm1 = P(1:(i-1)*foldLength);
    	trainPerm2 = P(i*foldLength+1:end);
    	trainPerm = [trainPerm1,trainPerm2];
	end
    
    % initialize the features and labels sets of training data
    trainInputs = zeros(inputWidth,trainLength);
    trainTargets = zeros(1, trainLength);
    
    % find the values of features and labels with their corresponding indexes
    for j=1:trainLength
        trainTargets(j) = targets(trainPerm(j));
        trainInputs(:,j) = inputs(:,trainPerm(j));
    end

    % Set up Division of Data for Training, Validation, Testing    
    net = patternnet(hiddenLayerSize,'trainlm','mse');
    net.layers{1}.transferFcn = 'logsig';
    net.layers{2}.transferFcn = 'logsig';
    % net = newff(trainInputs,trainTargets,hiddenLayerSize,{'logsig', 'tansig'},'trainlm');
    % net = newff(trainInputs,trainTargets,hiddenLayerSize,{'logsig', 'tansig'},'trainlm');
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 10/100;

    [net, tr] = train(net, trainInputs, trainTargets);
    
    % find the values of features and labels with their corresponding indexes
    validInputs = zeros(inputWidth,foldLength);
    validTargets = zeros(1, foldLength);
    for j=1:foldLength
        validTargets(j) = targets(validPerm(j));
        validInputs(:,j) = inputs(:,validPerm(j));
    end
    
    outputs = net(validInputs)
    r = round(outputs);
    confusion = confusion_matrix(r, validTargets)
    
    recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2))
    precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1))
    f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)))
    % tPrint("f1 = " + f1s(i));
    
    differences = abs(gsubtract(r,validTargets));
    accuracy = 1 - sum(differences)/foldLength;
    accuracies(i) = accuracy;
    nets{i} = net;
%     view(net)
  end

  disp(accuracies);
  average = mean(accuracies)

  f1s




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
