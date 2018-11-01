load('labels.mat')
load('facialPoints.mat')

points = reshape(points, [66*2,150]);
labels = labels';

k = 10;
topoNumber = 1;

%create the net
topo = cell(1,topoNumber);
topo{1} = [4];
% topo{2} = [10];
% topo{3} = [10,4];
% topo{5} = [3,4];
% topo{3} = [20,10];
% topo{4} = [10,10];
% topo{5} = [20];

% net = patternnet(10,'trainlm','mse');
% net.trainParam.epochs = 2000;

%create a cell of 10 neural nets
numNN = 10;
nets = cell(1,numNN);

accuracyMatrix = zeros(5,k);
topoaverages = zeros(1,topoNumber);
topomeans = zeros(1,topoNumber);
topostds = zeros(1,topoNumber);
for i=1:topoNumber
    net = patternnet(topo{i},'trainlm','mse');
    net.layers{1}.transferFcn = 'logsig';
    if length(topo{i})==2
        net.layers{2}.transferFcn = 'logsig';
        net.layers{3}.transferFcn = 'logsig';
    else
        net.layers{2}.transferFcn = 'logsig';
    end
    net.divideParam.trainRatio = 85/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 0/100;
    net.trainParam.epochs = 50;
    % net.learnParam.lr

    cv = crossvalidation(points,labels,net,nets,k);
    accuracyMatrix(i,:) = cv;
    topoaverages(i) = sum(cv)/k;
    topomeans(i) = mean(cv);
    topostds(i) = std(cv);
end

topoaverages
average = topomeans
topostds
