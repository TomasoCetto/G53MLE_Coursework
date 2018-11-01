%create Data
%load data
cd 'D:\Rapee\Msc Comp Sci\OneDrive - The University of Nottingham\Matlab\data\data\assessment\multiclass emotion';
load('emotions_data');      %x = samples by features; y = samples by labels(1,2,...6)
labels = zeros(6,612);      %labels = [0/1,0/1, ....x6] by samples
for i=1:612                 %can use dummyvar instead
    labels(y(i),i) = 1;
end
input = transpose(x);       %input = features by samples

%Record Arrays
trArr = [];                 %training records
confArr = [];               %confusion records
netArr = [];                %collection of networks from each fold

%k-fold 
K = 10;
CrossValIndices = crossvalind('Kfold', size(input,2), K);   %group dataset into 10 subdataset randomly and indicate them by indices
for i = 1: K
%i = 1;
    display(['Cross validation, folds ' num2str(i)]);
    trainInd = [];          %list of indices pointing to training data in the dataset
    testInd = [];           %list of indices pointing to test/validation data in the dataset
    
    for j=1:612     %make lists of index     
        if CrossValIndices(j)==i
            testInd = horzcat(testInd, j);      %put a training dataset index in the end of the training list
        else
            trainInd = horzcat(trainInd, j);    %put a test dataset index in the end of the test list
        end
    end
     
    %choice of topologies
    topo = cell(1,5);
    topo{1} = [10];
    topo{2} = [20];
    topo{3} = [20,10];
    topo{4} = [10,10];
    topo{5} = [10,20];
    
    %create network
    net = newff(input, labels, topo{1}, {'tansig', 'softmax'}, 'trainscg', 'learngd', 'crossentropy');
    net.divideFcn = 'divideind';            %set the network to use the indices to divide training and test/validation dataset
    net.trainParam.epochs = 100;
    net.divideParam.trainInd = trainInd;    %set the network indice of training dataset
    net.divideParam.valInd = testInd;       %set the network indice of test/validation dataset
    net.divideParam.testInd = [];           %no test data
    
    netArr = horzcat(netArr, net);          % store the network
    
    %training
    [net, tr] = train(net,input, labels);
    trArr = horzcat(trArr,tr);              %store the training result
    
    %testing: confusion and accuracy
    confArr = horzcat(confArr,confusion(labels(:,testInd) ,sim(net,input(:,testInd)))); %store 1-accuracy of network's output
    plotconfusion(labels(:,testInd) ,roundClass(sim(net,input(:,testInd))));                        %plot confusion matrix of network's output
    t2 = sim(net,input(:,testInd));
    confMat(labels(:,testInd) ,roundClass(sim(net,input(:,testInd))));                  %confusion matrix of network's rounded output
end

% % Display records
disp('best_vperf in each fold:');
for i=1:10
    display([num2str(trArr(i).best_vperf)]);
end
disp('best_epoch in each fold:');
for i=1:10
display([num2str(trArr(i).best_epoch)]);
end
disp('accuracy in each fold:');
for i=1:10
    display([num2str(abs(1-confArr(i)))]);
end
