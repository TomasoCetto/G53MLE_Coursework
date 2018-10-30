%create Data
%load data
cd 'D:\Rapee\Msc Comp Sci\OneDrive - The University of Nottingham\Matlab\data\data\assessment\multiclass emotion';
load('emotions_data');      %x = samples by features; y = samples by labels(1,2,...6)
labels = zeros(6,612);      %labels = [0/1,0/1, ....x6] by samples
for i=1:612                 %can use dummyvar instead
    labels(y(i),i) = 1;
end
input = transpose(x);       %input = features by samples

%Initial Variables
K = 10;

%Record Arrays
trArr = [];                 %training records
confArr = [];               %confusion records
netArr = [];                %collection of networks from each fold

%k-fold 
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
    
    %create network
    net = newff(input, labels,[10], {'tansig' 'softmax'}, 'trainscg', 'learngd', 'crossentropy');
    %net = newff(input, labels,[10],{'tansig' 'tansig'}, 'trainlm', 'learngd', 'mse');
    net.divideFcn = 'divideind';            %set the network to use the indices to divide training and test/validation dataset
    net.trainParam.epochs = 100;
    net.divideParam.trainInd = trainInd;    %set the network indice of training dataset
    net.divideParam.valInd = testInd;       %set the network indice of test/validation dataset
    net.divideParam.testInd = [];           %no test data
    
    netArr = horzcat(netArr, net);          % store the network
    
    %training
    [net, tr] = train(net,input, labels);
    trArr = horzcat(trArr,tr);              %store the training result
    confArr = horzcat(confArr,confusion(labels(:,testInd) ,sim(net,input(:,testInd)))); %store 1-accuracy
    plotconfusion(labels(:,testInd) ,sim(net,input(:,testInd)))
end

% % Display records
display('best_vperf for each fold:');
for i=1:10
    display([num2str(trArr(i).best_vperf)]);
end
display('1-accuracy for each fold:');
for i=1:10
    display([num2str(confArr(i))]);
end
display('best_epoch for each fold:');
for i=1:10
display([num2str(trArr(i).best_epoch)]);
end
