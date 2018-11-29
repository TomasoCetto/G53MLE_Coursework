load('facialPoints.mat');
load('labels.mat');

inputs = reshape(points,[66*2, 150])';
targets = labels';

% tree = decisionTreeLearning(inputs, targets);
% DrawDecisionTree(tree, "myTree")

% newInputs = normalize(inputs);
% newTree = decisionTreeLearning(newInputs,targets);
% DrawDecisionTree(newTree, "newTree")

% tree = fitctree(inputs,targets);
% view(tree,'mode', 'graph');

crossvalidations(inputs, targets);

function crossvalidations(inputs,targets)

    k = 10;
    c = cvpartition(length(inputs),'KFold', k);
    foldLength = length(targets)/k;

    trees = cell(1,10);
    recalls = zeros(1,k);
    precisions = zeros(1,k);
    f1s = zeros(1,k);
    % differences = zeros(k);
    accuracy = zeros(1, k);

    for i=1:c.NumTestSets
        trIDX = training(c,i);
        teIDX = test(c,i);

        trainingInputs = inputs(trIDX,:);
        trainingTargets = targets(trIDX);

        testingInputs = inputs(teIDX,:);
        testingTargets = targets(teIDX);


        %Create 10 trained trees
        trees{i} = decisionTreeLearning(trainingInputs,trainingTargets);
        DrawDecisionTree(trees{i}, "myTree")
        %tree = trees{i}
        %use the trained tree to classify my data 
        outputArray = zeros(1,foldLength);
        % disp(length(testingInputs));
        for j=1:foldLength
            value = evaluateOneSample(trees{i},testingInputs(j,:));
            outputArray(j) = value;
        end

        confusion = confusion_matrix(outputArray, testingTargets);

        differences = abs(outputArray - testingTargets);
        accuracy(i) = 1 - sum(differences)/foldLength;

        recalls(i) = confusion(1,1)/(confusion(1,1)+confusion(1,2));
        precisions(i) = confusion(1,1)/(confusion(1,1)+confusion(2,1));
        f1s(i) = 2*((precisions(i)*recalls(i))/(precisions(i)+recalls(i)));
        %recall and precision rates 
    end
    recalls
    precisions
    f1s
    accuracy
end



function output = evaluateOneSample(tree, input)
    if isempty(tree.kids)
        output = tree.class;
        return
    elseif input(tree.attribute) > tree.threshold
            output = evaluateOneSample(tree.kids{1},input);
    else
            output = evaluateOneSample(tree.kids{2},input);
    end
end
    

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


function tree = decisionTreeLearning(inputs, targets)

    % intialising tree and subtrees
%     tree = struct('op', "", 'kids', cell(1,2), 'class', [], 'attribute', 0,'threshold', 0);

    if sum(targets)==size(targets,2) || sum(targets)==0
    	tree.op = '';
    	tree.threshold = '';
        tree.kids = [];
        tree.class = majority_value(targets);
        return %return a leaf node. how?
    else
        [best_feature,best_threshold] = choose_attribute(inputs, targets);
        % fprintf("best_feature = %d\nbest_threshold = %f\n", best_feature, best_threshold);
      
        tree.op = best_feature;
        tree.kids = cell(1,2);
        tree.class = '';
        tree.attribute = best_feature;
        tree.threshold = best_threshold;

        [leftTreeIndex, rightTreeIndex] = split(inputs, best_threshold, best_feature);
        % test1 = leftTreeIndex;
        leftTreeInputs = inputs(leftTreeIndex,:);
        leftTreeTargets = targets(leftTreeIndex);
		rightTreeInputs = inputs(rightTreeIndex,:);
        rightTreeTargets = targets(rightTreeIndex);

        
        if(length(leftTreeInputs)==0)
            tree.class = majority_value(targets);
            tree.kids = [];
            return
        else
            tree.kids{1,1} = decisionTreeLearning(leftTreeInputs,leftTreeTargets);
        end
        
        if(length(rightTreeInputs)==0)
            tree.class = majority_value(targets);
            tree.kids = [];
            return
        else
            tree.kids{1,2} = decisionTreeLearning(rightTreeInputs,rightTreeTargets);
        end
        % return
    end

end

    
function [best_feature, best_threshold] = choose_attribute(features, targets)

	% TODO: compare sides to make sure the inputs match
	
	[sampleSize, attributeSize] = size(features);


	% if ~(sampleSize == targets.length):
	% 	disp('Size of inputs and targets does not match');
	% 	return


	% TODO: Calculate the number of positive and neegativ sampleSize
	
	[p, n] = Calculate_Ratio(targets);
	threshold = 0;
	bestAttribute = 0;
	bestThreshold = 0;
	bestGain = 0;
	entropy = Calculate_Entropy(p,n);

	% attributes = 10;
 	for i=1:attributeSize
		% TODO: calculate the estimate on informatton contaied
		for j=1:sampleSize
			threshold = features(j,i);
			leftChild = [];
			rightChild = [];
			% leftChildPos = 0;
			% leftChildNeg = 0;
			% rightChildPos = 0;
			% rightChildNeg = 0;

			for x=1:sampleSize
				if features(x,i) > threshold
					leftChild = [leftChild, x];
					% leftChildNeg++;
					% leftChildPos++;
				else
					rightChild = [rightChild, x];
					% rightChildNeg++;
					% rightChildPos++;
				end
			end

			[lp, ln] = Calculate_Ratio(getTargets(leftChild,targets));
			[rp, rn] = Calculate_Ratio(getTargets(rightChild,targets));
			% remainder = (lp+ln)/(p+n)*Calculate_Entropy(lp, ln) + (rp+rn)/(p+n)*Calculate_Entropy(rp, rn)
			remainder = Calculate_Remainder(lp,ln,rp,rn);
			gain = entropy - remainder;
			if gain > bestGain
				bestGain = gain;
				bestAttribute = i;
				bestThreshold = threshold;
			end
		end
	end

	best_threshold = bestThreshold;
% 	bestGain
    % fprintf("bestGain = %f\n", bestGain);
	best_feature = bestAttribute;
	% return (best_feature, best_threshold)
end    

function [leftTreeIndex rightTreeIndex] = split(inputs, threshold, best_feature)
leftTreeIndex = [];
rightTreeIndex = [];
  for i = 1:size(inputs,1)
    if(inputs(i, best_feature) > threshold)
        leftTreeIndex = [leftTreeIndex, i];
    else
        rightTreeIndex = [rightTreeIndex, i];
    end
        
  end
end



  
%end

function t = getTargets(indexes, targets)
	t = [];
	for i=1:length(indexes)
		t = [t, targets(indexes(i))];
	end
	% return t;
end	



function [positive, negative] = Calculate_Ratio(targets)
	n = 0;
	p = 0;
	for i=1:length(targets)
		if targets(i) == 0
			n=n+1;
		else
			p=p+1;
		end
	end
	% [positive, negative] = p, n;
	positive = p;
	negative = n;
	% return [positive, negative];
end


function entropy = Calculate_Entropy(p, n)
	posProb = p / (p+n);
	negProb = n / (p+n);
	if posProb == 0
		entropyOne = 0;
	else
		entropyOne = - posProb*log2(posProb);
        % entropy = 0;
        % return;
	end

	if negProb == 0
		entropyTwo = 0;
	else
		entropyTwo = - negProb*log2(negProb);
        % entropy = 0;
        % return;
	end

	% entropy = - posProb*log2(posProb) - negProb*log2(negProb)
	entropy = entropyOne + entropyTwo;
end


function remainder = Calculate_Remainder(lp, ln, rp, rn)
	total = lp+ln+rp+rn;
	remainder = (lp+ln)/total*Calculate_Entropy(lp, ln) + (rp+rn)/total*Calculate_Entropy(rp, rn);
end


function label = majority_value(targets)
    n=0;
    p=0;
    for i=1:length(targets)
        if targets(i) == 1
            p = p + 1;
        else
            n = n +1;
        end
    end
    if n<p
        label=1;
    else
        label=0;
    end
end

% function information_Gain