load('facialPoints.mat');
load('labels.mat');

inputs = reshape(points,[150, 66*2]);
% inputs = reshape(points,[66*2,150]);
targets = labels';

choose_attribute(inputs, targets);

% tree = fitctree(inputs,targets);


% DrawDecisionTree(tree, "myTree")



function [best_feature, best_threshold] = choose_attribute(features, targets)

	% TODO: compare sides to make sure the inputs match
	
	[sampleSize, attributeSize] = size(features);		%sampleSize=150, attributeSize=132


	% if (sampleSize != targets.length):
		% disp('Size of inputs and targets does not match');
		% return


	% Calculate the number of positive and negative sampleSize
	[p, n] = Calculate_Ratio(targets);

	threshold = 0;
	bestAttribute = 0;
	bestThreshold = 0;
	bestGain = 0; 					%the best gain for all possible combinations
	gainList = zeros(1,attributeSize);
	bestlp = 0;bestln=0;bestrn=0;bestrp=0;

	% attributeSize = 1;
 	for i=1:attributeSize
		% calculate the estimate entropy on informatton contaied
		entropy = Calculate_Entropy(p,n);
		% bestfeatureG = 0;
		for j=1:sampleSize
			threshold = features(j,i)
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
			remainder = Calculate_Remainder(lp,ln,rp,rn)
			gain = entropy - remainder;
			% if gain > bestfeatureG
			% 	bestfeatureG = gain;
			% end
			if gain > bestGain
				bestGain = gain;
				bestAttribute = i;
				bestThreshold = threshold;
				bestln = ln;bestrp=rp;bestrn=rn;bestlp=lp;
			end
		end

		% gainList(i) = bestfeatureG;
	end

	best_threshold = bestThreshold
	bestGain
	best_feature = bestAttribute
	% gainList
	% return (best_feature, best_threshold)
	bestlp
	bestrn
	bestln
	bestrp
end


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

	entropy = - (posProb*log2(posProb)) - (negProb*log2(negProb))
end


function remainder = Calculate_Remainder(lp, ln, rp, rn)
	total = lp+ln+rp+rn;
	remainder = (lp+ln)/total*Calculate_Entropy(lp, ln) + (rp+rn)/total*Calculate_Entropy(rp, rn)
end


% function information_Gain

