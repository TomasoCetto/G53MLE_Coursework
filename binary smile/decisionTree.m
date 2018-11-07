load('facialPoints.mat');
load('labels.mat');

inputs = reshape(points,[150, 66*2]);
% inputs = reshape(points,[66*2,150]);
targets = labels';


tree = fitctree(inputs,targets);


% DrawDecisionTree(tree, "myTree")



function [best_feature, best_threshold] = choose_attribute(features, targets):

	% TODO: compare sides to make sure the inputs match
	
	[examples, attributes] = size(features)


	if (examples =! len(targets)):
		disp('Size of inputs and targets does not match');
		return


	% TODO: Calculate the number of positive and neegativ examples
	n = 0;
	p = 0;
	for i=1:examples
		if targets(i) == 0
			n++;
		else
			p++;
		end
	end


 	for i=1:attributes
		% TODO: calculate the estimate on informatton contaied
		estimate = Calculate_Estimate(p,n)
	end
end

function [positive, negative] = 

function estimate = Calculate_Estimate(p, n)
	posProb = p / (p+n);
	negProb = n / (p+n);

	estimate = - posProb*log2(posProb) - negProb*log2(negProb)
end