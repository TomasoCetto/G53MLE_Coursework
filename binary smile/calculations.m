function [positive, negative] = Calculate_Ratio(targets)
	n = 0;
	p = 0;
	for i=1:targets.length
		if targets(i) == 0
			n++;
		else
			p++;
		end
	end
	return ([positive, negative] = [p, n])
	% return [positive, negative];
end


function entropy = Calculate_Entropy(p, n)
	posProb = p / (p+n);
	negProb = n / (p+n);

	return (entropy = - posProb*log2(posProb) - negProb*log2(negProb))
end