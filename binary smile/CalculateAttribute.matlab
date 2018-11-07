function val = CalculateAttribute(features, targets)

	[examples, attributes] = size(features)

	for i=1:attributes
		% TODO: calculate the estimate 