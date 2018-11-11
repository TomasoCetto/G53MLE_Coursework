function [gain] = Gain(pParent, nParent, pChildren, nChildren)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
%   pChildren = array of [number of p in left , number of p in right]
%   nChildrem = array of [number of n in left , number of n in right]
    r = zeros( 1, size(pChildren, 2) );
    for i=1:size(pChildren, 2)
        w = (pChildren(1,i)+nChildren(1,i)) / (pParent+nParent);
        e = BiEntropy( pChildren(1,i), nChildren(1,i) );
        r(i) = w*e;
    end
    remain = sum(r);

    gain = BiEntropy(pParent, nParent) - remain;
end

