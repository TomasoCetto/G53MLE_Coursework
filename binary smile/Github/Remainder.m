function [remain] = Remainder(pParent, nParent, pChildren, nChildren)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    r = zeros( 1, size(pChildren, 2) );
    for i=0:size(pChildren)
        w = (pChildren(i)+nChildren(i)) / (pParent+nParent);
        e = BiEntropy( pChildren(i), p(nChildren(i)) );
        r(i) = w*e;
    end
    remain = sum(r);
end

