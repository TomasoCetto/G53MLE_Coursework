function [i] = BiEntropy(p,n)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    pp = p/(p+n);
    np = n/(p+n);
    i = -(pp*log2(pp))-(np*log2(np));
end

