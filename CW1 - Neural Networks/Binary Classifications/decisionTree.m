load('facialPoints.mat');
load('labels.mat');

inputs = reshape(points,[66*2,150]);
targets = labels';


Mdl = fitctree(inputs,targets);