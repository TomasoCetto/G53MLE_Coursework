load('RBFRegrPredictions.mat')
load('NNRegPrediction.mat')
load('predictionMatrix_polyReg.mat');
load('regressionLinearPredictions.mat')


Regrbf = SVMRBFPredictionsHolder;
RegPolynomial = pMat;
RegLinear = predictionCat;
RegNN = RegPrediction;

SVMRBFPredictionsHolder = transpose(SVMRBFPredictionsHolder);
RegPolynomial = transpose(RegPolynomial);
RegLinear = transpose(RegLinear);

[ttestRegHResults(1), ttestRegPResults(1)] = ttest2(Regrbf,RegNN);
[ttestRegHResults(2), ttestRegPResults(2)] = ttest2(RegNN, RegLinear);
[ttestRegHResults(3), ttestRegPResults(3)] = ttest2(RegPolynomial, RegNN);