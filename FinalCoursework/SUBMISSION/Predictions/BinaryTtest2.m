load ('BinaryNNPredictions.mat')
load ('DecisionTreePrediction.mat')
load('SVMRBFPredictions.mat')
SVMGaussianPredictions = SVMPolynomialPredictions;
load('SVMPolynomialPredictions.mat')
load('LinearSVMClassification.mat')

NNaccuracy = binaryNNoutputs;
DTAccuracy = DecisionTreePredictionResult;
SVMrbf = SVMGaussianPredictions;
SVMLinear =linearPredictions;
SVMPolynomial = SVMPolynomialPredictions;


for i = 1:150
    binaryNNoutputs(i) = round(binaryNNoutputs(i), 0);
end

for i = 1:150
    SVMrbf(i) = abs(SVMrbf(i));
end

ttestBinaryHResults = zeros(1,7);
ttestBinaryPResults = zeros(1,7);

[ttestBinaryHResults(1), ttestBinaryPResults(1)] = ttest2(NNaccuracy, SVMrbf);
[ttestBinaryHResults(2), ttestBinaryPResults(2)] = ttest2(NNaccuracy, DTAccuracy);
[ttestBinaryHResults(3), ttestBinaryPResults(3)] = ttest2(NNaccuracy, SVMLinear);
[ttestBinaryHResults(4), ttestBinaryPResults(4)] = ttest2(NNaccuracy, SVMPolynomial);
[ttestBinaryHResults(5), ttestBinaryPResults(5)] = ttest2(SVMrbf, DTAccuracy);
[ttestBinaryHResults(6), ttestBinaryPResults(6)] = ttest2(DTAccuracy, SVMLinear);
[ttestBinaryHResults(7), ttestBinaryPResults(7)] = ttest2(DTAccuracy, SVMPolynomial);