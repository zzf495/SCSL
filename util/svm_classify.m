function y=svm_classify(trainData,testData,trainLabel,testLabel)
numLabels=length(unique(trainLabel));
numTest=size(testLabel,1);
trainData = double(trainData);
testData = double(testData);
trainLabel = double(trainLabel);
testLabel = double(testLabel);
%% # get probability estimates of test instances using each model
prob = zeros(numTest,numLabels);
for k=1:numLabels
    model = libsvmtrain(double(trainLabel==k),trainData, '-t 0 -b 1 -q');
    [~,~,p] = libsvmpredict(double(testLabel==k),testData, model, '-b 1 -q');
    prob(:,k) = p(:,model.Label==1);    %# probability of class==k
end
%# predict the class with the highest probability
% [~,pred] = max(prob,[],2);
% acc = sum(pred == testLabel) ./ numel(testLabel);    %# accuracy
%C = confusionmat(testLabel, pred)                   %# confusion matrix
y=prob;
% y=y./sum(y,2);
end