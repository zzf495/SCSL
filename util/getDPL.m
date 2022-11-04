function [probYt,trustable,predLabels] = getDPL(Zs,Ys,Zt,Ytpseudo,lastPredLabels,pos,selectRate)
%% input
%%%     Xs              The source sample set with m * ns
%%%     Ys              The source labels of Xs with ns * 1
%%%     Xt              The target sample set with m * nt
%%%     Ytpseudo        The pseudo labels of Xt with nt * 1
%%%     pos             The number that indicates the top-k probability
%%%     selectRate      The rate of the selection, 0 < selectRate <= 1
%% output
%%%     probYt          The top-k probability
%%%     trustable       The indicator matrix that indicates which sample is selected
%%%                     -   1      The sample is selected
%%%                     -   0      The sample is not selected
%%%     predLabels      The pseudo labels of Xt in this iteration
C=length(unique(Ys));
selectRate=max(0,min(selectRate,1));
%% SVM classification
probMatrix=svm_classify(Zs',Zt',Ys,Ytpseudo);
% The highest probability
[prob,predLabels]=max(probMatrix,[],2);
% The second-highest probability
prob2=probMatrix;
prob2(prob2==prob)=0;
[prob2,~]=max(prob2,[],2);
prob2=prob2';
prob=prob';predLabels=predLabels';
%% Run DPL
prob=prob-prob2;
[sortedProb,index] = sort(prob);
sortedPredLabels = predLabels(index);
trustable = zeros(1,length(prob));
for i = 1:C
    thisClassProb = sortedProb(sortedPredLabels==i);
    if ~isempty(thisClassProb)
        idx=min((floor(length(thisClassProb)*selectRate)+1),length(thisClassProb));
        trustable = trustable+ ((prob>thisClassProb(idx)).*(predLabels==i));
    end
end
if ~isempty(lastPredLabels)
    trustable(lastPredLabels~=Ytpseudo)=0;
end
%% Get top-k probability
[a,~]=sort(probMatrix,2);
probs=a(:,pos);
probs(trustable==1)=a(trustable==1,end);
probMatrix(probMatrix<probs)=0;
probMatrix=probMatrix./sum(abs(probMatrix),2);
%% Output the results
probYt=probMatrix;
trustable=trustable';
predLabels=predLabels';
end
