%% Implementation of SSL
%% Information
%%%         Selected Sample Labelling for Domain Adaptation
%%%         Author          ZeFeng Zheng et al.
%% Input
%%%         T               The iteration times
%%%         dim             The dimension of the projection subspace
%%%         alpha           The weight of manifold regularization
%%%         beta            The weight of discrimination
%%%         sC              The fuzzy number
%%%         lambda          The regularization term
%%%         kernel_type     Kernel parameter
%%%         gamma           The hyper-parameter of Kernel
%% Output
%%%         acc             The classification accuracy
%%%         acc_ite         The classification accuracies of iterations
%%%         A               The learned projection matrix
clc; clear all;
addpath(genpath('./util/'));
srcStr = {'caltech','caltech','caltech','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgtStr = {'amazon','webcam','dslr','caltech','webcam','dslr','caltech','amazon','dslr','caltech','amazon','webcam'};
result=[];
accIteration=[];
%% Parameter Setting
options= defaultOptions(struct(),...
                'T',11,...              % The iteration times
                'dim',64,...            % The dimension of the projection subspace
                'alpha',0.2,...         % The weight of manifold regularization
                'beta',5,...            % The weight of discrimination
                'sC',10,...             % The fuzzy number
                'kernel_type',0,...     % Kernel
                'gamma',1,...           % The hyper-parameter of Kernel
                'lambda',1);            % The regularization term
optsPCA.ReducedDim=512;                 % The dimension reduced before training
for i = 1:12
    %% Load data
    src = char(srcStr{i});
    tgt = char(tgtStr{i});
    fprintf('%d: %s_vs_%s\n',i,src,tgt);
    load(['./data/' src '_decaf.mat']);
    feas = feas ./ repmat(sum(feas,2),1,size(feas,2));
    Xs=feas';
    Ys = labels;
    load(['./data/' tgt '_decaf.mat']);
    feas = feas ./ repmat(sum(feas,2),1,size(feas,2));
    Xt=feas';
    Yt = labels;
    %% Run PCA to reduce the dimensionality
    domainS_features_ori=Xs';domainT_features=Xt';
    X = double([domainS_features_ori;domainT_features]);
    P_pca = PCA(X,optsPCA);
    domainS_features = domainS_features_ori*P_pca;
    domainT_features = domainT_features*P_pca;
    %% Run SSL
    Xs=L2Norm(domainS_features)';
    Xt=L2Norm(domainT_features)';
    [acc,acc_ite]=SSL(Xs,Ys,Xt,Yt,options);
    accIteration=[accIteration;acc_ite];
    result(i)=acc;
end
fprintf('Mean accuracy: %.4f\n',mean(result));
