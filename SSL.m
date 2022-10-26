function [acc,acc_ite,A]= SSL(Xs,Ys,Xt,RealYt,options)
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
%% Version
%%%         Initialization                  2022-08-10
options=defaultOptions(options,...
        'T',10,...              % The iterations
        'dim',100,...           % The dimension reduced
        'alpha',1,...           % The weight of manifold regularization
        'beta',1,...            % The weight of discrimination
        'sC',1,...              % The fuzzy number
        'kernel_type',0,...     % Kernel parameter
        'gamma',1,...           % The hyper-parameter of Kernel
        'lambda',0.1);          % The regularization term
acc_ite=[];
%% Parameters setting
dim=options.dim;
alpha=options.alpha;
beta=options.beta;
lambda = options.lambda;
ker = options.kernel_type;
gamma = options.gamma;
num_iter = options.T;
sC=options.sC;
%% Initialization
[X,ns,nt,n,m,C] = datasetMsg(Xs,Ys,Xt,1);
if (~strcmpi(ker,'primal')) && ker~=0
    X=kernelProject(ker,X,[],gamma);
    Xs=X(:,1:ns);
    Xt=X(:,ns+1:end);
    m=n;
end
% Init target pseudo labels
Yshot=hotmatrix(Ys,C,0);
Ytpseudo=classifySVM(Xs,Ys,Xt);
predLabels=Ytpseudo;
acc=getAcc(Ytpseudo,RealYt);
fprintf('init acc:%.4f\n',acc);
YsHotSameWeight=hotmatrix(Ys,C,1);
% Init U_s * G_s by Eq.(8)
Fs=Xs*YsHotSameWeight; % m * c
cWeight=zeros(1,C);
maxC=inf;
for i=1:C
    cWeight(i)=1/(ns-length(find(Ys==i)));
    maxC=min(maxC,length(find(Ys==i)));
end
% Init \hat{U}_s * G_s by Eq.(8)
cWeight=ones(ns,1)*cWeight;
YsHotDiffWeight=((1-Yshot).*cWeight);
Frs=Xs*YsHotDiffWeight;
% Set default value
Sw=0;
Sw2=0;
for iter = 1:num_iter
    if iter>1
        % Solve Eqs.(8)
        tmpT=Xt-Fs*probYt';
        Sw=tmpT*tmpT';
        Xtc=Xt(:,logical(trustable));
        % Solve Eqs.(10)-(11)
        hotYt=hotmatrix(predLabels(logical(trustable)),C,0);
        tmpT2=Xtc-Frs*hotYt';
        Sw2=tmpT2*tmpT2';
    end
    % Solve Eq.(5)
    if iter>1
        % Use [Xs,X_{t,tr}]
        Ymain=[Ys;predLabels(logical(trustable))];
        XL=[Xs,Xt(:,logical(trustable))];
    else
       % Use Xs in 1st iteration
       Ymain=Ys;
       XL=Xs;
    end
    XL= L2Norm(XL')';
    manifold.Metric='Cosine';
    manifold.WeightMode='Binary';
    manifold.NeighborMode='Supervised';%'Supervised';
    manifold.gnd=Ymain;
    manifold.normr=1;
    manifold.k=0;
    [Ls,D,~]=computeL(XL,manifold);
    Ls=XL*Ls*XL';
    Ds=XL*D*XL';
    % Solve Eq.(12)
    [A,~]=eigs(Ls+lambda*eye(m)+alpha*(Sw),beta*Sw2+1e-6*Ds,dim,'sm');
    Zs=A'*Xs;
    Zt=A'*Xt;
    % Solve DPL and get probability
    pos=C-sC+1;
    p=1-(iter/num_iter);
    [probYt,trustable,predLabels] = getDPL(Zs,Ys,Zt,predLabels,pos,p);
    % calculate ACC
    acc=getAcc(predLabels,RealYt);
    acc_ite(iter)=acc;
    fprintf('Iteration=%d, Acc:%0.3f\n', iter, acc);
end
end