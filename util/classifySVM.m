function Ytpseudo=classifySVM(Xs,Ys,Xt)
     nt=size(Xt,2);
     Y=zeros(nt,1);
     svmmodel = train(double(Ys), sparse(double(Xs')),'-s 1 -B 1.0 -q');
     [Ytpseudo,~,~] = predict(Y, sparse(Xt'), svmmodel,'-q');
end