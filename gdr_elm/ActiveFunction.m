function [H, ps1] = ActiveFunction(X, Inputweight, opt)
ps1=[];
tp = X*Inputweight;
switch opt.Kernel
    case 'minmax'
        if opt.train == 1;
            [H, ps1] = mapminmax((tp)');
            H = H';
        else 
            H = mapminmax('apply',(tp)',opt.ps1);
            H = H';
        end
    case 'tansig'
        if opt.train == 1;
            ps1 = max(max(tp));ps1 = 1/ps1;
            H = tansig(tp*ps1);
        else 
            H = tansig(tp*opt.ps1);
        end    
    case 'relu'
        H=tp;
        H(H<=0)=0;
    case 'sigmoid'
        H=1 ./ (1 + exp(-opt.SigPara*tp));
    case 'linear'
        H=tp;
    case 'radbas'
        H=radbas(X);
    case 'tanh'
        temp=exp(tp);
        H=(temp-1./temp)./(temp+1./temp);
end