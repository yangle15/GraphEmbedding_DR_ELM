function [beta,Htr,Htst] = ELMGAE(TrainData, TestData, S, Inputweight, options)

%   Input:
%       TrainData: Training samples, number of sample * Dimension;
%       train_Y: Training labels, number of sample * 1;
%       TestData:  Test samples, number of sample * Dimension;
%       S:       The adjancency matrix, number of sample * number of sample;
%       options: mostly inherit from the outer function.
%   Output:
%        Htr:    The obtained features for training samples;
%        Htst:   The obtained features for test samples;
%        beta:   The optimal beta;

    %%%%    Authors:    LE YANG
    %%%%    TSINGHUA UNIVERSITY, CHINA
    %%%%    EMAIL:      yangle15@mails.tsinghua.edu.cn;
    %%%%    DATE:       Nov. 2017

options.InputDim=size(TrainData,2);

% 将数据映射到随机空间
% Rand_inputweight=rand(options.InputDim,options.NumHiddenNeuron)*2-1;
options.show = 0;
[Htr, ~] = ActiveFunction(TrainData,Inputweight,options);


bestC=options.C;
D = diag(sum(S));
if bestC==0
    beta=(Htr'*Htr)\(Htr'*S*TrainData);
else
    if options.Sparse == 1
        rhohats = mean(Htr,1)';
        rho = 0.05;
        KLsum = sum(rho * log(rho ./ rhohats) + (1-rho) * log((1-rho) ./ (1-rhohats)));

        Hsquare =  Htr' *D* Htr;
        HsquareL = diag(max(Hsquare,[],2));
        beta=( ( eye(size(Htr',1)).*KLsum +HsquareL )*(1/bestC)+Hsquare) \ (Htr'*S*TrainData);
    elseif options.Sparse == 0
        if size(Htr,1)>options.NumHiddenNeuron
            beta=(Htr'*D*Htr+sparse(eye(options.NumHiddenNeuron)/bestC))\(Htr'*S*TrainData);
            if isnan(beta(1,1))
                beta=(Htr'*D*Htr+sparse(eye(options.NumHiddenNeuron)))\(Htr'*S*TrainData);
            end
        else
            beta=Htr'*(((D*Htr*Htr')+sparse(eye(size(Htr,1)))/bestC)\(S*TrainData));
        end
    end
end
beta = full(beta);    

options.show = 1;
Inputweight = beta';
[Htr, options.ps1] = ActiveFunction(TrainData,Inputweight,options);

options.train = 0;
options.show = 0;
[Htst, ~] = ActiveFunction(TestData,Inputweight,options);

end

