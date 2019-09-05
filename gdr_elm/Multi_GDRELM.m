function [TrAcc,TstAcc,H1,H2,Beta] = Multi_GDRELM(endType ,TrainData, TestData, train_Y, test_Y, S, options)

%   Input:
%       endType: 0 for least square method and 1 for classical ELM;  
%       train_X: Training samples, number of sample * Dimension;
%       train_Y: Training labels, number of sample * 1;
%       test_X:  Test samples, number of sample * Dimension;
%       test_Y:  Test labels, number of sample * 1;
%       S:       The adjancency matrix, number of sample * number of sample;
%       options: mostly inherit from the outer function.
%
%   Output:
%        TrAcc:  Training accuracy;
%        TsAcc:  Test accuracy;
%        TrTime; Training time;
%        H1:    The obtained features for training samples;
%        H2:   The obtained features for test samples;
%        beta:   The optimal beta of each layer;

    %%%%    Authors:    LE YANG
    %%%%    TSINGHUA UNIVERSITY, CHINA
    %%%%    EMAIL:      yangle15@mails.tsinghua.edu.cn;
    %%%%    DATE:       Nov. 2017

%% Option Check
if strcmp(options.Kernel,'sigmoid')
    if length(options.LayerNum)~=length(options.SigPara)
          error('The number of layer and length of options.SigPara must be equal !')
    end
end
% if length(options.LayerNum)~=(length(options.C)-1)
%       error('the number of hyperparameter C must be equal to the number of layer plus 1 !')
% end
if (length(options.LayerNum)==1)&&(options.LastLayer==1)
      error('The last layer can not be classical ELM when the number of layer is 1 !')
end
if (options.Sparse == 1)&&(~strcmp(options.Kernel,'sigmoid'))
      error('The kernel should be sigmoid when using the sparse learning. !')
end
%% Preprocessing
D = size(TrainData,2);
suboptions.Kernel = options.Kernel;
options.LayerNum = [D,options.LayerNum];

Ntr = size(TrainData,1);
Ntst = size(TestData,1);
labelSet = unique(train_Y);
K = length(labelSet); % number of classes
temp_T=zeros(K, Ntr);
for i = 1:Ntr
    for j = 1:K
        if labelSet(j) == train_Y(i)
            break; 
        end
    end
    temp_T(j,i)=1;
end
TrainLabel_onehot=temp_T*2-1;

if endType == 0
    flag = 1;
else
    flag = 2;
end

%% Generate the random input
for i = 1:length(options.LayerNum)-flag
    Rand_inputweight{i} = rand(options.LayerNum(i)+1,options.LayerNum(i+1))*2-1;
    if  size(Rand_inputweight{i},1) > size(Rand_inputweight{i},2)   %Using the orthogonal random matrix
        Rand_inputweight{i} = orth(Rand_inputweight{i});
    else
        Rand_inputweight{i} = orth(Rand_inputweight{i}')';
    end
end

%% Train the network
H1 = TrainData;
H2 = TestData;
    
for i = 1:length(options.LayerNum)-flag
    H1 = [H1 .1 * ones(size(H1,1),1)];
    H2 = [H2 .1 * ones(size(H2,1),1)];
    suboptions.C=options.C(i);
    suboptions.SigPara=options.SigPara(i);
    suboptions.InputDim=size(H1,2);
    suboptions.NumHiddenNeuron=options.LayerNum(i+1);
    suboptions.train=1;
    suboptions.Sparse = options.Sparse;
    [B, H1, H2]=ELMGAE(H1,H2, S, Rand_inputweight{i}, suboptions);
    argstr = ['Beta.w',int2str(i),'=B;'];
    eval(argstr);
end
%% ELM layer
if endType == 1

    Htr = [H1 .1 * ones(size(H1,1),1)];
    Htst = [H2 .1 * ones(size(H2,1),1)];
    Rand_inputweight_elm = rand(options.LayerNum(end-1)+1,options.LayerNum(end))*2-1;
    if  size(Rand_inputweight_elm,1) > size(Rand_inputweight_elm,2)   %Using the orthogonal random matrix
        Rand_inputweight_elm = orth(Rand_inputweight_elm);
    else
        Rand_inputweight_elm = orth(Rand_inputweight_elm')';
    end
    options.show = 1;
    Htr = Htr*Rand_inputweight_elm;
    l3 = max(max(Htr)); l3 = 1/l3;
    Htr = tansig(Htr*l3);
    Htst = tansig(Htst*Rand_inputweight_elm*l3);

    if size(Htr,1)>options.LayerNum(end)
        beta=(Htr'*Htr+sparse(eye(options.LayerNum(end))/options.C(end)))\(Htr'*TrainLabel_onehot');
    else
        beta=Htr'*(((Htr*Htr')+sparse(eye(size(Htr,1)))/options.C(end))\TrainLabel_onehot');
    end
    Beta = full(beta);
    
PredTr = Htr*Beta;
[~, label_index_expected]=max(PredTr,[],2);
postive = label_index_expected == train_Y;
TrAcc = sum(postive)/Ntr;

PredTest = Htst*Beta;
[~, label_index_expected]=max(PredTest,[],2);
postive = label_index_expected == test_Y;
TstAcc = sum(postive)/Ntst;

%% least square layer
elseif endType == 0
XX = H1;
XXtst = H2;

BetaF=(XX'*XX+eye(size(XX,2))/options.C(end))\(XX'*TrainLabel_onehot');

PredTr = XX*BetaF;
[~, label_index_expected]=max(PredTr,[],2);
postive = label_index_expected == train_Y;

TrAcc = sum(postive)/length(train_Y);


PredTest = XXtst*BetaF;
[~, label_index_expected]=max(PredTest,[],2);
postive = label_index_expected == test_Y;

TstAcc = sum(postive)/length(test_Y);
end
end
