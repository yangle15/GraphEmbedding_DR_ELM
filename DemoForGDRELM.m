clear
addpath('gdr_elm');
%% orl
load ORL_sample; 
num = 5; % this is the number of samples for each class during training

% if zscore the original data
zs = 0; % This hyper-parameter need to be selected

% select activation function of the GDR-ELM
ker = 1; % This hyper-parameter need to be selected

% number of neighborhood 
NN = round(num/2); % This hyper-parameter need to be selected

%% Signle Layer with least square
% target dimensionality
LN = 30;

%------------options--------------
options.Zsocre = zs;
options.NN = NN; % This hyper-parameter need to be selected
options.C=[10,10]; % This hyper-parameter need to be selected
options.SigPara=[1]; % This hyper-parameter need to be selected
if ker==0
    options.Kernel='sigmoid';options.Sparse = 1;
else
    options.Kernel='tansig';options.Sparse = 0;  
end
options.LayerNum=[LN];
options.LastLayer = 0; % choose the last classifier

%---------------GAE---------------
[Train_acc, Test_acc] = GDR_ELM(Pr, Tr, Pt,  Tt, options)


%% Signle Layer with ELM
% target dimensionality
LN = 30;

ELM_HIDDEN_LAYER = 500;
%------------options--------------
options.Zsocre = zs;
options.NN = NN; % This hyper-parameter need to be selected
options.C=[10,10]; % This hyper-parameter need to be selected
options.SigPara=[1]; % This hyper-parameter need to be selected
if ker==0
    options.Kernel='sigmoid';options.Sparse = 1;
else
    options.Kernel='tansig';options.Sparse = 0;  
end
options.LayerNum=[LN, ELM_HIDDEN_LAYER];
options.LastLayer = 1; % choose the last classifier

%---------------GAE---------------
[Train_acc, Test_acc] = GDR_ELM(Pr, Tr, Pt,  Tt, options)



%% Multi Layer (Three layers) with least square
% target dimensionality
LN = 30;

% two Hidden Layers 
HN = [70 50];
%------------options--------------
options.Zsocre = zs;
options.NN = NN;
options.C=[10,10,10,10]; % This hyper-parameter need to be selected
options.SigPara=[1,1,1]; % This hyper-parameter need to be selected
if ker==0
    options.Kernel='sigmoid';options.Sparse = 1;
else
    options.Kernel='tansig';options.Sparse = 0;  
end
options.LayerNum=[HN,LN];
options.LastLayer = 0;

%---------------GAE---------------
[Train_acc, Test_acc] = GDR_ELM(Pr, Tr, Pt,  Tt, options)
