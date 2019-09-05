function [TrAcc, TsAcc, TrTime, beta,Htr,Htst] = GDR_ELM(train_X,train_Y,test_X,test_Y, options)

%   Input:
%       train_X: Training samples, number of sample * Dimension;
%       train_Y: Training labels, number of sample * 1;
%       test_X:  Test samples, number of sample * Dimension;
%       test_Y:  Test labels, number of sample * 1;
%       options: function options:
%              options.NN:       Number of neighborhood when generate the adjacency matrix
%              options.C:        Hyperparameters controling the regualrization in each layer, with the formulation [C(1),C(2),..,C(K),C(k+1)];
%                                It should be mentioned that the last C(K+1) is used for the classification layer.
%              options.Kernel:   The selection of activation function, it can be 
%                                'sigmoid'--- the sigmoid activation function;
%                                'tansig' --- the tansig activation function;
%                                'linear' --- the linear activation function;
%                                'relu'   --- the RELU activation function;
%              options.SigPara:  The parameter control the sigmoid function when using the 'sigmoid' kernel.;
%                                With the formulation [SigPara(1),SigPara(2),..,SigPara(K)];
%              options.LayerNum: The number of hidden nodes in each layer; With the formulation [LN(1),LN(2),..];
%                                When options.LastLayer = 0, LN(K) is the target dimensionality.
%                                When options.LastLayer = 0, LN(K-1) is the target dimensionality,
%                                and the LN(K) is the number of hidden node in the classical ELM.
%              options.LastLayer:The option of classification layer. 
%                                0 for least square method and 1 for classical ELM; 
%              options.Sparse:   The option for sparse learning.  1 for sparse learning; 
%                                The kernel should be 'sigmoid' when using the sparse learning.
%              options.zsocre:   The option for conducting the Z trainsformation of the data. 1 for true.
%
%   Output:
%        TrAcc:  Training accuracy;
%        TsAcc:  Test accuracy;
%        TrTime; Training time;
%        beta:   The optimal beta of each layer;
%        Htr:    The obtained features for training samples;
%        Htst:   The obtained features for test samples;


    %%%%    Authors:    LE YANG
    %%%%    TSINGHUA UNIVERSITY, CHINA
    %%%%    EMAIL:      yangle15@mails.tsinghua.edu.cn;
    %%%%    DATE:       Nov. 2017

    % Preprocessing
    if options.Zsocre == 1
        train_X = zscore(train_X')';
        test_X = zscore(test_X')';
    end
    % Adjancency matrix setting
    optionsL.NN = options.NN ;
    optionsL.GraphDistanceFunction = 'euclidean';
    optionsL.GraphWeights = 'heat';
    optionsL.GraphWeightParam=0;
    optionsL.LaplacianNormalize = 0;
    optionsL.LaplacianDegree = 1;
    optionsL.dis = 1;
    optionsL.self = 1;
    optionsL.selfFactor = optionsL.NN/2;
    optionsL.weightType = 'GEN';
    optionsL.GEN_LDA  = 0;

    % Generating the adjancency matrix 
    tic;
    S_C = GraphGenerate(train_X,train_Y,optionsL);
%     S_C = S_C + eye(size(S_C));
    time1 = toc;

    options.InputDim=size(train_X,2);
    tic;
    %GDR-ELM
    [TrAcc,TsAcc,Htr,Htst,beta] = Multi_GDRELM(options.LastLayer,train_X,test_X, train_Y, test_Y, S_C, options);
    time2 = toc;
    TrTime = time1 + time2;
end