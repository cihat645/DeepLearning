%% CS294A/CS294W Self-taught Learning Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises.
%
%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize  = 28 * 28;
numLabels  = 5;
hiddenSize = 200;
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term   
maxIter = 400;

%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

%NOTE: to access the first handwritten number stored in mnistData, you type 'mnistData(:, 1)'
% so the columns hold all of the pixel intensity values 
% the rows correspond to separate images

% Load MNIST database files
mnistData   = loadMNISTImages('train-images.idx3-ubyte');
mnistLabels = loadMNISTLabels('train-labels.idx1-ubyte');



% Simulate a Labeled and Unlabeled set
labeledSet   = find(mnistLabels >= 0 & mnistLabels <= 4); %first we find all the indices with labels between 0 and 4 for our labeled set
unlabeledSet = find(mnistLabels >= 5); %find all indices of mnistLabels for unlabeled set (with labels 5 and above)
printf('size of unlabeledSet\n');
size(unlabeledSet)
unlabeledSet = unlabeledSet(randperm(10000), :);

numTrain = round(numel(labeledSet)/2); %number of training examples = half the # of elements in the labeledSet
trainSet = labeledSet(1:numTrain); %create our training labels data structure 
testSet  = labeledSet(numTrain+1:end); %create our test labels data structure
%the vars above store indices, not labels

unlabeledData = mnistData(:, unlabeledSet); %create matrix to store the pixel values for every image labeled as 5 and above

trainData   = mnistData(:, trainSet);
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-5

testData   = mnistData(:, testSet);
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-5

% Output Some Statistics
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));


%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  images. 

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, inputSize);

%% ----------------- YOUR CODE HERE ----------------------
%  Find opttheta by running the sparse autoencoder on
%  unlabeledTrainingImages

%opttheta = theta; %this was here initially

options.HessUpate = 'lbfgs';
options.MaxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.Display = 'iter';
options.GradObj = 'on';

%printf('training....\n');
%[opttheta, cost] = fminlbfgs( @(p) sparseAutoencoderCost(p, ...
%                                   inputSize, hiddenSize, ...
%                                   lambda, sparsityParam, ...
%                                   beta, unlabeledData), ...
%                              theta, options);


%save('opttheta.mat', 'opttheta');


%% -----------------------------------------------------

load('opttheta.mat')
                          
% Visualize weights
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
display_network(W1');

printf('Training COMPLETE\n')

%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  
%  You need to complete the code in feedForwardAutoencoder.m so that the 
%  following command will extract features from the data.

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData); %using trainData & the 
% learned parameters from the autoencoder, extract the activations (features) learned
% for the trainData and testData 

testFeatures = (feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData))';

                            
                                       
%%======================================================================
%% STEP 4: Train the softmax classifier

softmaxModel = struct;  
%% ----------------- YOUR CODE HERE ----------------------
%  Use softmaxTrain.m from the previous exercise to train a multi-class
%  classifier. 

%  Use lambda = 1e-4 for the weight regularization for softmax

% You need to compute softmaxModel using softmaxTrain on trainFeatures and
% trainLabels

trainFeatures = trainFeatures';
trainLabels = trainLabels';
printf('size of X and y: \n');
size(trainFeatures)
size(trainLabels)

lambda = 1e-4;
[all_theta] = oneVsAllOG(trainFeatures, trainLabels, numLabels, lambda);
printf('success\n');

%% -----------------------------------------------------

%%======================================================================
%% STEP 5: Testing 

%% ----------------- YOUR CODE HERE ----------------------
% Compute Predictions on the test set (testFeatures) using softmaxPredict
% and softmaxModel

pred = predictOneVsAll(all_theta, testFeatures);

%% -----------------------------------------------------

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

% (note that we shift the labels by 1, so that digit 0 now corresponds to
%  label 1)
%
% Accuracy is the proportion of correctly classified images
% The results for our implementation was:
%
% Accuracy: 98.3%
%
% 

% we're able to train a softmax classifier using the digits 5-9 to recongize the 
% digits 0 - 4 because of the learned features we attain from utilizing an autoencoder
% these learned features are penstrokes. In the regular softmax classifier exercise,
% the features were just pixels. Here, the features are penstrokes, allowing for a 
% more accurate prediction and overall, a better learning algorithm.

