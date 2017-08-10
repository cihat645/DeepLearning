function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize); %dimen: 10 x 200

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

%stack{1}.w dimen: 200 x 784, bias: 200 x 784
%stack{2}.w dimen: 200 x 200, bias: 200 x 200

m = columns(data);

a1 = data; %dimen: 784 x 10,000
z1 = (stack{1}.w * data) + repmat(stack{1}.b, 1, m);
a2 = sigmoid(z1); %dimen: 200 x 10,000
z2 = (stack{2}.w * a2) + repmat(stack{2}.b, 1, m);
a3 = sigmoid(z2); %dimen: 200 x 10,000

M = softmaxTheta * a3;
M = bsxfun(@minus, M, max(M, [], 1)); %preventing overflow of sigmoid function
hyp = exp(M);
hyp = bsxfun(@rdivide, hyp, sum(hyp)); %normalizing values


%make prediction: for every example (column), return index with max value
[scrap, pred] = max(hyp, [], 1);




% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
