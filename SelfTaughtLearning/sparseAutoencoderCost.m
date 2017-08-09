function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); %dimen: hidden_size x 1 (ex: 2 x 1)
b2grad = zeros(size(b2)); %dimen: visibleSize x 1 (ex: 64 x 1)


%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 


%why not using repmat on b1 and b2 was a problem:
% because we were taking a 2x10 matrix and trying to add a 2x1 vector (doesn't work)
% we were also trying to take a 64 x 10 matrix and add a 64 x 1 vector (doesn't work because we need the bias unit to be of the same size)

m = columns(data);
a1 = data; %dimen: 64 x 10
z1 = (W1 * a1) + repmat(b1, 1, m); %dimen: # of hidden units x # of examples (ex: 2 x 10)
a2 = sigmoid(z1);
z2 = (W2 * a2) + repmat(b2, 1, m);
a3 = sigmoid(z2); %dimen: # of output units x # of examples (ex: 64 x 10)

cost = sum(sum((data - a3).^2)) / (2 * m);
cost = cost + (lambda / 2) * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
avg_activation = mean(a2,2); %dimen: hidden_units x 1 (ex: 2 x 1)

cost = cost + beta * sum(sparsityParam * log(sparsityParam ./ avg_activation) + (1 - sparsityParam) * log((1 - sparsityParam) ./ (1 - avg_activation)));

%backprop:
delta3 = -(data - a3) .* sigmoid_gradient(z2); %dimen: 64 x 10
sparsity_delta = repmat(beta * ((-sparsityParam ./ avg_activation) + ((1-sparsityParam)./(1-avg_activation))),1, m);
%we use repmat here to create a matrix of sparisty delta to add to every element of (W2' * delta3)
delta2 = (W2' * delta3 + sparsity_delta) .* sigmoid_gradient(z1); %DIMEN: 2 x 10
%delta2 = (W2' * delta3) .* sigmoid_gradient(z1);

W1grad = (delta2 * a1') ./ m + lambda * W1; %dimen: 2 x 64
W2grad = (delta3 * a2') ./ m + lambda * W2; %dimen: 64 x 2
b1grad = mean(delta2,2); 
b2grad = mean(delta3,2);



%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end


function sigm_grad = sigmoid_gradient(z)
  sigm_grad = sigmoid(z) .* ( 1 - sigmoid(z));
end
